//! The Planner driven by a plaintext engine.
//!
//! Sharings are linear, so a plaintext value is a valid sharing (a degree-0
//! polynomial): an engine where `Multiply` is the product, `Reveal` the
//! identity and `MaskedMultiply` is `x·y + mask` exercises every line of the
//! Planner except the network. Each op is checked against its integer
//! reference, at ℓ = 61 and ℓ = 31.

use std::collections::VecDeque;

use anyhow::Result;
use planner::api::engine::{Application, DepthInput, RandomWireShares};
use async_trait::async_trait;
use fields::{mersenne_31::Mersenne31Field, MersennePrimeField, Mersenne61Field, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use planner::{OpParams, Op, OpDepthInput, OpType, OpResult, Planner, PlannerApplication, PlannerCounts};

type E<F> = FieldElement<F>;

// ---------------------------------------------------------------------------
// Integer view of the field
// ---------------------------------------------------------------------------

fn from_signed<F: ProtocolField + MersennePrimeField>(v: i128) -> E<F> {
    if v >= 0 {
        E::<F>::from(v as u64)
    } else {
        -E::<F>::from((-v) as u64)
    }
}

fn to_signed<F: ProtocolField + MersennePrimeField>(e: &E<F>) -> i128 {
    let c = F::to_canonical_u64(e) as i128;
    let p = F::MODULUS as i128;
    if c <= (p - 1) / 2 { c } else { c - p }
}

/// Liu et al.'s `Trunc_d` on a signed integer: shift the magnitude, keep
/// the sign.
fn trunc_ref(v: i128, d: usize) -> i128 {
    if v >= 0 { v >> d } else { -((-v) >> d) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `[-bound, bound)`.
    fn signed(&mut self, bound: i128) -> i128 {
        (self.next() as i128 % (2 * bound)) - bound
    }
}

// ---------------------------------------------------------------------------
// A scripted PlannerApplication
// ---------------------------------------------------------------------------

type Scene<F> = Box<dyn FnMut(&[OpResult<F>]) -> OpDepthInput<F> + Send>;

struct Script<F: ProtocolField> {
    counts: PlannerCounts,
    scenes: VecDeque<Scene<F>>,
    results: Vec<OpResult<F>>,
    outputs: Option<Vec<E<F>>>,
}

impl<F: ProtocolField> Script<F> {
    fn new(counts: PlannerCounts) -> Self {
        Self { counts, scenes: VecDeque::new(), results: Vec::new(), outputs: None }
    }

    fn then(mut self, scene: impl FnMut(&[OpResult<F>]) -> OpDepthInput<F> + Send + 'static) -> Self {
        self.scenes.push_back(Box::new(scene));
        self
    }

    fn play(&mut self) -> OpDepthInput<F> {
        match self.scenes.pop_front() {
            Some(mut scene) => scene(&self.results),
            None => OpDepthInput::Done(Vec::new()),
        }
    }
}

#[async_trait]
impl<F: ProtocolField> PlannerApplication<F> for Script<F> {
    fn preprocessing_count(&self) -> PlannerCounts {
        self.counts.clone()
    }

    async fn input_sharing_termination(&mut self, _party: usize, _shares: Vec<E<F>>) -> Result<OpDepthInput<F>> {
        Ok(OpDepthInput::Waiting)
    }

    async fn on_preprocessing_complete(&mut self, _wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        Ok(self.play())
    }

    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        assert_eq!(depth, self.results.len() + 1, "op-depths complete in order");
        self.results.push(result);
        Ok(self.play())
    }

    async fn on_output(&mut self, outputs: Vec<E<F>>) -> Result<()> {
        self.outputs = Some(outputs);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The plaintext engine
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq, Clone)]
enum Batch {
    Multiply(usize, usize),
    Reveal(usize, usize),
    Masked(usize, usize),
}

async fn run<F: ProtocolField, A: Application<F>>(app: &mut A, seed: u64) -> Vec<Batch> {
    let mut rng = Rng(seed | 1);
    let wires = app.random_wires();
    let bits: Vec<E<F>> = (0..wires.bits).map(|_| if rng.next() & 1 == 1 { E::<F>::one() } else { -E::<F>::one() }).collect();
    let sharings: Vec<E<F>> = (0..wires.sharings).map(|_| E::<F>::from(rng.next() >> 3)).collect();
    let mut batches = Vec::new();
    let mut next = app.on_preprocessing_complete(RandomWireShares::new(bits, sharings)).await.unwrap();
    loop {
        next = match next {
            DepthInput::Waiting => panic!("the script stalled"),
            DepthInput::Done(outputs) => {
                app.on_output(outputs).await.unwrap();
                return batches;
            }
            DepthInput::Multiply { depth, x, y } => {
                batches.push(Batch::Multiply(depth, x.len()));
                let products = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect();
                app.on_depth_complete(depth, products).await.unwrap()
            }
            DepthInput::Reveal { depth, values } => {
                batches.push(Batch::Reveal(depth, values.len()));
                app.on_reveal_complete(depth, values).await.unwrap()
            }
            DepthInput::MaskedMultiply { depth, x, y, mask } => {
                batches.push(Batch::Masked(depth, x.len()));
                let opened = x.iter().zip(y.iter()).zip(mask.iter()).map(|((a, b), m)| a * b + m).collect();
                app.on_reveal_complete(depth, opened).await.unwrap()
            }
        };
    }
}

fn vec<F: ProtocolField + MersennePrimeField>(values: &[i128]) -> Vec<E<F>> {
    values.iter().map(|v| from_signed::<F>(*v)).collect()
}

fn signed_all<F: ProtocolField + MersennePrimeField>(r: &OpResult<F>) -> Vec<i128> {
    match r {
        OpResult::Shares(s) | OpResult::Public(s) => s.iter().map(to_signed::<F>).collect(),
        OpResult::Masked { public, .. } => public.iter().map(to_signed::<F>).collect(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn comparison_family<F: ProtocolField + MersennePrimeField>(pairs: Vec<(i128, i128)>, seed: u64) {
    let n = pairs.len();
    let a: Vec<i128> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<i128> = pairs.iter().map(|p| p.1).collect();
    let types = [OpType::Compare, OpType::ComparePub, OpType::Max, OpType::Min, OpType::MaxPub, OpType::MinPub];
    let ops = types.iter().map(|t| OpParams::new(*t, n)).collect();
    let mut script = Script::<F>::new(PlannerCounts::new(ops, 0));
    for (i, op_type) in types.iter().enumerate() {
        let (a2, b2, op_type) = (a.clone(), b.clone(), *op_type);
        script = script.then(move |_| {
            let op = match op_type {
                OpType::Compare => Op::Compare { a: vec(&a2), b: vec(&b2) },
                OpType::ComparePub => Op::ComparePub { a: vec(&a2), c: vec(&b2) },
                OpType::Max => Op::Max { a: vec(&a2), b: vec(&b2) },
                OpType::Min => Op::Min { a: vec(&a2), b: vec(&b2) },
                OpType::MaxPub => Op::MaxPub { a: vec(&a2), c: vec(&b2) },
                OpType::MinPub => Op::MinPub { a: vec(&a2), c: vec(&b2) },
                _ => unreachable!(),
            };
            OpDepthInput::Op { depth: i + 1, op }
        });
    }
    let mut planner = Planner::new(script).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, seed));
    let rounds: usize = (1..=6).map(|d| planner.plan().op_depth(d).unwrap().rounds()).sum();
    assert_eq!(batches.len(), rounds);

    let results = &planner.app().results;
    let want: Vec<Vec<i128>> = vec![
        pairs.iter().map(|(a, b)| (a < b) as i128).collect(),
        pairs.iter().map(|(a, b)| (a < b) as i128).collect(),
        pairs.iter().map(|(a, b)| *a.max(b)).collect(),
        pairs.iter().map(|(a, b)| *a.min(b)).collect(),
        pairs.iter().map(|(a, b)| *a.max(b)).collect(),
        pairs.iter().map(|(a, b)| *a.min(b)).collect(),
    ];
    for (i, op_type) in types.iter().enumerate() {
        assert_eq!(signed_all(&results[i]), want[i], "{op_type:?} at ℓ={}", F::BITS);
    }
}

#[test]
fn comparison_family_exhaustive_on_small_values() {
    let pairs: Vec<(i128, i128)> = (-6..=6).flat_map(|a| (-6..=6).map(move |b| (a, b))).collect();
    comparison_family::<Mersenne61Field>(pairs.clone(), 1);
    comparison_family::<Mersenne31Field>(pairs, 1);
}

#[test]
fn comparison_family_randomised_over_the_domain() {
    fn go<F: ProtocolField + MersennePrimeField>(seed: u64) {
        let bound = 1i128 << (F::BITS - 2);
        let mut rng = Rng(seed);
        let mut pairs: Vec<(i128, i128)> = (0..200).map(|_| (rng.signed(bound), rng.signed(bound))).collect();
        pairs.extend([(bound - 1, -bound), (-bound, bound - 1), (bound - 1, bound - 1), (0, 0), (-1, 0), (0, -1), (-bound, -bound)]);
        comparison_family::<F>(pairs, seed);
    }
    go::<Mersenne61Field>(0x1234);
    go::<Mersenne31Field>(0x5678);
}

fn truncation<F: ProtocolField + MersennePrimeField>(seed: u64) {
    let ell = F::BITS;
    let bound = 1i128 << (ell - 2);
    let mut rng = Rng(seed);
    for d in [1usize, 8, 16, ell - 3] {
        let mut xs: Vec<i128> = (0..64).map(|_| rng.signed(bound)).collect();
        xs.extend([0, 1, -1, bound - 1, -bound, (1 << d) - 1, -((1 << d) - 1), 1 << d, -(1 << d)]);
        // Factors whose product stays in the domain.
        let half = 1i128 << ((ell - 2) / 2);
        let pairs: Vec<(i128, i128)> = (0..64).map(|_| (rng.signed(half), rng.signed(half))).collect();
        let (x2, p2) = (xs.clone(), pairs.clone());
        let ops = vec![OpParams::new(OpType::Truncate, xs.len()), OpParams::new(OpType::FixedMul, pairs.len())];
        let script = Script::<F>::new(PlannerCounts::new(ops, 0))
            .then(move |_| OpDepthInput::Op { depth: 1, op: Op::Truncate { x: vec(&x2), d } })
            .then(move |_| OpDepthInput::Op {
                depth: 2,
                op: Op::FixedMul {
                    x: vec(&p2.iter().map(|p| p.0).collect::<Vec<_>>()),
                    y: vec(&p2.iter().map(|p| p.1).collect::<Vec<_>>()),
                    d,
                },
            });
        let mut planner = Planner::new(script).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let batches = rt.block_on(run(&mut planner, seed ^ d as u64));
        assert_eq!(batches, vec![Batch::Reveal(1, xs.len()), Batch::Masked(2, pairs.len())]);

        let results = &planner.app().results;
        for (x, got) in xs.iter().zip(signed_all(&results[0])) {
            let want = trunc_ref(*x, d);
            assert!((got - want).abs() <= 2, "Trunc_{d}({x}) = {got}, want {want} ± 2 at ℓ={ell}");
        }
        for ((x, y), got) in pairs.iter().zip(signed_all(&results[1])) {
            let want = trunc_ref(x * y, d);
            assert!((got - want).abs() <= 2, "FixedMul_{d}({x}, {y}) = {got}, want {want} ± 2 at ℓ={ell}");
        }
    }
}

#[test]
fn truncation_and_fixed_point_multiplication_within_two() {
    truncation::<Mersenne61Field>(0xabcd);
    truncation::<Mersenne31Field>(0xef01);
}

#[test]
fn reveal_mask_reveal_mul_and_add() {
    type F = Mersenne61Field;
    let xs = vec![5i128, -7, 1 << 40, -(1 << 50)];
    let ys = vec![3i128, 3, 2, -2];
    let (x1, x2, x3, x4) = (xs.clone(), xs.clone(), xs.clone(), xs.clone());
    let (y3, y4) = (ys.clone(), ys.clone());
    let ops = vec![
        OpParams::new(OpType::Reveal, 4),
        OpParams::new(OpType::MaskReveal, 4),
        OpParams::new(OpType::Mul, 4),
        OpParams::new(OpType::Add, 4),
    ];
    let script = Script::<F>::new(PlannerCounts::new(ops, 0))
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::Reveal { x: vec(&x1) } })
        .then(move |_| OpDepthInput::Op { depth: 2, op: Op::MaskReveal { x: vec(&x2) } })
        .then(move |_| OpDepthInput::Op { depth: 3, op: Op::Mul { x: vec(&x3), y: vec(&y3) } })
        .then(move |_| OpDepthInput::Op { depth: 4, op: Op::Add { x: vec(&x4), y: vec(&y4) } });
    let mut planner = Planner::new(script).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, 7));
    assert_eq!(batches, vec![Batch::Reveal(1, 4), Batch::Reveal(2, 4), Batch::Multiply(3, 4)]);

    let results = &planner.app().results;
    assert!(matches!(results[0], OpResult::Public(_)));
    assert_eq!(signed_all(&results[0]), xs);
    let OpResult::Masked { public, mask } = &results[1] else { panic!("MaskReveal returns Masked") };
    for ((c, r), x) in public.iter().zip(mask.iter()).zip(xs.iter()) {
        assert_eq!(to_signed::<F>(&(c - &r.value)), *x, "c − r recovers x");
        assert_eq!(r.bits.len(), 61);
    }
    assert_eq!(signed_all(&results[2]), xs.iter().zip(ys.iter()).map(|(x, y)| x * y).collect::<Vec<_>>());
    assert_eq!(signed_all(&results[3]), xs.iter().zip(ys.iter()).map(|(x, y)| x + y).collect::<Vec<_>>());
}

/// Over a non-Mersenne field — here the degree-4 extension of Mersenne-61,
/// the engine's default — the Planner runs `Mul`, `Add` and `Reveal`, asks
/// for no random bits of its own, and passes the application's through; a
/// declaration with a Mersenne-only op is refused when the Planner is built.
#[test]
fn a_non_mersenne_field_runs_mul_add_and_reveal() {
    type F = fields::DefaultField;
    let e = |v: u64| E::<F>::from(v);
    let counts = PlannerCounts::new(
        vec![OpParams::new(OpType::Mul, 3), OpParams::new(OpType::Add, 3), OpParams::new(OpType::Reveal, 3)],
        0,
    )
    .with_random_wires(5, 2);
    let script = Script::<F>::new(counts)
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::Mul { x: vec![e(2), e(3), e(4)], y: vec![e(5), e(6), e(7)] } })
        .then(move |r| OpDepthInput::Op { depth: 2, op: Op::Add { x: r[0].clone_shares(), y: vec![e(1), e(1), e(1)] } })
        .then(move |r| OpDepthInput::Op { depth: 3, op: Op::Reveal { x: r[1].clone_shares() } });
    let mut planner = Planner::new(script).unwrap();
    assert_eq!(planner.random_wires(), planner::RandomWires::new(5, 2), "the app's wires, nothing of the Planner's");
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, 11));
    assert_eq!(batches, vec![Batch::Multiply(1, 3), Batch::Reveal(2, 3)]);
    let OpResult::Public(opened) = &planner.app().results[2] else { panic!("Reveal returns Public") };
    assert_eq!(opened, &vec![e(11), e(19), e(29)]);

    for op_type in [OpType::Compare, OpType::MinPub, OpType::Truncate, OpType::FixedMul, OpType::MaskReveal] {
        let script = Script::<F>::new(PlannerCounts::new(vec![OpParams::new(op_type, 1)], 0));
        let err = Planner::new(script).err().expect("refused").to_string();
        assert!(err.contains("Mersenne"), "{op_type:?}: {err}");
    }
}

/// A circuit over several op-depths, with a prefix run (fewer elements
/// than declared), an Add-only depth, and the plan checked
/// against the batches actually run.
#[test]
fn multi_depth_circuit_follows_the_plan() {
    type F = Mersenne61Field;
    let counts = PlannerCounts::new(
        vec![
            OpParams::new(OpType::Mul, 3),
            OpParams::new(OpType::Add, 2),
            OpParams::new(OpType::Compare, 4),
            OpParams::new(OpType::Truncate, 1),
        ],
        1,
    );
    let script = Script::<F>::new(counts)
        .then(|_| OpDepthInput::Op { depth: 1, op: Op::Mul { x: vec(&[2, 3, 4]), y: vec(&[5, 6, 7]) } })
        .then(|r| {
            let p = r[0].clone_shares(); // [10, 18, 28]
            OpDepthInput::Op { depth: 2, op: Op::Add { x: p[..2].to_vec(), y: vec(&[1, 1]) } }
        })
        .then(|r| {
            let s = r[1].clone_shares(); // [11, 19]; compare 2 of the declared 4
            OpDepthInput::Op { depth: 3, op: Op::Compare { a: s, b: vec(&[15, 15]) } }
        })
        .then(|_| OpDepthInput::Op { depth: 4, op: Op::Truncate { x: vec(&[1000]), d: 4 } })
        .then(|r| OpDepthInput::Done(r[3].clone_shares()));
    let mut planner = Planner::new(script).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, 99));

    // Depth 1: one multiply. Depth 2: nothing. Depth 3: reveal, 6 tree
    // levels for 2 comparisons, xor. Depth 4: reveal.
    let widths = [30, 29, 15, 7, 3, 1];
    let mut want = vec![Batch::Multiply(1, 3), Batch::Reveal(2, 2)];
    want.extend(widths.iter().enumerate().map(|(i, w)| Batch::Multiply(3 + i, w * 2)));
    want.push(Batch::Multiply(9, 2));
    want.push(Batch::Reveal(10, 1));
    assert_eq!(batches, want);

    let plan = planner.plan();
    assert_eq!(plan.engine_depths(), 10);
    assert_eq!(plan.preprocessing_counts().gates_per_depth, vec![3, 0, 30 * 4, 29 * 4, 15 * 4, 7 * 4, 3 * 4, 4, 4, 0]);

    let results = &planner.app().results;
    assert_eq!(signed_all(&results[0]), vec![10, 18, 28]);
    assert_eq!(signed_all(&results[1]), vec![11, 19]);
    assert_eq!(signed_all(&results[2]), vec![1, 0]);
    let t = signed_all(&results[3])[0];
    assert!((t - 62).abs() <= 2, "Trunc_4(1000) = {t}");
    assert_eq!(planner.app().outputs.as_ref().unwrap().len(), 1);
}

/// Scheduling what was not declared is an error at the hook, not a hang.
#[test]
fn undeclared_ops_are_refused() {
    type F = Mersenne61Field;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let refuse = |params: OpParams, depth: usize, op: Op<F>, message: &str| {
        let script = Script::<F>::new(PlannerCounts::new(vec![params], 0)).then(move |_| OpDepthInput::Op { depth, op: op_clone(&op) });
        let mut planner = Planner::new(script).unwrap();
        let err = rt.block_on(planner.on_preprocessing_complete(RandomWireShares::new(Vec::new(), Vec::new()))).unwrap_err();
        assert!(err.to_string().contains(message), "{err}");
    };
    refuse(OpParams::new(OpType::Mul, 1), 1, Op::Compare { a: vec(&[1]), b: vec(&[2]) }, "declared");
    refuse(OpParams::new(OpType::Mul, 1), 1, Op::Mul { x: vec(&[1, 1]), y: vec(&[2, 2]) }, "declared");
    refuse(OpParams::new(OpType::Mul, 1), 2, Op::Mul { x: vec(&[1]), y: vec(&[2]) }, "in order");
}

fn op_clone<F: ProtocolField + MersennePrimeField>(op: &Op<F>) -> Op<F> {
    match op {
        Op::Mul { x, y } => Op::Mul { x: x.clone(), y: y.clone() },
        Op::Compare { a, b } => Op::Compare { a: a.clone(), b: b.clone() },
        _ => unimplemented!("only what this test clones"),
    }
}

// Small conveniences for the multi-depth script.
trait CloneShares<F: ProtocolField> {
    fn clone_shares(&self) -> Vec<E<F>>;
}
impl<F: ProtocolField> CloneShares<F> for OpResult<F> {
    fn clone_shares(&self) -> Vec<E<F>> {
        match self {
            OpResult::Shares(s) | OpResult::Public(s) => s.clone(),
            OpResult::Masked { public, .. } => public.clone(),
        }
    }
}
