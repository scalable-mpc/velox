//! The Planner driven by a plaintext engine.
//!
//! Sharings are linear, so a plaintext value is a valid sharing (a degree-0
//! polynomial): an engine where `Multiply` is the product, `Reveal` the
//! identity and `MaskedMultiply` is `x·y + mask` exercises every line of the
//! Planner except the network. Each op is checked against its integer
//! reference, at p_bits = 61, 31 and 127, and over the ML-DSA field
//! (`q = 2^23 − 2^13 + 1`) for the ops that need only uniform masks.

use std::collections::VecDeque;

use anyhow::Result;
use planner::api::engine::{Application, DepthInput, RandomWireShares};
use async_trait::async_trait;
use fields::{mersenne_31::Mersenne31Field, Mersenne127Field, MersennePrimeField, Mersenne61Field, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use planner::{primitives::{carry_tree::CarryTree, mersenne::from_u128}, OpParams, Op, OpDepthInput, OpType, OpResult, Planner, PlannerApplication, PlannerCounts};

type E<F> = FieldElement<F>;

// ---------------------------------------------------------------------------
// Integer view of the field
// ---------------------------------------------------------------------------

fn from_signed<F: ProtocolField + MersennePrimeField>(v: i128) -> E<F> {
    let magnitude = from_u128::<F>(v.unsigned_abs());
    if v >= 0 { magnitude } else { -magnitude }
}

/// `p < 2^127`, so the canonical value and `p` both fit an `i128`.
fn to_signed<F: ProtocolField + MersennePrimeField>(e: &E<F>) -> i128 {
    let c = F::to_canonical_u128(e) as i128;
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
    /// Uniform in `[-bound, bound)`, from two draws so that the 127-bit
    /// field's domain (`bound = 2^125`) is covered.
    fn signed(&mut self, bound: i128) -> i128 {
        let wide = ((self.next() as u128) << 64) | self.next() as u128;
        (wide % (2 * bound) as u128) as i128 - bound
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
        assert_eq!(signed_all(&results[i]), want[i], "{op_type:?} at p_bits={}", F::BITS);
    }
}

#[test]
fn comparison_family_exhaustive_on_small_values() {
    let pairs: Vec<(i128, i128)> = (-6..=6).flat_map(|a| (-6..=6).map(move |b| (a, b))).collect();
    comparison_family::<Mersenne61Field>(pairs.clone(), 1);
    comparison_family::<Mersenne31Field>(pairs.clone(), 1);
    comparison_family::<Mersenne127Field>(pairs, 1);
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
    go::<Mersenne127Field>(0x9abc);
}

fn truncation<F: ProtocolField + MersennePrimeField>(seed: u64) {
    let p_bits = F::BITS;
    let bound = 1i128 << (p_bits - 2);
    let mut rng = Rng(seed);
    for d in [1usize, 8, 16, p_bits - 3] {
        let mut xs: Vec<i128> = (0..64).map(|_| rng.signed(bound)).collect();
        xs.extend([0, 1, -1, bound - 1, -bound, (1 << d) - 1, -((1 << d) - 1), 1 << d, -(1 << d)]);
        // Factors whose product stays in the domain.
        let half = 1i128 << ((p_bits - 2) / 2);
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
            assert!((got - want).abs() <= 2, "Trunc_{d}({x}) = {got}, want {want} ± 2 at p_bits={p_bits}");
        }
        for ((x, y), got) in pairs.iter().zip(signed_all(&results[1])) {
            let want = trunc_ref(x * y, d);
            assert!((got - want).abs() <= 2, "FixedMul_{d}({x}, {y}) = {got}, want {want} ± 2 at p_bits={p_bits}");
        }
    }
}

#[test]
fn truncation_and_fixed_point_multiplication_within_two() {
    truncation::<Mersenne61Field>(0xabcd);
    truncation::<Mersenne31Field>(0xef01);
    truncation::<Mersenne127Field>(0x2345);
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

/// Over the ML-DSA field, `q = 2^23 − 2^13 + 1`, `MaskReveal` draws masks
/// from the uniform-mask prefix: six engine depths before op-depth 1, over
/// the candidates. Every mask is below `q`, its bits are 0/1 and compose it,
/// and `c − r` recovers `x`. 3,000 masks make about 3 rejections likely.
#[test]
fn ml_dsa_field_masks_come_from_the_uniform_mask_prefix() {
    type F = fields::MlDsaField;
    const Q: u128 = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
    const N: usize = 3000;
    let xs: Vec<E<F>> = (0..N as u64).map(|i| E::<F>::from(i * 2797 % Q as u64)).collect();
    let x_op = xs.clone();
    let script = Script::<F>::new(PlannerCounts::new(vec![OpParams::new(OpType::MaskReveal, N)], 0))
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::MaskReveal { x: x_op.clone() } });
    let mut planner = Planner::new(script).unwrap();
    let candidates = planner.plan().mask_test().unwrap().1;
    assert_eq!(candidates, N + 40 + 24);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, 23));
    let c = candidates;
    assert_eq!(
        batches,
        vec![
            Batch::Multiply(1, c),
            Batch::Multiply(2, 2 * c),
            Batch::Multiply(3, c),
            Batch::Multiply(4, c),
            Batch::Multiply(5, c),
            Batch::Reveal(6, c),
            Batch::Reveal(7, N),
        ]
    );
    let OpResult::Masked { public, mask } = &planner.app().results[0] else { panic!("MaskReveal returns Masked") };
    // The candidates, rebuilt from the plaintext engine's bits (the
    // Planner's come first): the kept masks are the first N below q.
    let mut rng = Rng(23 | 1);
    let candidate_values: Vec<u128> =
        (0..c).map(|_| (0..23).map(|i| ((rng.next() & 1) as u128) << i).sum()).collect();
    let good: Vec<u128> = candidate_values.iter().copied().filter(|r| *r < Q).take(N).collect();
    assert!(candidate_values.iter().any(|r| *r >= Q), "this seed exercises a rejection");
    assert_eq!(mask.iter().map(|r| F::canonical_u128(&r.value).unwrap()).collect::<Vec<_>>(), good);
    for ((opened, r), x) in public.iter().zip(mask).zip(&xs) {
        assert_eq!(&(opened - &r.value), x, "c − r recovers x");
        assert_eq!(r.bits.len(), 23);
        let bits: Vec<u128> = r.bits.iter().map(|b| F::canonical_u128(b).unwrap()).collect();
        assert!(bits.iter().all(|b| *b <= 1), "bits are 0/1");
        let composed: u128 = bits.iter().enumerate().map(|(i, b)| b << i).sum();
        assert!(composed < Q, "the mask is below q");
        assert_eq!(F::canonical_u128(&r.value), Some(composed));
    }
}

/// `LessThanPub` over the ML-DSA field, through the full Planner: the
/// uniform-mask prefix, one reveal, then the two 23-bit carry trees side by
/// side at five engine depths. Each result is exactly `[x < c]`, also at the
/// edges of `[0, q)` and around an ML-DSA rejection bound.
#[test]
fn ml_dsa_field_less_than_pub_is_exact() {
    type F = fields::MlDsaField;
    const Q: u128 = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
    let edges = [0, 1, Q / 2, Q - 2, Q - 1];
    let mut pairs: Vec<(u128, u128)> = edges.iter().flat_map(|&x| edges.map(|c| (x, c))).collect();
    // ML-DSA-65: 2(γ1 − β) − 1 with γ1 = 2^19, β = 196.
    let bound = 2 * ((1 << 19) - 196) - 1;
    pairs.extend((0..20).map(|i| (bound - 10 + i, bound)));
    let mut rng = Rng(0x51ed);
    pairs.extend((0..500).map(|_| ((rng.next() as u128) % Q, (rng.next() as u128) % Q)));
    let n = pairs.len();
    let x: Vec<E<F>> = pairs.iter().map(|&(x, _)| from_u128::<F>(x)).collect();
    let c: Vec<E<F>> = pairs.iter().map(|&(_, c)| from_u128::<F>(c)).collect();
    let script = Script::<F>::new(PlannerCounts::new(vec![OpParams::new(OpType::LessThanPub, n)], 0))
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::LessThanPub { x: x.clone(), c: c.clone() } });
    let mut planner = Planner::new(script).unwrap();
    let candidates = planner.plan().mask_test().unwrap().1;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, 41));

    let mut expected = vec![
        Batch::Multiply(1, candidates),
        Batch::Multiply(2, 2 * candidates),
        Batch::Multiply(3, candidates),
        Batch::Multiply(4, candidates),
        Batch::Multiply(5, candidates),
        Batch::Reveal(6, candidates),
        Batch::Reveal(7, n),
    ];
    let widths = CarryTree::new(23).level_widths();
    assert_eq!(widths.len(), 5);
    expected.extend(widths.iter().enumerate().map(|(level, w)| Batch::Multiply(8 + level, 2 * w * n)));
    assert_eq!(batches, expected);

    let OpResult::Shares(out) = &planner.app().results[0] else { panic!("LessThanPub returns shares") };
    for ((x, c), got) in pairs.iter().zip(out) {
        assert_eq!(F::canonical_u128(got), Some(u128::from(x < c)), "x = {x}, c = {c}");
    }
}

/// `CanonicalMod2m` with the ML-DSA `m = 4` over the ML-DSA field, through
/// the full Planner: the uniform-mask prefix (this seed rejects 2 of the
/// 645 candidates), one reveal, five levels of the 23-bit and 4-bit trees side
/// by side, and the selection. Each result is exactly `x mod 16`, also for
/// `x = −16·w + (q − 1)/2` as ML-DSA computes it.
#[test]
fn ml_dsa_field_canonical_mod2m_is_exact() {
    type F = fields::MlDsaField;
    const Q: u128 = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
    const N: usize = 600;
    const SEED: u64 = 4;
    let mut values: Vec<u128> = vec![0, 1, 15, 16, 17, Q / 2, Q - 17, Q - 16, Q - 2, Q - 1];
    let mut rng = Rng(0xc0de);
    values.extend((0..N - values.len()).map(|i| {
        let w = (rng.next() as u128) % Q;
        if i % 2 == 0 { w } else { ((Q - 16) * w + (Q - 1) / 2) % Q }
    }));
    let x: Vec<E<F>> = values.iter().map(|&v| from_u128::<F>(v)).collect();
    let script = Script::<F>::new(PlannerCounts::new(vec![OpParams::new(OpType::CanonicalMod2m { m: 4 }, N)], 0))
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::CanonicalMod2m { x: x.clone(), m: 4 } });
    let mut planner = Planner::new(script).unwrap();
    let candidates = planner.plan().mask_test().unwrap().1;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, SEED));

    // The run's candidates, rebuilt from the plaintext engine's bits: 2 of
    // them are at or above q.
    let mut bits = Rng(SEED | 1);
    let has_rejected_candidate = (0..candidates).any(|_| (0..23).map(|i| ((bits.next() & 1) as u128) << i).sum::<u128>() >= Q);
    assert!(has_rejected_candidate, "this seed exercises a rejection");

    let mut expected = vec![
        Batch::Multiply(1, candidates),
        Batch::Multiply(2, 2 * candidates),
        Batch::Multiply(3, candidates),
        Batch::Multiply(4, candidates),
        Batch::Multiply(5, candidates),
        Batch::Reveal(6, candidates),
        Batch::Reveal(7, N),
    ];
    let (wrap, low) = (CarryTree::new(23).level_widths(), CarryTree::new(4).level_widths());
    expected.extend((0..wrap.len()).map(|level| Batch::Multiply(8 + level, (wrap[level] + 2 * low.get(level).copied().unwrap_or(0)) * N)));
    expected.push(Batch::Multiply(8 + wrap.len(), N));
    assert_eq!(batches, expected);

    let OpResult::Shares(out) = &planner.app().results[0] else { panic!("CanonicalMod2m returns shares") };
    for (v, got) in values.iter().zip(out) {
        assert_eq!(F::canonical_u128(got), Some(v % 16), "x = {v}");
    }
}

/// `CanonicalMod` with the ML-DSA-44 `d = 44` over the ML-DSA field, through
/// the full Planner: the uniform-mask prefix (this seed rejects 2 of the
/// 645 candidates), one reveal, five levels of the wrap tree and the
/// `y = g^r` product tree side by side, six power steps, and the selection.
/// Each result is exactly `x mod 44`, also for `x = −44·w + (q − 1)/2` as
/// ML-DSA-44 computes it.
#[test]
fn ml_dsa_field_canonical_mod_is_exact() {
    type F = fields::MlDsaField;
    const Q: u128 = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
    const N: usize = 600;
    const SEED: u64 = 4;
    let mut values: Vec<u128> = vec![0, 1, 43, 44, 45, Q / 2, Q - 45, Q - 44, Q - 2, Q - 1];
    let mut rng = Rng(0xbead);
    values.extend((0..N - values.len()).map(|i| {
        let w = (rng.next() as u128) % Q;
        if i % 2 == 0 { w } else { ((Q - 44) * w + (Q - 1) / 2) % Q }
    }));
    let x: Vec<E<F>> = values.iter().map(|&v| from_u128::<F>(v)).collect();
    let script = Script::<F>::new(PlannerCounts::new(vec![OpParams::new(OpType::CanonicalMod { d: 44 }, N)], 0))
        .then(move |_| OpDepthInput::Op { depth: 1, op: Op::CanonicalMod { x: x.clone(), d: 44 } });
    let mut planner = Planner::new(script).unwrap();
    let candidates = planner.plan().mask_test().unwrap().1;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, SEED));

    let mut expected = vec![
        Batch::Multiply(1, candidates),
        Batch::Multiply(2, 2 * candidates),
        Batch::Multiply(3, candidates),
        Batch::Multiply(4, candidates),
        Batch::Multiply(5, candidates),
        Batch::Reveal(6, candidates),
        Batch::Reveal(7, N),
    ];
    // The wrap tree and the product tree of 23 factors (11, 6, 3, 1, 1).
    let wrap = CarryTree::new(23).level_widths();
    let product = [11, 6, 3, 1, 1];
    assert_eq!(wrap.len(), 5);
    expected.extend((0..5).map(|level| Batch::Multiply(8 + level, (wrap[level] + product[level]) * N)));
    // The powers y^2..y^43, then the selection.
    expected.extend([1, 2, 4, 8, 16, 11].iter().enumerate().map(|(j, w)| Batch::Multiply(13 + j, w * N)));
    expected.push(Batch::Multiply(19, N));
    assert_eq!(batches, expected);

    let OpResult::Shares(out) = &planner.app().results[0] else { panic!("CanonicalMod returns shares") };
    for (v, got) in values.iter().zip(out) {
        assert_eq!(F::canonical_u128(got), Some(v % 44), "x = {v}");
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

/// Runs one `Mod2m` op-depth per `(m, xs)` case and checks every result
/// against `x mod 2^m`, and every engine batch against the plan: a reveal,
/// then the carry tree's levels.
fn mod2m<F: ProtocolField + MersennePrimeField>(cases: Vec<(usize, Vec<u128>)>, seed: u64) {
    let ops = cases.iter().map(|(m, xs)| OpParams::new(OpType::Mod2m { m: *m }, xs.len())).collect();
    let mut script = Script::<F>::new(PlannerCounts::new(ops, 0));
    for (i, (m, xs)) in cases.iter().cloned().enumerate() {
        script = script.then(move |_| OpDepthInput::Op {
            depth: i + 1,
            op: Op::Mod2m { x: xs.iter().map(|x| from_u128::<F>(*x)).collect(), m },
        });
    }
    let mut planner = Planner::new(script).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt.block_on(run(&mut planner, seed));

    let mut want_batches = Vec::new();
    for (m, xs) in &cases {
        want_batches.push(Batch::Reveal(want_batches.len() + 1, xs.len()));
        for width in CarryTree::new(*m).level_widths() {
            want_batches.push(Batch::Multiply(want_batches.len() + 1, width * xs.len()));
        }
    }
    assert_eq!(batches, want_batches);

    for ((m, xs), result) in cases.iter().zip(&planner.app().results) {
        let OpResult::Shares(got) = result else { panic!("Mod2m returns shares") };
        for (x, e) in xs.iter().zip(got) {
            assert_eq!(F::to_canonical_u128(e), x % (1u128 << m), "{x} mod 2^{m} at p_bits={}", F::BITS);
        }
    }
}

#[test]
fn mod2m_exhaustive_on_small_values() {
    let xs: Vec<u128> = (0..1 << 10).collect();
    mod2m::<Mersenne61Field>((1..=10).map(|m| (m, xs.clone())).collect(), 0x3141);
}

/// The decryption's shape: `m = 63` on inputs up to the 86-bit limit, plus
/// both ends of the range of `m`.
#[test]
fn mod2m_randomised_at_127_bits() {
    let limit = 1u128 << 86;
    let mut rng = Rng(0x2718);
    let mut xs: Vec<u128> = (0..200).map(|_| (((rng.next() as u128) << 64) | rng.next() as u128) % limit).collect();
    xs.extend([0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1, limit - 1]);
    mod2m::<Mersenne127Field>(vec![(63, xs.clone()), (1, xs.clone()), (86, xs)], 0x1618);
}
