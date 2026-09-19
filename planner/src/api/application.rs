//! The application-facing API: what an application asks the Planner for,
//! and what it gets back.
//!
//! An application implements [`PlannerApplication`] — the hooks of the
//! engine's `Application`, one level up — and returns an [`OpDepthInput`]
//! from each: one vector [`Op`] to run as one op-depth. When the op-depth
//! completes it receives one [`OpResult`]. Before preprocessing it declares
//! every op-depth's type and width in [`PlannerCounts`]; an op-depth may
//! then run fewer elements than declared, never more, and never a different
//! type.

use anyhow::{bail, Result};
use crate::api::engine::RandomWireShares;
use async_trait::async_trait;
use fields::{MersennePrimeField, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

use crate::primitives::edabit::EdaBit;

/// The application's side of the Planner: the hooks of `Application`, with
/// `OpDepthInput` in place of `DepthInput` and op results in place of
/// products.
#[async_trait]
pub trait PlannerApplication<F: ProtocolField + MersennePrimeField>: Send + 'static {
    /// Pure, read once before preprocessing, the same at every party.
    fn preprocessing_count(&self) -> PlannerCounts;

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        Vec::new()
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<OpDepthInput<F>>;

    /// `wires` holds the random wires the application asked for in
    /// `PlannerCounts`; the Planner has already taken its own.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>>;

    /// Op-depth `depth` has completed.
    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>>;

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let _ = outputs;
        Ok(())
    }
}


/// The Application tells the Planner to execute these set of operations next. 
pub enum OpDepthInput<F: ProtocolField> {
    Waiting,
    /// Run `op` as op-depth `depth`, counting from 1.
    Op { depth: usize, op: Op<F> },
    Done(Vec<FieldElement<F>>),
}

impl<F: ProtocolField> OpDepthInput<F> {
    pub fn op(depth: usize, op: Op<F>) -> Result<Self> {
        if depth == 0 {
            bail!("op-depth 0; op-depths count from 1");
        }
        Ok(Self::Op { depth, op })
    }

    pub fn is_waiting(&self) -> bool {
        matches!(self, Self::Waiting)
    }
}

impl<F: ProtocolField> std::fmt::Debug for OpDepthInput<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Waiting => write!(f, "Waiting"),
            Self::Op { depth, op } => write!(f, "Op(depth {}, {:?})", depth, op),
            Self::Done(outputs) => write!(f, "Done({} output wires)", outputs.len()),
        }
    }
}

/// What an op-depth produced.
pub enum OpResult<F: ProtocolField + MersennePrimeField> {
    Shares(Vec<FieldElement<F>>),
    Public(Vec<FieldElement<F>>),
    /// `MaskReveal`: the public `x + r` and `r`'s edaBit, elementwise.
    Masked { public: Vec<FieldElement<F>>, mask: Vec<EdaBit<F>> },
}

impl<F: ProtocolField + MersennePrimeField> OpResult<F> {
    pub fn shares(self) -> Result<Vec<FieldElement<F>>> {
        match self {
            Self::Shares(s) => Ok(s),
            other => bail!("expected shared results, got {:?}", other),
        }
    }

    pub fn public(self) -> Result<Vec<FieldElement<F>>> {
        match self {
            Self::Public(p) => Ok(p),
            other => bail!("expected public results, got {:?}", other),
        }
    }
}

impl<F: ProtocolField + MersennePrimeField> std::fmt::Debug for OpResult<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shares(s) => write!(f, "Shares({})", s.len()),
            Self::Public(p) => write!(f, "Public({})", p.len()),
            Self::Masked { public, .. } => write!(f, "Masked({})", public.len()),
        }
    }
}

/// One operation over vectors of sharings, elementwise. Public operands
/// (`c`) are field elements every party holds in the clear.
///
/// Domains: `Compare`, `Max`, `Min` and their `Pub` variants need
/// `|a|, |b| < 2^{ℓ−2}` read as signed integers, so that `a − b` keeps its
/// sign in the top bit; `Truncate` and `FixedMul` need `|x|, |x·y| <
/// 2^{ℓ−2}` and produce `Trunc_d(·)` up to `±2` (Liu et al. §3).
pub enum Op<F: ProtocolField> {
    /// `x · y`, one round.
    Mul { x: Vec<FieldElement<F>>, y: Vec<FieldElement<F>> },
    /// `x + y`, local.
    Add { x: Vec<FieldElement<F>>, y: Vec<FieldElement<F>> },
    /// `[a < b]`, as a sharing of 0 or 1.
    Compare { a: Vec<FieldElement<F>>, b: Vec<FieldElement<F>> },
    /// `[a < c]` for public `c`.
    ComparePub { a: Vec<FieldElement<F>>, c: Vec<FieldElement<F>> },
    Max { a: Vec<FieldElement<F>>, b: Vec<FieldElement<F>> },
    Min { a: Vec<FieldElement<F>>, b: Vec<FieldElement<F>> },
    /// `max(a, c)` for public `c`; `Relu(x)` is `MaxPub(x, 0)`.
    MaxPub { a: Vec<FieldElement<F>>, c: Vec<FieldElement<F>> },
    MinPub { a: Vec<FieldElement<F>>, c: Vec<FieldElement<F>> },
    /// Open `x` to everyone. `x` must be blinded by a fresh random sharing:
    /// what a party learns in the reconstruction is the sharing polynomial
    /// of a public combination of the revealed values, which is harmless
    /// only when those polynomials are uniformly random.
    Reveal { x: Vec<FieldElement<F>> },
    /// Open `x + r` for a Planner-managed random `r`, and return `r`'s
    /// edaBit alongside — the primitive under `Truncate` and `Compare`.
    MaskReveal { x: Vec<FieldElement<F>> },
    /// `Trunc_d(x)`: drop the low `d` bits, keeping the sign.
    Truncate { x: Vec<FieldElement<F>>, d: usize },
    /// `Trunc_d(x · y)` in one round: a fixed-point multiplication with `d`
    /// fractional bits.
    FixedMul { x: Vec<FieldElement<F>>, y: Vec<FieldElement<F>>, d: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpType {
    Mul,
    Add,
    Compare,
    ComparePub,
    Max,
    Min,
    MaxPub,
    MinPub,
    Reveal,
    MaskReveal,
    Truncate,
    FixedMul,
}

impl OpType {
    /// edaBits an element of this type consumes: one for anything that
    /// blinds a value with a random `r` it also needs the bits of.
    pub fn edabits_per_element(self) -> usize {
        match self {
            OpType::Mul | OpType::Add | OpType::Reveal => 0,
            _ => 1,
        }
    }
}

impl<F: ProtocolField> Op<F> {
    pub fn op_type(&self) -> OpType {
        match self {
            Op::Mul { .. } => OpType::Mul,
            Op::Add { .. } => OpType::Add,
            Op::Compare { .. } => OpType::Compare,
            Op::ComparePub { .. } => OpType::ComparePub,
            Op::Max { .. } => OpType::Max,
            Op::Min { .. } => OpType::Min,
            Op::MaxPub { .. } => OpType::MaxPub,
            Op::MinPub { .. } => OpType::MinPub,
            Op::Reveal { .. } => OpType::Reveal,
            Op::MaskReveal { .. } => OpType::MaskReveal,
            Op::Truncate { .. } => OpType::Truncate,
            Op::FixedMul { .. } => OpType::FixedMul,
        }
    }

    /// Elements the op works on.
    pub fn len(&self) -> usize {
        match self {
            Op::Mul { x, .. } | Op::Add { x, .. } | Op::Reveal { x } | Op::MaskReveal { x }
            | Op::Truncate { x, .. } | Op::FixedMul { x, .. } => x.len(),
            Op::Compare { a, .. } | Op::ComparePub { a, .. } | Op::Max { a, .. } | Op::Min { a, .. }
            | Op::MaxPub { a, .. } | Op::MinPub { a, .. } => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The parameters this op runs with — what a declaration is compared to.
    pub fn params(&self) -> OpParams {
        OpParams::new(self.op_type(), self.len())
    }

    /// Operand vectors agree in length, and `d` is in range for `ℓ` bits.
    pub fn validate(&self, ell: usize) -> Result<()> {
        let pair = |l: usize, r: usize, what: &str| -> Result<()> {
            if l != r {
                bail!("{:?}: {} operands of length {} and {}", self.op_type(), what, l, r);
            }
            Ok(())
        };
        match self {
            Op::Mul { x, y } | Op::Add { x, y } => pair(x.len(), y.len(), "x, y"),
            Op::Compare { a, b } | Op::Max { a, b } | Op::Min { a, b } => pair(a.len(), b.len(), "a, b"),
            Op::ComparePub { a, c } | Op::MaxPub { a, c } | Op::MinPub { a, c } => pair(a.len(), c.len(), "a, c"),
            Op::Reveal { .. } | Op::MaskReveal { .. } => Ok(()),
            Op::Truncate { d, .. } => Self::check_d(*d, ell),
            Op::FixedMul { x, y, d } => {
                pair(x.len(), y.len(), "x, y")?;
                Self::check_d(*d, ell)
            }
        }
    }

    fn check_d(d: usize, ell: usize) -> Result<()> {
        // ΠTrunc's correction term is 2^{ℓ−d−2}, so d ≤ ℓ − 3 keeps it ≥ 2.
        if d == 0 || d + 2 >= ell {
            bail!("truncation by {} bits is outside 1..={} for an {}-bit field", d, ell - 3, ell);
        }
        Ok(())
    }
}

/// Shapes, not contents.
impl<F: ProtocolField> std::fmt::Debug for Op<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Truncate { d, .. } | Op::FixedMul { d, .. } => write!(f, "{:?}({} elements, d={})", self.op_type(), self.len(), d),
            _ => write!(f, "{:?}({} elements)", self.op_type(), self.len()),
        }
    }
}

/// An op's parameters without its operands: its type and its width. What
/// an application declares for each op-depth before preprocessing, when
/// the operands — sharings the circuit has not produced yet — do not exist.
/// The op eventually run at that depth must have the same type and at most
/// this width; see [`Op::params`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpParams {
    pub op_type: OpType,
    /// Elements — the width of the vector op.
    pub elements: usize,
}

impl OpParams {
    pub fn new(op_type: OpType, elements: usize) -> Self {
        Self { op_type, elements }
    }

    pub fn edabits(&self) -> usize {
        self.elements * self.op_type.edabits_per_element()
    }

    /// True if an op with parameters `run` may run under this declaration.
    pub fn covers(&self, run: &OpParams) -> bool {
        self.op_type == run.op_type && run.elements <= self.elements
    }
}

/// What the Planner needs to know before preprocessing: the parameters of
/// every op-depth's op, plus the random wires the application reads itself.
#[derive(Clone, Debug, Default)]
pub struct PlannerCounts {
    /// Op-depth 1 first: the op each op-depth will run, minus its operands.
    pub ops: Vec<OpParams>,
    /// Random bits the application consumes directly, beyond the Planner's.
    pub rand_bits: usize,
    /// Random sharings the application consumes directly.
    pub sharings: usize,
    /// Output wires to reconstruct.
    pub output: usize,
}

impl PlannerCounts {
    pub fn new(ops: Vec<OpParams>, output: usize) -> Self {
        Self { ops, rand_bits: 0, sharings: 0, output }
    }

    pub fn with_random_wires(mut self, rand_bits: usize, sharings: usize) -> Self {
        self.rand_bits = rand_bits;
        self.sharings = sharings;
        self
    }

    pub fn depth(&self) -> usize {
        self.ops.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::Mersenne61Field;

    type F = Mersenne61Field;

    fn v(n: usize) -> Vec<FieldElement<F>> {
        (0..n as u64).map(FieldElement::<F>::from).collect()
    }

    #[test]
    fn validation() {
        assert!(Op::<F>::Mul { x: v(2), y: v(3) }.validate(61).is_err());
        assert!(Op::<F>::Compare { a: v(2), b: v(2) }.validate(61).is_ok());
        assert!(Op::<F>::Truncate { x: v(1), d: 0 }.validate(61).is_err());
        assert!(Op::<F>::Truncate { x: v(1), d: 58 }.validate(61).is_ok());
        assert!(Op::<F>::Truncate { x: v(1), d: 59 }.validate(61).is_err());
        assert!(Op::<F>::FixedMul { x: v(1), y: v(1), d: 16 }.validate(61).is_ok());
        assert!(Op::<F>::FixedMul { x: v(1), y: v(2), d: 16 }.validate(61).is_err());
        assert!(OpDepthInput::<F>::op(0, Op::Add { x: v(1), y: v(1) }).is_err());
    }

    #[test]
    fn params_count_edabits_and_cover_runs() {
        assert_eq!(OpParams::new(OpType::Compare, 5).edabits(), 5);
        assert_eq!(OpParams::new(OpType::Mul, 5).edabits(), 0);
        assert_eq!(OpParams::new(OpType::Reveal, 5).edabits(), 0);
        assert_eq!(OpParams::new(OpType::MaskReveal, 5).edabits(), 5);
        assert_eq!(OpParams::new(OpType::FixedMul, 5).edabits(), 5);
        let counts = PlannerCounts::new(vec![OpParams::new(OpType::Add, 1), OpParams::new(OpType::Max, 2)], 3)
            .with_random_wires(4, 5);
        assert_eq!((counts.depth(), counts.rand_bits, counts.sharings, counts.output), (2, 4, 5, 3));
        let declared = OpParams::new(OpType::Compare, 3);
        assert!(declared.covers(&OpParams::new(OpType::Compare, 2)));
        assert!(declared.covers(&declared));
        assert!(!declared.covers(&OpParams::new(OpType::Compare, 4)));
        assert!(!declared.covers(&OpParams::new(OpType::ComparePub, 1)));
        assert_eq!(Op::<F>::Compare { a: v(2), b: v(2) }.params(), OpParams::new(OpType::Compare, 2));
    }
}
