//! The operations, one file each, all on one template.
//!
//! An op-depth runs one vector operation, and an operation is a fixed list
//! of *steps*, each one batch the engine runs: a reveal, a multiply, or a
//! masked multiply. The op says what goes
//! into each step ([`operands`](Operation::operands)), takes what comes
//! back ([`on_step_complete`](Operation::on_step_complete)), and when its
//! last step has run turns its state into the application's result
//! ([`into_result`](Operation::into_result)). Steps are numbered from 0 in
//! the op's own list; an op never sees a step that is not its own.
//!
//! # The template
//!
//! Each file `ops/<name>.rs` reads top to bottom:
//!
//! 1. a header giving the semantics, the rounds, the steps, and the
//!    preprocessing per element;
//! 2. `steps(ell)`: the op's steps — the one thing the plan (which sizes
//!    the engine's preprocessing from them) and the op share;
//! 3. the state struct — inputs, preprocessing drawn, intermediates, output,
//!    in step order;
//! 4. `impl Operation`: `operands` and `on_step_complete` as a `match` on the
//!    step index, and `into_result`.
//!
//! Two pipelines are shared and live in their own files, not as operations:
//! [`drelu`] (reveal → carry tree → xor, the front of every comparison) and
//! [`fixed_point`] (the ΠTrunc local steps, the back of both truncating
//! ops).
//!
//! Public operands are `FieldElement`s like sharings; under Shamir the
//! arithmetic on them is identical, which is why every `Pub` variant is the
//! same file as its shared sibling with `other` holding public elements.

use anyhow::{bail, Result};
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

use crate::{
    api::application::{Op, OpType, OpResult},
    primitives::edabit::EdaBit,
};

pub mod add;
pub mod compare;
pub mod drelu;
pub mod fixed_mul;
pub mod fixed_point;
pub mod mask_reveal;
pub mod max;
pub mod min;
pub mod mul;
pub mod reveal;
pub mod truncate;

/// Shorthand every op file uses.
pub(crate) type E<F> = FieldElement<F>;

/// The three things the engine can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineOperationType {
    Reveal,
    Multiply,
    MaskedMultiply,
}

/// One step of an op, engine-agnostic: which batch the engine runs, and
/// how many gates (or values opened) per element of the op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpStep {
    pub engine_op: EngineOperationType,
    /// Gates (or values opened) per element of the op.
    pub num_engine_operands: usize,
}

impl OpStep {
    pub const fn new(engine_op: EngineOperationType, num_engine_operands: usize) -> Self {
        Self { engine_op, num_engine_operands }
    }
}

/// The steps of an op type. `ell` is `Some(ℓ)` over a Mersenne prime
/// field; the comparison family's steps depend on it, and `Plan::compile`
/// has already refused those types when it is `None`.
pub fn steps_of(op_type: OpType, ell: Option<usize>) -> Vec<OpStep> {
    let ell = || ell.expect("a Mersenne-only op type planned over a non-Mersenne field");
    match op_type {
        OpType::Mul => mul::steps(),
        OpType::Add => add::steps(),
        OpType::Reveal => reveal::steps(),
        OpType::MaskReveal => mask_reveal::steps(),
        OpType::Truncate => truncate::steps(),
        OpType::FixedMul => fixed_mul::steps(),
        OpType::Compare | OpType::ComparePub => compare::steps(ell()),
        OpType::Max | OpType::MaxPub => max::steps(ell()),
        OpType::Min | OpType::MinPub => min::steps(ell()),
    }
}

/// What an op puts into one step.
pub enum EngineOperands<F: ProtocolField> {
    Reveal(Vec<E<F>>),
    Multiply { x: Vec<E<F>>, y: Vec<E<F>> },
    MaskedMultiply { x: Vec<E<F>>, y: Vec<E<F>>, mask: Vec<E<F>> },
}

impl<F: ProtocolField> EngineOperands<F> {
    pub fn batch_type(&self) -> EngineOperationType {
        match self {
            EngineOperands::Reveal(_) => EngineOperationType::Reveal,
            EngineOperands::Multiply { .. } => EngineOperationType::Multiply,
            EngineOperands::MaskedMultiply { .. } => EngineOperationType::MaskedMultiply,
        }
    }

    /// Results the step will return.
    pub fn len(&self) -> usize {
        match self {
            EngineOperands::Reveal(v) => v.len(),
            EngineOperands::Multiply { x, .. } | EngineOperands::MaskedMultiply { x, .. } => x.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(x, y)` pairs as a multiply batch.
    pub fn multiply(pairs: impl IntoIterator<Item = (E<F>, E<F>)>) -> Self {
        let (x, y) = pairs.into_iter().unzip();
        EngineOperands::Multiply { x, y }
    }
}

/// The template.
pub trait Operation<F: ProtocolField>: Send {
    /// The operands this op puts into its step `step`.
    fn operands(&self, step: usize) -> EngineOperands<F>;

    /// Step `step` has run; `results` are its products or opened values, as
    /// many as `operands(step)` had.
    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()>;

    /// The application's result, once every step has run.
    fn into_result(self: Box<Self>) -> OpResult<F>;
}

/// Build the operation for `op`, drawing its edaBits from `edabits` in
/// element order.
pub fn build<F: ProtocolField>(
    op: Op<F>,
    edabits: &mut impl Iterator<Item = EdaBit<F>>,
) -> Result<Box<dyn Operation<F>>> {
    let n = op.len();
    let mut draw = |count: usize| -> Result<Vec<EdaBit<F>>> {
        let taken: Vec<EdaBit<F>> = edabits.take(count).collect();
        if taken.len() != count {
            bail!("ran out of edaBits: needed {} more", count - taken.len());
        }
        Ok(taken)
    };
    Ok(match op {
        Op::Mul { x, y } => Box::new(mul::Mul::new(x, y)),
        Op::Add { x, y } => Box::new(add::Add::new(x, y)),
        Op::Reveal { x } => Box::new(reveal::Reveal::new(x)),
        Op::MaskReveal { x } => Box::new(mask_reveal::MaskReveal::new(x, draw(n)?)),
        Op::Truncate { x, d } => Box::new(truncate::Truncate::new(x, d, draw(n)?)),
        Op::FixedMul { x, y, d } => Box::new(fixed_mul::FixedMul::new(x, y, d, draw(n)?)),
        Op::Compare { a, b } | Op::ComparePub { a, c: b } => Box::new(compare::Compare::new(a, b, draw(n)?)),
        Op::Max { a, b } | Op::MaxPub { a, c: b } => Box::new(max::Max::new(a, b, draw(n)?)),
        Op::Min { a, b } | Op::MinPub { a, c: b } => Box::new(min::Min::new(a, b, draw(n)?)),
    })
}

/// A step outside the op's list.
pub(crate) fn no_such_step(op_type: OpType, step: usize) -> anyhow::Error {
    anyhow::anyhow!("{:?} has no step {}", op_type, step)
}
