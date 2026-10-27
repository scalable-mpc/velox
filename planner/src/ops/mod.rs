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
//! 2. `steps(..)`: the op's steps — the one thing the plan (which sizes
//!    the engine's preprocessing from them) and the op share. Its argument
//!    is what the step count depends on: the edaBit width for the
//!    comparison family and `LessThanPub`,
//!    `m` for `Mod2m`, the prime's `MaskShape` for the uniform-mask prefix,
//!    and none for the one-step ops;
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
//! One operation is not an application op: [`uniform_mask`], the prefix
//! that the Planner runs itself before op-depth 1 over a prime
//! `2^p_bits_0 − 2^p_bits_1 + 1`. It keeps the template, but no `Op` variant
//! builds it.
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
pub mod canonical_mod2m;
pub mod compare;
pub mod drelu;
pub mod fixed_mul;
pub mod fixed_point;
pub mod less_than_pub;
pub mod mask_reveal;
pub mod max;
pub mod min;
pub mod mod2m;
pub mod mul;
pub mod reveal;
pub mod truncate;
pub mod uniform_mask;

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

/// The steps of an op type. `mask_bits` is `Plan::mask_bits`, the width of
/// one edaBit, and the comparison family's carry tree runs over these bits.
/// `Plan::compile` has already refused the types that the field cannot run.
pub fn steps_of(op_type: OpType, mask_bits: Option<usize>) -> Vec<OpStep> {
    let bits = || mask_bits.expect("an op type with edaBits planned over a field without uniform masks");
    match op_type {
        OpType::Mul => mul::steps(),
        OpType::Add => add::steps(),
        OpType::Reveal => reveal::steps(),
        OpType::MaskReveal => mask_reveal::steps(),
        OpType::Truncate => truncate::steps(),
        OpType::FixedMul => fixed_mul::steps(),
        OpType::Mod2m { m } => mod2m::steps(m),
        OpType::Compare | OpType::ComparePub => compare::steps(bits()),
        OpType::Max | OpType::MaxPub => max::steps(bits()),
        OpType::Min | OpType::MinPub => min::steps(bits()),
        OpType::LessThanPub => less_than_pub::steps(bits()),
        OpType::CanonicalMod2m { m } => canonical_mod2m::steps(bits(), m),
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
/// element order. `mask_bits` is `Plan::mask_bits`, the edaBit width that
/// `LessThanPub` gives to its carry trees.
pub fn build<F: ProtocolField>(
    op: Op<F>,
    edabits: &mut impl Iterator<Item = EdaBit<F>>,
    mask_bits: Option<usize>,
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
        Op::Mod2m { x, m } => Box::new(mod2m::Mod2m::new(x, m, draw(n)?)),
        Op::Compare { a, b } | Op::ComparePub { a, c: b } => Box::new(compare::Compare::new(a, b, draw(n)?)),
        Op::Max { a, b } | Op::MaxPub { a, c: b } => Box::new(max::Max::new(a, b, draw(n)?)),
        Op::Min { a, b } | Op::MinPub { a, c: b } => Box::new(min::Min::new(a, b, draw(n)?)),
        Op::LessThanPub { x, c } => {
            let bits = mask_bits.expect("validated: the field has uniform masks");
            Box::new(less_than_pub::LessThanPub::new(x, c, bits, draw(n)?))
        }
        Op::CanonicalMod2m { x, m } => {
            let bits = mask_bits.expect("validated: the field has uniform masks");
            Box::new(canonical_mod2m::CanonicalMod2m::new(x, m, bits, draw(n)?))
        }
    })
}

/// A step outside the op's list.
pub(crate) fn no_such_step(op_type: OpType, step: usize) -> anyhow::Error {
    anyhow::anyhow!("{:?} has no step {}", op_type, step)
}
