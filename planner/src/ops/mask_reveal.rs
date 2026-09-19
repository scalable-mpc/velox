//! `MaskReveal { x }` — open `c = x + r` for a Planner-managed random `r`,
//! and hand back `r`'s edaBit alongside.
//!
//! Rounds: 1. Steps: one reveal. Preprocessing: one edaBit per element.
//!
//! This is the primitive under truncation and comparison, exposed so an
//! application can build its own bit tricks on a value it knows is
//! uniformly blinded.

use anyhow::Result;
use fields::{MersennePrimeField, ProtocolField};

use crate::{
    api::application::{OpType, OpResult},
    primitives::edabit::EdaBit,
};

use super::{no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    vec![OpStep::new(EngineOperationType::Reveal, 1)]
}

pub struct MaskReveal<F: ProtocolField + MersennePrimeField> {
    x: Vec<E<F>>,
    eda: Vec<EdaBit<F>>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField + MersennePrimeField> MaskReveal<F> {
    pub fn new(x: Vec<E<F>>, eda: Vec<EdaBit<F>>) -> Self {
        Self { x, eda, out: Vec::new() }
    }
}

impl<F: ProtocolField + MersennePrimeField> Operation<F> for MaskReveal<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        EngineOperands::Reveal(self.x.iter().zip(self.eda.iter()).map(|(x, r)| x + &r.value).collect())
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        match step {
            0 => Ok(self.out = results),
            _ => Err(no_such_step(OpType::MaskReveal, step)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Masked { public: self.out, mask: self.eda }
    }
}
