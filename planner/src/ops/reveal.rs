//! `Reveal { x }` — open `x` to everyone.
//!
//! Rounds: 1. Steps: one reveal.
//!
//! Contract: `x` must be blinded by a fresh random sharing. What a party
//! learns in the reconstruction is the sharing polynomial of a public
//! combination of the revealed values, which is harmless only when those
//! polynomials are uniformly random. `MaskReveal` is the safe form.

use anyhow::Result;
use fields::ProtocolField;

use crate::api::application::{OpType, OpResult};

use super::{no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    vec![OpStep::new(EngineOperationType::Reveal, 1)]
}

pub struct Reveal<F: ProtocolField> {
    x: Vec<E<F>>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField> Reveal<F> {
    pub fn new(x: Vec<E<F>>) -> Self {
        Self { x, out: Vec::new() }
    }
}

impl<F: ProtocolField> Operation<F> for Reveal<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        EngineOperands::Reveal(self.x.clone())
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        match step {
            0 => Ok(self.out = results),
            _ => Err(no_such_step(OpType::Reveal, step)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Public(self.out)
    }
}
