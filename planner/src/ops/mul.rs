//! `Mul { x, y }` — `x · y`, elementwise.
//!
//! Rounds: 1. Steps: one multiply.

use anyhow::Result;
use fields::{MersennePrimeField, ProtocolField};

use crate::api::application::{OpType, OpResult};

use super::{no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    vec![OpStep::new(EngineOperationType::Multiply, 1)]
}

pub struct Mul<F: ProtocolField> {
    x: Vec<E<F>>,
    y: Vec<E<F>>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField> Mul<F> {
    pub fn new(x: Vec<E<F>>, y: Vec<E<F>>) -> Self {
        Self { x, y, out: Vec::new() }
    }
}

impl<F: ProtocolField + MersennePrimeField> Operation<F> for Mul<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        EngineOperands::Multiply { x: self.x.clone(), y: self.y.clone() }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        match step {
            0 => Ok(self.out = results),
            _ => Err(no_such_step(OpType::Mul, step)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
