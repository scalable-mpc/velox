//! `Add { x, y }` — `x + y`, elementwise.
//!
//! Rounds: 0. Steps: none; the sum is formed at construction.

use anyhow::Result;
use fields::ProtocolField;

use crate::api::application::{OpType, OpResult};

use super::{no_such_step, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    Vec::new()
}

pub struct Add<F: ProtocolField> {
    out: Vec<E<F>>,
}

impl<F: ProtocolField> Add<F> {
    pub fn new(x: Vec<E<F>>, y: Vec<E<F>>) -> Self {
        Self { out: x.iter().zip(y.iter()).map(|(l, r)| l + r).collect() }
    }
}

impl<F: ProtocolField> Operation<F> for Add<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        EngineOperands::Reveal(Vec::new())
    }

    fn on_step_complete(&mut self, step: usize, _results: Vec<E<F>>) -> Result<()> {
        Err(no_such_step(OpType::Add, step))
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
