//! `Min { a, b }` / `MinPub { a, c }` — `min(a, b)`. `|a|, |b| < 2^{ℓ−2}`.
//!
//! Rounds: 3 + tree levels (9 at ℓ = 61). Steps: the [`DReLU`] pipeline
//! on `a − b`, then one multiply, the select: `min = a − DReLU · (a − b)`.
//! Preprocessing: one edaBit per element.

use anyhow::Result;
use fields::ProtocolField;

use crate::{api::application::OpResult, primitives::edabit::EdaBit};

use super::{drelu::DReLU, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps(ell: usize) -> Vec<OpStep> {
    let mut steps = super::drelu::steps(ell);
    steps.push(OpStep::new(EngineOperationType::Multiply, 1));
    steps
}

pub struct Min<F: ProtocolField> {
    a: Vec<E<F>>,
    diff: Vec<E<F>>,
    drelu: DReLU<F>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField> Min<F> {
    pub fn new(a: Vec<E<F>>, other: Vec<E<F>>, eda: Vec<EdaBit<F>>) -> Self {
        let diff: Vec<E<F>> = a.iter().zip(other.iter()).map(|(l, r)| l - r).collect();
        Self { drelu: DReLU::new(diff.clone(), eda), a, diff, out: Vec::new() }
    }
}

impl<F: ProtocolField> Operation<F> for Min<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        if step < self.drelu.steps() {
            self.drelu.operands(step)
        } else {
            EngineOperands::multiply(self.drelu.result().iter().cloned().zip(self.diff.iter().cloned()))
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        if step < self.drelu.steps() {
            self.drelu.on_step_complete(step, results)
        } else {
            // a − DReLU · (a − b)
            self.out = self.a.iter().zip(results.iter()).map(|(a, p)| a - p).collect();
            Ok(())
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
