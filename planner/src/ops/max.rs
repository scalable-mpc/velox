//! `Max { a, b }` / `MaxPub { a, c }` — `max(a, b)`. `|a|, |b| < 2^{ℓ−2}`.
//! `Relu(x)` is `MaxPub(x, 0)`.
//!
//! Rounds: 3 + tree levels (9 at ℓ = 61). Steps: the [`DReLU`] pipeline
//! on `a − b`, then one multiply, the select: `max = DReLU · (a − b) + b`.
//! Preprocessing: one edaBit per element.

use anyhow::Result;
use fields::{MersennePrimeField, ProtocolField};

use crate::{api::application::OpResult, primitives::edabit::EdaBit};

use super::{drelu::DReLU, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps(ell: usize) -> Vec<OpStep> {
    let mut steps = super::drelu::steps(ell);
    steps.push(OpStep::new(EngineOperationType::Multiply, 1));
    steps
}

pub struct Max<F: ProtocolField + MersennePrimeField> {
    diff: Vec<E<F>>,
    other: Vec<E<F>>,
    drelu: DReLU<F>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField + MersennePrimeField> Max<F> {
    pub fn new(a: Vec<E<F>>, other: Vec<E<F>>, eda: Vec<EdaBit<F>>) -> Self {
        let diff: Vec<E<F>> = a.iter().zip(other.iter()).map(|(l, r)| l - r).collect();
        Self { drelu: DReLU::new(diff.clone(), eda), diff, other, out: Vec::new() }
    }
}

impl<F: ProtocolField + MersennePrimeField> Operation<F> for Max<F> {
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
            // DReLU · (a − b) + b
            self.out = results.iter().zip(self.other.iter()).map(|(p, b)| p + b).collect();
            Ok(())
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
