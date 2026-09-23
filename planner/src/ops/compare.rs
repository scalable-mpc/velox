//! `Compare { a, b }` / `ComparePub { a, c }` — `[a < b]`, a sharing of 0
//! or 1. `|a|, |b| < 2^{ℓ−2}`.
//!
//! Rounds: 2 + tree levels (8 at ℓ = 61). Steps: the [`DReLU`] pipeline
//! on `a − b`; the result is `1 − DReLU(a − b)`. Preprocessing: one edaBit
//! per element.

use anyhow::Result;
use fields::ProtocolField;

use crate::{api::application::OpResult, primitives::edabit::EdaBit};

use super::{drelu::DReLU, EngineOperands, Operation, OpStep, E};

pub fn steps(ell: usize) -> Vec<OpStep> {
    super::drelu::steps(ell)
}

pub struct Compare<F: ProtocolField> {
    drelu: DReLU<F>,
}

impl<F: ProtocolField> Compare<F> {
    /// `other` is `b` (shared) or `c` (public); the arithmetic is the same.
    pub fn new(a: Vec<E<F>>, other: Vec<E<F>>, eda: Vec<EdaBit<F>>) -> Self {
        let diff = a.iter().zip(other.iter()).map(|(l, r)| l - r).collect();
        Self { drelu: DReLU::new(diff, eda) }
    }
}

impl<F: ProtocolField> Operation<F> for Compare<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        self.drelu.operands(step)
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        self.drelu.on_step_complete(step, results)
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        let one = E::<F>::one();
        OpResult::Shares(self.drelu.result().iter().map(|d| &one - d).collect())
    }
}
