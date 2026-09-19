//! `Truncate { x, d }` — `Trunc_d(x)`: drop the low `d` bits, keep the
//! sign; error `{0, ±1, ±2}`. `|x| < 2^{ℓ−2}`.
//!
//! Rounds: 1. Steps: one reveal, of `x + 2^{ℓ−2} + r`. Preprocessing: one
//! edaBit per element.

use anyhow::Result;
use fields::{MersennePrimeField, ProtocolField};

use crate::{
    api::application::{OpType, OpResult},
    primitives::edabit::EdaBit,
};

use super::{fixed_point, no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    vec![OpStep::new(EngineOperationType::Reveal, 1)]
}

pub struct Truncate<F: ProtocolField + MersennePrimeField> {
    x: Vec<E<F>>,
    d: usize,
    eda: Vec<EdaBit<F>>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField + MersennePrimeField> Truncate<F> {
    pub fn new(x: Vec<E<F>>, d: usize, eda: Vec<EdaBit<F>>) -> Self {
        Self { x, d, eda, out: Vec::new() }
    }
}

impl<F: ProtocolField + MersennePrimeField> Operation<F> for Truncate<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        let offset = fixed_point::offset::<F>();
        EngineOperands::Reveal(self.x.iter().zip(self.eda.iter()).map(|(x, r)| x + &offset + &r.value).collect())
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        match step {
            0 => Ok(self.out = results.iter().zip(self.eda.iter()).map(|(c, r)| fixed_point::unmask(c, r, self.d)).collect()),
            _ => Err(no_such_step(OpType::Truncate, step)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
