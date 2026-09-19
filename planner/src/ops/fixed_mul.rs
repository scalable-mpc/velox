//! `FixedMul { x, y, d }` — `Trunc_d(x · y)` in one round: a fixed-point
//! multiplication with `d` fractional bits (Liu et al. Protocol 3.2,
//! ΠFixed-Mult); error `{0, ±1, ±2}`. `|x · y| < 2^{ℓ−2}`.
//!
//! Rounds: 1. Steps: one masked multiply — the engine opens
//! `x·y + r + 2^{ℓ−2}` and registers the product for verification.
//! Preprocessing: one edaBit per element.

use anyhow::Result;
use fields::{MersennePrimeField, ProtocolField};

use crate::{
    api::application::{OpType, OpResult},
    primitives::edabit::EdaBit,
};

use super::{fixed_point, no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

pub fn steps() -> Vec<OpStep> {
    vec![OpStep::new(EngineOperationType::MaskedMultiply, 1)]
}

pub struct FixedMul<F: ProtocolField + MersennePrimeField> {
    x: Vec<E<F>>,
    y: Vec<E<F>>,
    d: usize,
    eda: Vec<EdaBit<F>>,
    out: Vec<E<F>>,
}

impl<F: ProtocolField + MersennePrimeField> FixedMul<F> {
    pub fn new(x: Vec<E<F>>, y: Vec<E<F>>, d: usize, eda: Vec<EdaBit<F>>) -> Self {
        Self { x, y, d, eda, out: Vec::new() }
    }
}

impl<F: ProtocolField + MersennePrimeField> Operation<F> for FixedMul<F> {
    fn operands(&self, _step: usize) -> EngineOperands<F> {
        let offset = fixed_point::offset::<F>();
        EngineOperands::MaskedMultiply {
            x: self.x.clone(),
            y: self.y.clone(),
            mask: self.eda.iter().map(|r| &r.value + &offset).collect(),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        match step {
            0 => Ok(self.out = results.iter().zip(self.eda.iter()).map(|(c, r)| fixed_point::unmask(c, r, self.d)).collect()),
            _ => Err(no_such_step(OpType::FixedMul, step)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
