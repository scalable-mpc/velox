//! `Mod2m { x, m }` — `x mod 2^m`, exactly, for an unsigned
//! `0 ≤ x < 2^{ℓ−1−κ}` and `1 ≤ m ≤ ℓ − 1 − κ` (Catrina–de Hoogh's Mod2m,
//! with the carry tree as its bitwise less-than).
//!
//! Rounds: `1 + ⌈log₂ m⌉` (7 at `m = 63`). Steps, `L` being the tree's
//! levels:
//!
//! - step `0`, reveal: open `c = x + r`, with `r = Σ_{i<ℓ−1} 2^i [r_i]` the
//!   edaBit without its top bit. `x + r < 2^{ℓ−1−κ} + 2^{ℓ−1} < p`, so `c`
//!   is the integer sum, and it hides `x` up to a statistical distance of
//!   `x / 2^{ℓ−1} < 2^{−κ}`.
//! - steps `1..=L`, multiply: the carry tree over the public bits of
//!   `c' = c mod 2^m` and `¬[r_i]`, `i < m`, gives `[u] = [c' < r_lo]` with
//!   `r_lo = r mod 2^m`.
//! - then, locally: `[x mod 2^m] = c' − [r_lo] + 2^m · [u]` — `c' − r_lo` is
//!   `x mod 2^m` or that minus `2^m`, and `u` says which.
//!
//! Preprocessing: one edaBit per element.

use anyhow::{bail, Result};
use fields::ProtocolField;

use crate::{
    api::application::{OpType, OpResult},
    primitives::{
        carry_tree::{CarryTree, Slot},
        edabit::EdaBit,
        mersenne,
    },
};

use super::{no_such_step, EngineOperationType, EngineOperands, Operation, OpStep, E};

/// κ: the opened `c` hides `x` up to a statistical distance of `2^{−κ}`.
pub const STATISTICAL_MASK_BITS: usize = 40;

/// `m` must leave the mask its `κ` bits above `x`: `1 ≤ m ≤ ℓ − 1 − κ`.
/// Checked both when the plan is compiled and when the op is run.
pub fn check_m(m: usize, ell: usize) -> Result<()> {
    let max = ell.saturating_sub(1 + STATISTICAL_MASK_BITS);
    if m == 0 || m > max {
        if max == 0 {
            bail!("Mod2m needs an ℓ above {}, and this field has ℓ = {}", 1 + STATISTICAL_MASK_BITS, ell);
        }
        bail!("Mod2m by {} bits is outside 1..={} for an {}-bit field", m, max, ell);
    }
    Ok(())
}

pub fn steps(m: usize) -> Vec<OpStep> {
    let mut steps = vec![OpStep::new(EngineOperationType::Reveal, 1)];
    steps.extend(CarryTree::new(m).level_widths().into_iter().map(|w| OpStep::new(EngineOperationType::Multiply, w)));
    steps
}

pub struct Mod2m<F: ProtocolField> {
    tree: CarryTree,
    x: Vec<E<F>>,
    m: usize,
    eda: Vec<EdaBit<F>>,
    /// After the reveal: per element, the public `c' = c mod 2^m`.
    c_low: Vec<E<F>>,
    /// After the reveal: per element, the tree slots at the current level.
    slots: Vec<Vec<Slot<F>>>,
    /// `[x mod 2^m]`, per element, once the tree has folded.
    out: Vec<E<F>>,
}

impl<F: ProtocolField> Mod2m<F> {
    pub fn new(x: Vec<E<F>>, m: usize, eda: Vec<EdaBit<F>>) -> Self {
        Self { tree: CarryTree::new(m), x, m, eda, c_low: Vec::new(), slots: Vec::new(), out: Vec::new() }
    }

    /// `[x mod 2^m] = c' − [r_lo] + 2^m · (1 − carry)`, per element, with
    /// `[r_lo] = Σ_{i<m} 2^i [r_i]`.
    fn finish(&mut self) {
        let one = E::<F>::one();
        let powers: Vec<E<F>> = (0..=self.m).map(mersenne::pow2::<F>).collect();
        self.out = self
            .c_low
            .iter()
            .zip(self.eda.iter())
            .zip(self.slots.iter())
            .map(|((c_low, eda), slots)| {
                let r_low = eda.bits[..self.m].iter().zip(&powers).fold(E::<F>::zero(), |acc, (b, pow)| acc + b * pow);
                c_low - r_low + &powers[self.m] * (&one - self.tree.carry(slots))
            })
            .collect();
    }
}

impl<F: ProtocolField> Operation<F> for Mod2m<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        match step {
            0 => {
                let top = mersenne::pow2::<F>(mersenne::ell::<F>() - 1);
                EngineOperands::Reveal(
                    self.x.iter().zip(self.eda.iter()).map(|(x, r)| x + &r.value - r.msb() * &top).collect(),
                )
            }
            s => EngineOperands::multiply(self.slots.iter().flat_map(|slots| self.tree.level_operands(s - 1, slots))),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let levels = self.tree.num_levels();
        let one = E::<F>::one();
        match step {
            // The reveal: `c' = c mod 2^m` is public; its bits seed the tree.
            0 => {
                let mask = (1u128 << self.m) - 1;
                for (c, eda) in results.iter().zip(self.eda.iter()) {
                    let c_low = mersenne::canonical::<F>(c) & mask;
                    let c_bits: Vec<u64> = (0..self.m).map(|i| ((c_low >> i) & 1) as u64).collect();
                    let not_r: Vec<E<F>> = eda.bits[..self.m].iter().map(|r| &one - r).collect();
                    self.c_low.push(mersenne::from_u128::<F>(c_low));
                    self.slots.push(self.tree.leaves(&c_bits, &not_r));
                }
                if levels == 0 {
                    self.finish();
                }
                Ok(())
            }
            // A tree level: fold each element's slots; after the last level
            // the carry gives `[u] = [c' < r_lo]`.
            s if s <= levels => {
                let level = s - 1;
                let width = self.tree.level_widths()[level];
                for (slots, chunk) in self.slots.iter_mut().zip(results.chunks(width)) {
                    *slots = self.tree.level_absorb(level, slots, chunk);
                }
                if s == levels {
                    self.finish();
                }
                Ok(())
            }
            s => Err(no_such_step(OpType::Mod2m { m: self.m }, s)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}
