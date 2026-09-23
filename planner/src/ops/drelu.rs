//! The comparison pipeline every comparing op shares: `DReLU(v) = [v ≥ 0]`
//! for a sharing `v` with `|v| < 2^{ℓ−1}` (Liu et al. Protocol 5.1, with the
//! bitwise less-than replaced by the carry tree).
//!
//! Steps, `L` being the tree's levels (6 at ℓ = 61):
//!
//! - step `0`, reveal: open `y = 2v + r`, `r` an edaBit. Over a Mersenne
//!   prime `msb(v) = lsb(2v)`, and `lsb(2v) = y_0 ⊕ r_0 ⊕ [y < r]`.
//! - steps `1..=L`, multiply: the carry tree over the public bits of `y`
//!   and `¬[r]` gives `[c] = [y < r]`.
//! - step `L + 1`, multiply: `[b] · [c]` with `[b] = y_0 ⊕ [r_0]` (affine);
//!   `DReLU = 1 − ([b] + [c] − 2[b][c])`.
//!
//! Not an [`Operation`](super::Operation) itself: it has the template's
//! shape, and `Compare`, `Max`, `Min` delegate their leading steps to it.

use anyhow::{bail, Result};
use fields::ProtocolField;

use crate::primitives::mersenne;

use crate::primitives::{
    carry_tree::{CarryTree, Slot},
    edabit::EdaBit,
};

use super::{EngineOperationType, EngineOperands, OpStep, E};

pub fn steps(ell: usize) -> Vec<OpStep> {
    let tree = CarryTree::new(ell);
    let mut steps = vec![OpStep::new(EngineOperationType::Reveal, 1)];
    steps.extend(tree.level_widths().into_iter().map(|w| OpStep::new(EngineOperationType::Multiply, w)));
    steps.push(OpStep::new(EngineOperationType::Multiply, 1));
    steps
}

pub struct DReLU<F: ProtocolField> {
    tree: CarryTree,
    v: Vec<E<F>>,
    eda: Vec<EdaBit<F>>,
    /// After the reveal: per element, the tree slots at the current level.
    slots: Vec<Vec<Slot<F>>>,
    /// `[b] = y_0 ⊕ [r_0]`, per element.
    b: Vec<E<F>>,
    /// `[c] = [y < r]`, per element, once the tree has folded.
    c: Vec<E<F>>,
    /// `[v ≥ 0]`, per element, after the xor.
    out: Vec<E<F>>,
}

impl<F: ProtocolField> DReLU<F> {
    pub fn new(v: Vec<E<F>>, eda: Vec<EdaBit<F>>) -> Self {
        Self { tree: CarryTree::new(mersenne::ell::<F>()), v, eda, slots: Vec::new(), b: Vec::new(), c: Vec::new(), out: Vec::new() }
    }

    /// Steps in the pipeline: the reveal, the levels, the xor.
    pub fn steps(&self) -> usize {
        self.tree.num_levels() + 2
    }

    /// `[v ≥ 0]`; filled once the last step has completed.
    pub fn result(&self) -> &[E<F>] {
        &self.out
    }

    pub fn operands(&self, step: usize) -> EngineOperands<F> {
        let levels = self.tree.num_levels();
        match step {
            0 => EngineOperands::Reveal(self.v.iter().zip(self.eda.iter()).map(|(v, r)| v + v + &r.value).collect()),
            s if s <= levels => {
                EngineOperands::multiply(self.slots.iter().flat_map(|slots| self.tree.level_operands(s - 1, slots)))
            }
            _ => EngineOperands::multiply(self.b.iter().cloned().zip(self.c.iter().cloned())),
        }
    }

    pub fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let levels = self.tree.num_levels();
        let one = E::<F>::one();
        match step {
            // The reveal: `y` is public — its bits seed the tree, its LSB the xor.
            0 => {
                for (y, eda) in results.iter().zip(self.eda.iter()) {
                    let y_bits: Vec<u64> = (0..mersenne::ell::<F>()).map(|i| mersenne::bit::<F>(y, i)).collect();
                    self.b.push(if y_bits[0] == 1 { &one - &eda.bits[0] } else { eda.bits[0].clone() });
                    let not_r: Vec<E<F>> = eda.bits.iter().map(|r| &one - r).collect();
                    let slots = self.tree.leaves(&y_bits, &not_r);
                    if levels == 0 {
                        self.c.push(&one - self.tree.carry(&slots));
                    }
                    self.slots.push(slots);
                }
            }
            // A tree level: fold each element's slots; after the last level
            // the carry gives `[c] = [y < r]`.
            s if s <= levels => {
                let level = s - 1;
                let width = self.tree.level_widths()[level];
                for (slots, chunk) in self.slots.iter_mut().zip(results.chunks(width)) {
                    *slots = self.tree.level_absorb(level, slots, chunk);
                }
                if level + 1 == levels {
                    self.c = self.slots.iter().map(|slots| &one - self.tree.carry(slots)).collect();
                }
            }
            // The xor: `DReLU = 1 − ([b] + [c] − 2[b][c])`.
            s if s == levels + 1 => {
                let two = E::<F>::from(2u64);
                self.out = self
                    .b
                    .iter()
                    .zip(self.c.iter())
                    .zip(results.iter())
                    .map(|((b, c), bc)| &one - (b + c - &two * bc))
                    .collect();
            }
            s => bail!("the DReLU pipeline has no step {}", s),
        }
        Ok(())
    }
}
