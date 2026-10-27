//! `CanonicalMod2m { x, m }` — `x mod 2^m` for a shared `x` read as an
//! integer in `[0, p)`, over a field with uniform masks, exactly and with no
//! bound on `x`. ML-DSA-65 and 87 need it with `m = 4`: the high bits `w1`
//! of `w` are the low 4 bits of `−16·w + (q − 1)/2` (Quorus, Protocol 6).
//! `Mod2m` cannot do this: it needs `x < 2^{p_bits−41}` and a Mersenne field.
//!
//! Rounds: `2 + ⌈log₂ mask_bits⌉`, with `mask_bits` the edaBit width (7 over
//! the ML-DSA field). Steps, `L = ⌈log₂ mask_bits⌉` being the levels of the
//! widest tree:
//!
//! - step `0`, reveal: open `a = x + r mod p`, with `r` a uniform mask in
//!   `[0, p)` and its bits. Then `a₀ = a mod 2^m` and
//!   `a₁ = (a + p) mod 2^m` are public; `r' = r mod 2^m` is linear in the
//!   bits of `r`.
//! - steps `1..=L`, multiply: three carry trees per element, side by side:
//!   the wrap bit `[w] = [a < r]` over all `mask_bits` bits, and `[a₀ < r']`,
//!   `[a₁ < r']` over the low `m` bits.
//! - step `L + 1`, multiply: `[w] · ([m₁] − [m₀])`, with
//!   `[m₀] = a₀ − [r'] + 2^m·[a₀ < r']` and `[m₁] = a₁ − [r'] + 2^m·[a₁ < r']`.
//! - then, locally: `[x mod 2^m] = [m₀] + [w]·([m₁] − [m₀])`.
//!
//! Why: as integers, `x = a − r + p·w`. With `w = 0`,
//! `x mod 2^m = (a − r) mod 2^m = m₀`; with `w = 1`,
//! `x mod 2^m = (a + p − r) mod 2^m = m₁`. Over a Mersenne field the edaBit
//! can be `r = p` (all ones): then `a = x`, `w = 1`, and `m₁ = x mod 2^m`.
//!
//! Multiplications per element: the `mask_bits`-bit tree, two `m`-bit trees and 1
//! (36 over the ML-DSA field with `m = 4`). Preprocessing: one edaBit per
//! element.

use anyhow::{bail, Result};
use fields::ProtocolField;

use crate::{
    api::application::OpResult,
    primitives::{
        carry_tree::{CarryTree, Slot},
        edabit::EdaBit,
        mersenne,
    },
};

use super::{EngineOperationType, EngineOperands, Operation, OpStep, E};

/// The multiplications of `tree` at `level`; 0 once the tree has folded.
fn level_width(tree: &CarryTree, level: usize) -> usize {
    tree.level_widths().get(level).copied().unwrap_or(0)
}

/// `bits` is the edaBit width.
pub fn steps(bits: usize, m: usize) -> Vec<OpStep> {
    let (wrap, low) = (CarryTree::new(bits), CarryTree::new(m));
    let levels = wrap.num_levels().max(low.num_levels());
    let mut steps = vec![OpStep::new(EngineOperationType::Reveal, 1)];
    steps.extend(
        (0..levels).map(|l| OpStep::new(EngineOperationType::Multiply, level_width(&wrap, l) + 2 * level_width(&low, l))),
    );
    steps.push(OpStep::new(EngineOperationType::Multiply, 1));
    steps
}

pub struct CanonicalMod2m<F: ProtocolField> {
    /// The tree for the wrap bit `[a < r]`, over all the bits of `r`.
    wrap_tree: CarryTree,
    /// The tree for `[a₀ < r']` and `[a₁ < r']`, over the low `m` bits.
    low_tree: CarryTree,
    x: Vec<E<F>>,
    m: usize,
    eda: Vec<EdaBit<F>>,
    /// After the reveal: per element, the slots of the three trees (wrap,
    /// `a₀`, `a₁`) at the current level.
    slots: Vec<[Vec<Slot<F>>; 3]>,
    /// After the reveal: per element, the public `a₀` and `a₁`.
    a_low: Vec<[E<F>; 2]>,
    /// After the trees: per element, the wrap bit `[w] = [a < r]`.
    w: Vec<E<F>>,
    /// After the trees: per element, `[m₀] = (a − r) mod 2^m`.
    m0: Vec<E<F>>,
    /// After the trees: per element, `[m₁] = (a + p − r) mod 2^m`.
    m1: Vec<E<F>>,
    /// `[x mod 2^m]`, per element, after the last step.
    out: Vec<E<F>>,
}

impl<F: ProtocolField> CanonicalMod2m<F> {
    /// `bits` is the edaBit width, and `eda` holds one edaBit per element
    /// of `x`.
    pub fn new(x: Vec<E<F>>, m: usize, bits: usize, eda: Vec<EdaBit<F>>) -> Self {
        Self {
            wrap_tree: CarryTree::new(bits),
            low_tree: CarryTree::new(m),
            x,
            m,
            eda,
            slots: Vec::new(),
            a_low: Vec::new(),
            w: Vec::new(),
            m0: Vec::new(),
            m1: Vec::new(),
            out: Vec::new(),
        }
    }

    /// The tree steps: the levels of the deeper tree.
    fn tree_levels(&self) -> usize {
        self.wrap_tree.num_levels().max(self.low_tree.num_levels())
    }

    /// After the last tree level, per element: `[w] = 1 − carry` and
    /// `[m_i] = a_i − [r'] + 2^m·(1 − carry_i)`.
    fn finish_trees(&mut self) {
        let one = E::<F>::one();
        let two_m = mersenne::pow2::<F>(self.m);
        for ((slots, a_low), eda) in self.slots.iter().zip(&self.a_low).zip(&self.eda) {
            let r_low = (0..self.m).fold(E::<F>::zero(), |acc, i| acc + &eda.bits[i] * mersenne::pow2::<F>(i));
            let [wrap, low0, low1] = slots;
            self.w.push(&one - self.wrap_tree.carry(wrap));
            self.m0.push(&a_low[0] - &r_low + &two_m * (&one - self.low_tree.carry(low0)));
            self.m1.push(&a_low[1] - &r_low + &two_m * (&one - self.low_tree.carry(low1)));
        }
    }
}

impl<F: ProtocolField> Operation<F> for CanonicalMod2m<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        let levels = self.tree_levels();
        match step {
            0 => EngineOperands::Reveal(self.x.iter().zip(&self.eda).map(|(x, r)| x + &r.value).collect()),
            s if s <= levels => {
                let level = s - 1;
                // A tree that has folded takes no gates.
                let level_operands = |tree: &CarryTree, slots: &[Slot<F>]| {
                    if level < tree.num_levels() {
                        tree.level_operands(level, slots)
                    } else {
                        Vec::new()
                    }
                };
                EngineOperands::multiply(self.slots.iter().flat_map(|[wrap, low0, low1]| {
                    [level_operands(&self.wrap_tree, wrap), level_operands(&self.low_tree, low0), level_operands(&self.low_tree, low1)]
                        .concat()
                }))
            }
            _ => EngineOperands::multiply(self.w.iter().zip(self.m0.iter().zip(&self.m1)).map(|(w, (m0, m1))| (w.clone(), m1 - m0))),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let levels = self.tree_levels();
        match step {
            // The reveal: the bits of `a`, `a₀` and `a₁` seed the trees.
            0 => {
                let p = F::PRIME_MODULUS.expect("CanonicalMod2m runs over a prime field");
                let bits = self.wrap_tree.bits();
                let low_mask = (1u128 << self.m) - 1;
                let bits_of = |v: u128, k: usize| -> Vec<u64> { (0..k).map(|i| ((v >> i) & 1) as u64).collect() };
                let one = E::<F>::one();
                for (opened, eda) in results.iter().zip(&self.eda) {
                    let a = mersenne::canonical::<F>(opened);
                    let (a0, a1) = (a & low_mask, (a + p) & low_mask);
                    let not_r: Vec<E<F>> = eda.bits.iter().map(|r| &one - r).collect();
                    self.slots.push([
                        self.wrap_tree.leaves(&bits_of(a, bits), &not_r),
                        self.low_tree.leaves(&bits_of(a0, self.m), &not_r[..self.m]),
                        self.low_tree.leaves(&bits_of(a1, self.m), &not_r[..self.m]),
                    ]);
                    self.a_low.push([mersenne::from_u128::<F>(a0), mersenne::from_u128::<F>(a1)]);
                }
            }
            // A tree level: each element's products are the wrap tree's, then
            // the two low trees'. A tree that has folded takes none.
            s if s <= levels => {
                let level = s - 1;
                let (wrap_width, low_width) = (level_width(&self.wrap_tree, level), level_width(&self.low_tree, level));
                for (slots, chunk) in self.slots.iter_mut().zip(results.chunks(wrap_width + 2 * low_width)) {
                    let (wrap, low) = chunk.split_at(wrap_width);
                    if wrap_width > 0 {
                        slots[0] = self.wrap_tree.level_absorb(level, &slots[0], wrap);
                    }
                    if low_width > 0 {
                        slots[1] = self.low_tree.level_absorb(level, &slots[1], &low[..low_width]);
                        slots[2] = self.low_tree.level_absorb(level, &slots[2], &low[low_width..]);
                    }
                }
            }
            // The selection: `[x mod 2^m] = [m₀] + [w]·([m₁] − [m₀])`.
            s if s == levels + 1 => {
                self.out = self.m0.iter().zip(results).map(|(m0, product)| m0 + product).collect();
                return Ok(());
            }
            s => bail!("CanonicalMod2m {{ m: {} }} has no step {}", self.m, s),
        }
        if step == levels {
            self.finish_trees();
        }
        Ok(())
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{Mersenne61Field, MlDsaField};

    /// Runs the op with a plaintext engine (a plaintext value is a valid
    /// sharing, a multiply is the product, a reveal the identity) and checks
    /// `x mod 2^m` for each `(x, r)`.
    fn check_in_plaintext<F: ProtocolField>(cases: &[(u128, u128)], m: usize, bits: usize) {
        let x = cases.iter().map(|&(x, _)| mersenne::from_u128::<F>(x)).collect();
        let eda = cases
            .iter()
            .map(|&(_, r)| {
                let signs: Vec<E<F>> = (0..bits).map(|i| if (r >> i) & 1 == 1 { E::<F>::one() } else { -E::<F>::one() }).collect();
                EdaBit::from_signs(&signs, bits).unwrap()
            })
            .collect();
        let mut op = Box::new(CanonicalMod2m::new(x, m, bits, eda));
        for (step, planned) in steps(bits, m).iter().enumerate() {
            let operands = op.operands(step);
            assert_eq!(operands.batch_type(), planned.engine_op, "step {step}");
            assert_eq!(operands.len(), planned.num_engine_operands * cases.len(), "step {step}");
            let results = match operands {
                EngineOperands::Multiply { x, y } => x.iter().zip(&y).map(|(a, b)| a * b).collect(),
                EngineOperands::Reveal(v) => v,
                EngineOperands::MaskedMultiply { .. } => unreachable!(),
            };
            op.on_step_complete(step, results).unwrap();
        }
        let OpResult::Shares(out) = op.into_result() else { panic!("CanonicalMod2m returns shares") };
        for (&(x, r), got) in cases.iter().zip(out) {
            assert_eq!(mersenne::canonical::<F>(&got), x % (1 << m), "x = {x}, r = {r}, m = {m}");
        }
    }

    /// Every pair of edge values for `x` and `r` over the ML-DSA field, for
    /// several `m`: the wrap `a < r` and both low borrows occur.
    #[test]
    fn edges_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let edges = [0, 1, 15, 16, 8191, 8192, q / 2, q - 16, q - 2, q - 1];
        let mut cases = Vec::new();
        for x in edges {
            cases.extend(edges.map(|r| (x, r)));
        }
        for m in [1, 4, 13, 22] {
            check_in_plaintext::<MlDsaField>(&cases, m, 23);
        }
    }

    /// Random `x` and `r` over the ML-DSA field, at the ML-DSA `m = 4`.
    #[test]
    fn random_values_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u128::from(state) % q
        };
        let cases: Vec<(u128, u128)> = (0..3000).map(|_| (next(), next())).collect();
        check_in_plaintext::<MlDsaField>(&cases, 4, 23);
    }

    /// Over a Mersenne field the edaBit can be `r = p` (all ones).
    #[test]
    fn a_mersenne_mask_of_all_ones() {
        let p = (1u128 << 61) - 1;
        let cases = [(0, p), (5, p), (p - 1, p), (p - 1, 3), (12345, p - 7), (p / 3, 1)];
        for m in [1, 4, 40] {
            check_in_plaintext::<Mersenne61Field>(&cases, m, 61);
        }
    }

    #[test]
    fn seven_rounds_and_36_multiplications_at_m_4() {
        let steps = steps(23, 4);
        let multiplies: usize = steps.iter().filter(|s| s.engine_op == EngineOperationType::Multiply).map(|s| s.num_engine_operands).sum();
        assert_eq!((steps.len(), multiplies), (7, 36));
    }
}
