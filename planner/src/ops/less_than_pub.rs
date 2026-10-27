//! `LessThanPub { x, c }` — `[x < c]` for a shared `x` and a public `c`,
//! both read as integers in `[0, p)`, over a field with uniform masks
//! (Rabbit's linear form, Makri et al., CRYPTO 2021). The result is exact:
//! the mask is uniform in `[0, p)`, so no statistical bits are needed and
//! `x` has no bound.
//!
//! Rounds: `1 + ⌈log₂ mask_bits⌉`, with `mask_bits` the edaBit width (6 over
//! the ML-DSA field, `mask_bits = 23`). Steps, `L` being the tree's levels:
//!
//! - step `0`, reveal: open `a = x + r mod p`. Then `b = a − c mod p` and
//!   `[a < c]` are public.
//! - steps `1..=L`, multiply: two carry trees per element, side by side,
//!   for `[a < r]` and `[b < r]` over the bits of `r`.
//! - then, locally: `[x < c] = [a < c] − [a < r] + [b < r]`.
//!
//! Why: as integers, `x − c = b − r + p·([a < r] − [a < c])`, and
//! `b − r + p·[b < r]` is in `[0, p)`. `x − c` is in `[0, p)` when `x ≥ c`
//! and in `(−p, 0)` when `x < c`, so `[a < r] − [a < c] − [b < r]` is 0 or
//! −1, and `−1` exactly when `x < c`. Over a Mersenne field the edaBit can be
//! `r = p` (all ones); the identity holds for it too.
//!
//! Preprocessing: one edaBit per element.

use anyhow::Result;
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

/// `bits` is the edaBit width.
pub fn steps(bits: usize) -> Vec<OpStep> {
    let mut steps = vec![OpStep::new(EngineOperationType::Reveal, 1)];
    steps.extend(CarryTree::new(bits).level_widths().into_iter().map(|w| OpStep::new(EngineOperationType::Multiply, 2 * w)));
    steps
}

pub struct LessThanPub<F: ProtocolField> {
    tree: CarryTree,
    x: Vec<E<F>>,
    c: Vec<E<F>>,
    eda: Vec<EdaBit<F>>,
    /// After the reveal: per element, the public `[a < c]`.
    a_below_c: Vec<E<F>>,
    /// After the reveal: per element, the slots of the trees for `[a < r]`
    /// and `[b < r]` at the current level.
    slots: Vec<[Vec<Slot<F>>; 2]>,
    /// `[x < c]`, per element, once the trees have folded.
    out: Vec<E<F>>,
}

impl<F: ProtocolField> LessThanPub<F> {
    pub fn new(x: Vec<E<F>>, c: Vec<E<F>>, bits: usize, eda: Vec<EdaBit<F>>) -> Self {
        Self { tree: CarryTree::new(bits), x, c, eda, a_below_c: Vec::new(), slots: Vec::new(), out: Vec::new() }
    }

    /// `[a < r] = 1 − carry_a` and `[b < r] = 1 − carry_b`, so
    /// `[x < c] = [a < c] + carry_a − carry_b`.
    fn finish(&mut self) {
        self.out = self
            .a_below_c
            .iter()
            .zip(&self.slots)
            .map(|(a_below_c, [a, b])| a_below_c + self.tree.carry(a) - self.tree.carry(b))
            .collect();
    }
}

impl<F: ProtocolField> Operation<F> for LessThanPub<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        match step {
            0 => EngineOperands::Reveal(self.x.iter().zip(&self.eda).map(|(x, r)| x + &r.value).collect()),
            s => EngineOperands::multiply(
                self.slots.iter().flat_map(|[a, b]| [self.tree.level_operands(s - 1, a), self.tree.level_operands(s - 1, b)].concat()),
            ),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let levels = self.tree.num_levels();
        match step {
            // The reveal: the bits of `a` and `b` seed the two trees.
            0 => {
                let p = F::PRIME_MODULUS.expect("LessThanPub runs over a prime field");
                let bits = self.tree.bits();
                let bits_of = |v: u128| -> Vec<u64> { (0..bits).map(|i| ((v >> i) & 1) as u64).collect() };
                let one = E::<F>::one();
                for ((opened, c), eda) in results.iter().zip(&self.c).zip(&self.eda) {
                    let a = mersenne::canonical::<F>(opened);
                    let c = mersenne::canonical::<F>(c);
                    let b = (a + p - c) % p;
                    let not_r: Vec<E<F>> = eda.bits.iter().map(|r| &one - r).collect();
                    self.a_below_c.push(if a < c { one.clone() } else { E::<F>::zero() });
                    self.slots.push([self.tree.leaves(&bits_of(a), &not_r), self.tree.leaves(&bits_of(b), &not_r)]);
                }
                if levels == 0 {
                    self.finish();
                }
                Ok(())
            }
            // A tree level: each element's products are the `a` tree's, then
            // the `b` tree's.
            s if s <= levels => {
                let level = s - 1;
                let width = self.tree.level_widths()[level];
                for ([a, b], chunk) in self.slots.iter_mut().zip(results.chunks(2 * width)) {
                    *a = self.tree.level_absorb(level, a, &chunk[..width]);
                    *b = self.tree.level_absorb(level, b, &chunk[width..]);
                }
                if s == levels {
                    self.finish();
                }
                Ok(())
            }
            s => Err(no_such_step(OpType::LessThanPub, s)),
        }
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{Mersenne61Field, MlDsaField};

    /// Run the op on each `(x, c, r)` with a plaintext engine, and assert
    /// that the result is `x < c`. A plaintext value is a valid sharing, a
    /// multiply is the product and a reveal the identity. `r` is the mask,
    /// as an edaBit `bits` wide.
    fn check_less_than_in_plaintext<F: ProtocolField>(cases: &[(u128, u128, u128)], bits: usize) {
        let element = mersenne::from_u128::<F>;
        let x = cases.iter().map(|&(x, _, _)| element(x)).collect();
        let c = cases.iter().map(|&(_, c, _)| element(c)).collect();
        let eda = cases
            .iter()
            .map(|&(_, _, r)| {
                let signs: Vec<E<F>> = (0..bits).map(|i| if (r >> i) & 1 == 1 { E::<F>::one() } else { -E::<F>::one() }).collect();
                EdaBit::from_signs(&signs, bits).unwrap()
            })
            .collect();
        let mut op = Box::new(LessThanPub::new(x, c, bits, eda));
        for (step, planned) in steps(bits).iter().enumerate() {
            let operands = op.operands(step);
            assert_eq!(operands.batch_type(), planned.engine_op);
            assert_eq!(operands.len(), planned.num_engine_operands * cases.len());
            let results = match operands {
                EngineOperands::Multiply { x, y } => x.iter().zip(&y).map(|(a, b)| a * b).collect(),
                EngineOperands::Reveal(v) => v,
                EngineOperands::MaskedMultiply { .. } => unreachable!(),
            };
            op.on_step_complete(step, results).unwrap();
        }
        let OpResult::Shares(out) = op.into_result() else { panic!("LessThanPub returns shares") };
        for (&(x, c, r), got) in cases.iter().zip(&out) {
            assert_eq!(mersenne::canonical::<F>(got), u128::from(x < c), "x = {x}, c = {c}, r = {r}");
        }
    }

    /// Every combination of edge values for `x`, `c` and the mask `r`, over
    /// the ML-DSA field: both wraps (`a < r`, `b < r`) occur.
    #[test]
    fn edges_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let edges = [0, 1, 2, q / 2, q / 2 + 1, q - 2, q - 1];
        let mut cases = Vec::new();
        for x in edges {
            for c in edges {
                cases.extend(edges.map(|r| (x, c, r)));
            }
        }
        check_less_than_in_plaintext::<MlDsaField>(&cases, 23);
    }

    /// Random `x`, `c` and `r` over the ML-DSA field, and `x` near the bound
    /// of the ML-DSA-65 rejection test, `c = 2(γ1 − β) − 1`.
    #[test]
    fn random_values_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u128::from(state) % q
        };
        let bound = 2 * ((1 << 19) - 196) - 1;
        let mut cases: Vec<(u128, u128, u128)> = (0..2000).map(|_| (next(), next(), next())).collect();
        cases.extend((0..200).map(|i| (bound - 100 + i, bound, next())));
        check_less_than_in_plaintext::<MlDsaField>(&cases, 23);
    }

    /// Over a Mersenne field the edaBit can be `r = p` (all ones), and the
    /// result is still exact.
    #[test]
    fn a_mersenne_mask_of_all_ones() {
        let p = (1u128 << 61) - 1;
        let cases = [(0, 0, p), (0, 1, p), (p - 1, p - 1, p), (5, p - 1, p), (p - 1, 5, p), (7, 9, 3), (9, 7, p - 3)];
        check_less_than_in_plaintext::<Mersenne61Field>(&cases, 61);
    }

    #[test]
    fn six_rounds_over_the_ml_dsa_field() {
        let steps = steps(23);
        assert_eq!(steps.len(), 6);
        assert_eq!(steps[0].engine_op, EngineOperationType::Reveal);
    }
}
