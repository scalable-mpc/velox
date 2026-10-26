//! Uniform masks in `[0, p)` with their bits, over a prime field that is not
//! a Mersenne field: the Planner's prefix, run before op-depth 1. Not an
//! op an application schedules: the Planner runs it itself, on the template
//! of the other ops.
//!
//! Over `p = 2^ℓ − 1` an edaBit of `ℓ` random bits is already a mask (its one
//! extra value `2^ℓ − 1 ≡ 0` is accepted, as Liu et al. do). Over a prime
//! `p = 2^ℓ − 2^k + 1` with `k > 1`, such as the ML-DSA field
//! (`ℓ = 23`, `k = 13`), a candidate `r` of `ℓ` random bits is at least `p`
//! with probability `(2^k − 1)/2^ℓ`, and such a mask would leak its masked
//! value almost completely. The prefix rejects those candidates exactly:
//!
//! - `s` = the number of ones in the top `t = ℓ − k` bits, and `ρ` = the low
//!   `k` bits as a field element. Both are linear in the bits.
//! - `r ≥ p` exactly when `s = t` and `ρ ≠ 0`.
//! - Open `e = ρ · s(s − 1)···(s − t + 1)`. A good candidate, `r = p − 1`
//!   included, opens as 0; a bad one as `t!·ρ ≠ 0`. The opened values of the
//!   kept candidates are all 0, so the opening tells nothing about them.
//!
//! `s(s − 1)···(s − t + 1)` is a polynomial of degree `t` in one variable
//! with public coefficients, evaluated by Paterson–Stockmeyer with blocks of
//! 4: `s², s³, s⁴`, then Horner in `s⁴`.
//!
//! Steps (`m = ⌊t/4⌋`; for `t = 10`: 6 multiplications per candidate,
//! 5 multiplication rounds and a reveal):
//!
//! - step `0`, multiply: `s · s`.
//! - step `1`, multiply: `s² · s` and `s² · s²`.
//! - steps `2..2 + m`, multiply: Horner, `acc · s⁴`.
//! - step `2 + m`, multiply: `ρ · acc`.
//! - step `3 + m`, reveal: `e`.

use anyhow::{bail, Result};
use fields::ProtocolField;

use crate::{
    api::application::OpResult,
    primitives::{edabit::EdaBit, mersenne},
};

use super::{EngineOperationType, EngineOperands, Operation, OpStep, E};

/// Paterson–Stockmeyer block: the powers `s..s⁴` are calculated, and the
/// polynomial is evaluated by Horner in `s⁴`.
const BLOCK: usize = 4;

/// A shortfall of good candidates has probability below `2^-SHORTFALL_BITS`.
const SHORTFALL_BITS: usize = 40;

/// The shape of a prime `p = 2^ℓ − 2^k + 1`: an `ℓ`-bit candidate is below
/// `p` unless its top `ℓ − k` bits are all ones and its low `k` bits are
/// not all zeros.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskShape {
    /// `ℓ`: the bits of a mask.
    pub bits: usize,
    /// `k`: the low bits.
    pub low_bits: usize,
}

impl MaskShape {
    /// The shape of `p`, if `p − 1 = 2^k · (2^t − 1)` with `t ≥ 5`, which
    /// keeps the rejection rate at or below `2^-5` (see
    /// [`candidates_for`](Self::candidates_for)).
    pub fn of_prime(p: u128) -> Option<Self> {
        let low_bits = (p - 1).trailing_zeros() as usize;
        let odd = (p - 1) >> low_bits;
        let top_bits = (odd + 1).checked_ilog2()? as usize;
        if odd + 1 != 1u128 << top_bits || top_bits < 5 {
            return None;
        }
        Some(Self { bits: low_bits + top_bits, low_bits })
    }

    /// `t = ℓ − k`.
    pub fn top_bits(&self) -> usize {
        self.bits - self.low_bits
    }

    /// Candidates to make for `masks` good ones. A candidate is bad with
    /// probability `β = (2^k − 1)/2^ℓ`, so the bad ones among `C` are about
    /// Poisson(`λ = Cβ`), and P(X ≥ S) ≤ (eλ/S)^S ≤ 2^-S when S ≥ 2eλ. The
    /// surplus `S = 40 + ⌈8·masks·β⌉` satisfies both `S ≥ 40` and, for
    /// `β ≤ 2^-5`, `S ≥ 2e·(masks + S)·β`.
    pub fn candidates_for(&self, masks: usize) -> usize {
        if masks == 0 {
            return 0;
        }
        let bad = (8 * masks as u128 * ((1u128 << self.low_bits) - 1)).div_ceil(1u128 << self.bits);
        masks + SHORTFALL_BITS + bad as usize
    }

    /// `m = ⌊t/4⌋`: the Horner steps.
    fn horner_steps(&self) -> usize {
        self.top_bits() / BLOCK
    }
}

/// The prefix's steps, per candidate.
pub fn steps(shape: MaskShape) -> Vec<OpStep> {
    let mut steps = vec![OpStep::new(EngineOperationType::Multiply, 1), OpStep::new(EngineOperationType::Multiply, 2)];
    steps.extend((0..=shape.horner_steps()).map(|_| OpStep::new(EngineOperationType::Multiply, 1)));
    steps.push(OpStep::new(EngineOperationType::Reveal, 1));
    steps
}

/// The prefix as an operation over the candidates.
pub struct UniformMasks<F: ProtocolField> {
    shape: MaskShape,
    candidates: Vec<EdaBit<F>>,
    /// `s(s − 1)···(s − t + 1)`, constant term first.
    coefficients: Vec<E<F>>,
    /// `[s]` per candidate.
    s: Vec<E<F>>,
    /// `[ρ]` per candidate.
    rho: Vec<E<F>>,
    /// `[s^a]` for `a = 0..=4` per candidate, filled by steps 0 and 1.
    powers: Vec<[E<F>; BLOCK + 1]>,
    /// The Horner accumulator, per candidate.
    acc: Vec<E<F>>,
    /// `[e]` after the `ρ` step, then the opened `e`.
    e: Vec<E<F>>,
}

impl<F: ProtocolField> UniformMasks<F> {
    pub fn new(shape: MaskShape, candidates: Vec<EdaBit<F>>) -> Self {
        let s = candidates.iter().map(|c| c.bits[shape.low_bits..].iter().fold(E::<F>::zero(), |acc, b| acc + b)).collect();
        let rho = candidates
            .iter()
            .map(|c| (0..shape.low_bits).fold(E::<F>::zero(), |acc, i| acc + &c.bits[i] * mersenne::pow2::<F>(i)))
            .collect();
        // The coefficients of `s(s − 1)···(s − t + 1)`: multiply by `(s − j)`
        // for each `j < t`.
        let mut coefficients = vec![E::<F>::one()];
        for j in 0..shape.top_bits() {
            let j = E::<F>::from(j as u64);
            let mut next = vec![E::<F>::zero(); coefficients.len() + 1];
            for (i, c) in coefficients.iter().enumerate() {
                next[i + 1] = &next[i + 1] + c;
                next[i] = &next[i] - &(c * &j);
            }
            coefficients = next;
        }
        Self { shape, candidates, coefficients, s, rho, powers: Vec::new(), acc: Vec::new(), e: Vec::new() }
    }

    /// Block `i` of the polynomial, `Σ_{a<4} c_{4i+a}·s^a`: public
    /// coefficients times shared powers, so it is local.
    fn block(&self, i: usize, powers: &[E<F>; BLOCK + 1]) -> E<F> {
        (0..BLOCK)
            .filter_map(|a| self.coefficients.get(BLOCK * i + a).map(|c| c * &powers[a]))
            .fold(E::<F>::zero(), |acc, term| acc + term)
    }
}

impl<F: ProtocolField> Operation<F> for UniformMasks<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        let m = self.shape.horner_steps();
        match step {
            0 => EngineOperands::multiply(self.s.iter().map(|s| (s.clone(), s.clone()))),
            1 => EngineOperands::multiply(
                self.powers.iter().flat_map(|p| [(p[2].clone(), p[1].clone()), (p[2].clone(), p[2].clone())]),
            ),
            _ if step < 2 + m => EngineOperands::multiply(self.acc.iter().zip(&self.powers).map(|(a, p)| (a.clone(), p[BLOCK].clone()))),
            _ if step == 2 + m => EngineOperands::multiply(self.rho.iter().cloned().zip(self.acc.iter().cloned())),
            _ => EngineOperands::Reveal(self.e.clone()),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let m = self.shape.horner_steps();
        match step {
            0 => {
                let one = E::<F>::one();
                let zero = E::<F>::zero();
                self.powers =
                    self.s.iter().zip(results).map(|(s, s2)| [one.clone(), s.clone(), s2, zero.clone(), zero.clone()]).collect();
            }
            1 => {
                for (p, pair) in self.powers.iter_mut().zip(results.chunks_exact(2)) {
                    p[3] = pair[0].clone();
                    p[4] = pair[1].clone();
                }
                self.acc = self.powers.iter().map(|p| self.block(m, p)).collect();
            }
            _ if step < 2 + m => {
                // Horner step for block i = m − 1 − (step − 2).
                let i = m - 1 - (step - 2);
                self.acc = results.iter().zip(&self.powers).map(|(r, p)| r + self.block(i, p)).collect();
            }
            // The ρ step gives `[e]`, the reveal its value.
            _ if step <= 3 + m => self.e = results,
            _ => bail!("the uniform-mask prefix has no step {}", step),
        }
        Ok(())
    }

    /// Per candidate, the opened `e` and the candidate itself; keep those
    /// with `e = 0` ([`keep_good`]).
    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Masked { public: self.e, mask: self.candidates }
    }
}

/// The first `count` candidates whose opened `e` is 0, in candidate order,
/// which every party reads from the same public values.
pub fn keep_good<F: ProtocolField>(result: OpResult<F>, count: usize) -> Result<Vec<EdaBit<F>>> {
    let OpResult::Masked { public, mask } = result else {
        bail!("the uniform-mask prefix returned {:?}", result);
    };
    let good: Vec<EdaBit<F>> = public.iter().zip(mask).filter(|(e, _)| **e == E::<F>::zero()).map(|(_, c)| c).take(count).collect();
    if good.len() < count {
        bail!("{} of the uniform-mask candidates are good and {} are needed", good.len(), count);
    }
    Ok(good)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::MlDsaField;

    type F = MlDsaField;
    const Q: u128 = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;

    fn candidate(r: u128) -> EdaBit<F> {
        let signs: Vec<E<F>> = (0..23).map(|i| if (r >> i) & 1 == 1 { E::<F>::one() } else { -E::<F>::one() }).collect();
        EdaBit::from_signs(&signs, 23).unwrap()
    }

    /// The prefix against a plaintext engine: a plaintext value is a valid
    /// sharing, a multiply is the product and a reveal the identity.
    fn evaluate_in_plaintext(candidates: Vec<EdaBit<F>>) -> OpResult<F> {
        let shape = MaskShape::of_prime(Q).unwrap();
        let mut op = Box::new(UniformMasks::new(shape, candidates));
        for (step, planned) in steps(shape).iter().enumerate() {
            let results = match op.operands(step) {
                EngineOperands::Multiply { x, y } => {
                    assert_eq!(planned.engine_op, EngineOperationType::Multiply);
                    x.iter().zip(&y).map(|(a, b)| a * b).collect()
                }
                EngineOperands::Reveal(v) => {
                    assert_eq!(planned.engine_op, EngineOperationType::Reveal);
                    v
                }
                EngineOperands::MaskedMultiply { .. } => unreachable!(),
            };
            op.on_step_complete(step, results).unwrap();
        }
        op.into_result()
    }

    #[test]
    fn shapes() {
        assert_eq!(MaskShape::of_prime(Q), Some(MaskShape { bits: 23, low_bits: 13 }));
        assert_eq!(MaskShape::of_prime((1 << 61) - 1), Some(MaskShape { bits: 61, low_bits: 1 }));
        assert_eq!(MaskShape::of_prime(101), None, "100 = 4·25, and 26 is not a power of 2");
        assert_eq!(MaskShape::of_prime(97), None, "96 = 32·3: t = 2 is below 5");
    }

    #[test]
    fn six_multiplications_in_five_rounds_then_a_reveal() {
        let steps = steps(MaskShape::of_prime(Q).unwrap());
        let multiplies: usize =
            steps.iter().filter(|s| s.engine_op == EngineOperationType::Multiply).map(|s| s.num_engine_operands).sum();
        assert_eq!((steps.len(), multiplies), (6, 6));
        assert_eq!(steps.last().unwrap().engine_op, EngineOperationType::Reveal);
    }

    #[test]
    fn rejects_exactly_the_candidates_at_or_above_q() {
        let mut values: Vec<u128> = vec![0, 1, 8191, 8192, Q - 2, Q - 1, Q, Q + 1, (1 << 23) - 2, (1 << 23) - 1];
        values.extend((0..8192).map(|low| (1023 << 13) | low));
        let mut state = 0x9e37u64;
        values.extend((0..2000).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as u128) & ((1 << 23) - 1)
        }));
        let OpResult::Masked { public, mask } = evaluate_in_plaintext(values.iter().map(|&r| candidate(r)).collect()) else { panic!() };
        for ((r, e), c) in values.iter().zip(&public).zip(&mask) {
            assert_eq!(*e == E::<F>::zero(), *r < Q, "r = {r}");
            assert_eq!(F::canonical_u128(&c.value), Some(r % Q));
        }
    }

    #[test]
    fn keeps_the_first_good_candidates_in_order() {
        let values = [Q + 5, 3, Q - 1, (1 << 23) - 1, 7];
        let result = evaluate_in_plaintext(values.iter().map(|&r| candidate(r)).collect());
        let kept: Vec<u128> =
            keep_good(result, 3).unwrap().iter().map(|c| F::canonical_u128(&c.value).unwrap()).collect();
        assert_eq!(kept, vec![3, Q - 1, 7]);
        assert!(keep_good(evaluate_in_plaintext(vec![candidate(Q), candidate(1)]), 2).is_err(), "a shortfall is an error");
    }

    #[test]
    fn surplus_covers_the_rejections() {
        let shape = MaskShape::of_prime(Q).unwrap();
        assert_eq!(shape.candidates_for(0), 0);
        assert_eq!(shape.candidates_for(1), 1 + 40 + 1);
        // 2,070 masks: about 2 rejections expected, 17 extra by the bound.
        assert_eq!(shape.candidates_for(2070), 2070 + 40 + 17);
    }
}
