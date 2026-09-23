//! edaBits: a random value together with sharings of its bits, assembled
//! locally from the engine's random bits.
//!
//! The engine hands random bits out as sharings of `±1` (`rand_bit` builds
//! them as `r / sqrt(r²)`). A 0/1 bit is `[b] = (1 + [s]) / 2`, and `ℓ` of
//! them compose `[r] = Σ 2^i [b_i]` — Liu et al.'s `ΠSolvedBits` output
//! `([r], [r]_B)`, or an edaBit in Escudero et al.'s terms. Both steps are
//! linear, so nothing here communicates. `r` ranges over `[0, 2^ℓ)`; the
//! one value `2^ℓ − 1 ≡ 0 mod p` occurs with probability `2^-ℓ` and is
//! accepted, as the paper does.

use anyhow::{bail, Result};
use fields::ProtocolField;

use super::mersenne;
use lambdaworks_math::field::element::FieldElement;

/// A random `r` with its bits, `bits[i]` the sharing of bit `i` (LSB first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdaBit<F: ProtocolField> {
    pub value: FieldElement<F>,
    pub bits: Vec<FieldElement<F>>,
}

impl<F: ProtocolField> EdaBit<F> {
    /// From `ℓ` sharings of `±1`.
    pub fn from_signs(signs: &[FieldElement<F>]) -> Result<Self> {
        if signs.len() != mersenne::ell::<F>() {
            bail!("an edaBit needs {} sign sharings, got {}", mersenne::ell::<F>(), signs.len());
        }
        let half = FieldElement::<F>::from(2u64).inv().expect("2 is invertible in an odd field");
        let one = FieldElement::<F>::one();
        let bits: Vec<FieldElement<F>> = signs.iter().map(|s| (&one + s) * &half).collect();
        let value = bits
            .iter()
            .enumerate()
            .fold(FieldElement::<F>::zero(), |acc, (i, b)| acc + b * FieldElement::<F>::from(1u64 << i));
        Ok(Self { value, bits })
    }

    /// `[r_{ℓ−1}]`, the sign bit of `r` read as a signed integer.
    pub fn msb(&self) -> &FieldElement<F> {
        &self.bits[mersenne::ell::<F>() - 1]
    }

    /// `[Trunc_d(r)]` from the bits (Liu et al. Theorem 3.1, Protocol 3.1
    /// step 2): shift down by `d` and fill the top `d` bits with the MSB.
    pub fn trunc_shift(&self, d: usize) -> FieldElement<F> {
        let ell = mersenne::ell::<F>();
        assert!(d < ell, "truncation by {} bits of an {}-bit value", d, ell);
        let mut acc = FieldElement::<F>::zero();
        for i in d..ell {
            acc = acc + &self.bits[i] * FieldElement::<F>::from(1u64 << (i - d));
        }
        for i in (ell - d)..ell {
            acc = acc + self.msb() * FieldElement::<F>::from(1u64 << i);
        }
        acc
    }
}

/// The edaBits a circuit consumes, laid out by op-depth in fixed slices so
/// that op-depth `d` draws the same material at every party regardless of
/// scheduling — the same argument the engine makes for its masks.
pub struct EdaBitPool<F: ProtocolField> {
    edabits: Vec<EdaBit<F>>,
    /// `offsets[d − 1]` is where op-depth `d`'s slice starts.
    offsets: Vec<usize>,
    total: usize,
}

impl<F: ProtocolField> EdaBitPool<F> {
    /// Lay out `per_depth[d − 1]` edaBits for op-depth `d`.
    pub fn plan(per_depth: &[usize]) -> Self {
        let mut offsets = Vec::with_capacity(per_depth.len());
        let mut total = 0;
        for count in per_depth {
            offsets.push(total);
            total += count;
        }
        Self { edabits: Vec::new(), offsets, total }
    }

    /// Bits the engine must supply: `ℓ` per edaBit. Zero, without asking
    /// for `ℓ`, when the circuit uses none — which is every circuit over a
    /// non-Mersenne field.
    pub fn bits_needed(&self) -> usize {
        if self.total == 0 {
            return 0;
        }
        self.total * mersenne::ell::<F>()
    }

    /// Assemble the pool from the first `bits_needed()` engine signs.
    pub fn fill(&mut self, signs: &[FieldElement<F>]) -> Result<()> {
        if signs.len() < self.bits_needed() {
            bail!("the edaBit pool needs {} random bits, got {}", self.bits_needed(), signs.len());
        }
        if self.total == 0 {
            return Ok(());
        }
        self.edabits = signs[..self.bits_needed()]
            .chunks(mersenne::ell::<F>())
            .map(EdaBit::from_signs)
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// The first `count` edaBits of op-depth `depth`'s slice. Read, not
    /// drained, so re-running a depth draws the same material.
    pub fn for_depth(&self, depth: usize, count: usize) -> Result<Vec<EdaBit<F>>> {
        let Some(&start) = depth.checked_sub(1).and_then(|i| self.offsets.get(i)) else {
            bail!("op-depth {} is outside the {} depths declared", depth, self.offsets.len());
        };
        let end = self.offsets.get(depth).copied().unwrap_or(self.total);
        if start + count > end {
            bail!("op-depth {} needs {} edaBits but declared {}", depth, count, end - start);
        }
        if start + count > self.edabits.len() {
            bail!("the edaBit pool is not filled: op-depth {} needs [{}..{}) of {}", depth, start, start + count, self.edabits.len());
        }
        Ok(self.edabits[start..start + count].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{mersenne_31::Mersenne31Field, Mersenne61Field};

    fn signs_of<F: ProtocolField>(r: u64) -> Vec<FieldElement<F>> {
        (0..mersenne::ell::<F>())
            .map(|i| if (r >> i) & 1 == 1 { FieldElement::<F>::one() } else { -FieldElement::<F>::one() })
            .collect()
    }

    /// Plaintext `Trunc_d` of a field element read as a signed integer.
    fn trunc_ref<F: ProtocolField>(r: u64, d: usize) -> u64 {
        let p = (1u64 << mersenne::ell::<F>()) - 1;
        if r <= (p - 1) / 2 { r >> d } else { (p - ((p - r) >> d)) % p }
    }

    fn check<F: ProtocolField>() {
        let p = (1u64 << mersenne::ell::<F>()) - 1;
        for r in [0u64, 1, 5, p / 2, p / 2 + 1, p - 2, p - 1] {
            let e = EdaBit::<F>::from_signs(&signs_of::<F>(r)).unwrap();
            assert_eq!(mersenne::canonical::<F>(&e.value), r % p, "value of r={r}");
            for i in 0..mersenne::ell::<F>() {
                assert_eq!(mersenne::canonical::<F>(&e.bits[i]), (r >> i) & 1, "bit {i} of r={r}");
            }
            assert_eq!(mersenne::canonical::<F>(e.msb()), r >> (mersenne::ell::<F>() - 1));
            for d in [1usize, 2, 7, 16, mersenne::ell::<F>() - 2] {
                // r = p reads as 0 in the field; Theorem 3.1 is stated for r < p.
                if r == p { continue; }
                assert_eq!(mersenne::canonical::<F>(&e.trunc_shift(d)), trunc_ref::<F>(r, d), "trunc_{d} of r={r}");
            }
        }
        assert!(EdaBit::<F>::from_signs(&signs_of::<F>(3)[..mersenne::ell::<F>() - 1]).is_err());
    }

    #[test]
    fn bits_and_truncation_match_integers() {
        check::<Mersenne61Field>();
        check::<Mersenne31Field>();
    }

    #[test]
    fn pool_slices_by_depth() {
        type F = Mersenne61Field;
        let mut pool = EdaBitPool::<F>::plan(&[2, 0, 1]);
        assert_eq!(pool.bits_needed(), 3 * 61);
        assert!(pool.for_depth(1, 1).is_err(), "not filled yet");
        let signs: Vec<_> = (0..3).flat_map(|r| signs_of::<F>(100 + r)).collect();
        pool.fill(&signs).unwrap();
        assert_eq!(mersenne::canonical::<F>(&pool.for_depth(1, 2).unwrap()[1].value), 101);
        assert_eq!(mersenne::canonical::<F>(&pool.for_depth(3, 1).unwrap()[0].value), 102);
        assert_eq!(pool.for_depth(1, 1).unwrap(), pool.for_depth(1, 1).unwrap(), "read, not drained");
        assert!(pool.for_depth(2, 1).is_err(), "depth 2 declared none");
        assert!(pool.for_depth(1, 3).is_err());
        assert!(pool.for_depth(4, 0).is_err());
        assert!(pool.for_depth(0, 0).is_err());
    }
}
