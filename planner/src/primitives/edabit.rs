//! edaBits: a random value together with sharings of its bits, assembled
//! locally from the engine's random bits.
//!
//! The engine hands random bits out as sharings of `±1` (`rand_bit` builds
//! them as `r / sqrt(r²)`). A 0/1 bit is `[b] = (1 + [s]) / 2`, and `p_bits`
//! of them compose `[r] = Σ 2^i [b_i]` — Liu et al.'s `ΠSolvedBits` output
//! `([r], [r]_B)`, or an edaBit in Escudero et al.'s terms. Both steps are
//! linear, so nothing here communicates. Over `p = 2^p_bits − 1`, `r` ranges
//! over `[0, 2^p_bits)`; the one value `2^p_bits − 1 ≡ 0 mod p` occurs with
//! probability `2^-p_bits` and is accepted, as the paper does. Over another
//! prime field, these are candidates, and the uniform-mask prefix
//! (`uniform_mask`) keeps only those below `p`.

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
    /// From `bits` sharings of `±1`: `p_bits` over a Mersenne field, the mask
    /// width of a uniform-mask candidate otherwise.
    pub fn from_signs(signs: &[FieldElement<F>], bits: usize) -> Result<Self> {
        if signs.len() != bits {
            bail!("an edaBit needs {} sign sharings, got {}", bits, signs.len());
        }
        let half = FieldElement::<F>::from(2u64).inv().expect("2 is invertible in an odd field");
        let one = FieldElement::<F>::one();
        let bits: Vec<FieldElement<F>> = signs.iter().map(|s| (&one + s) * &half).collect();
        let value = bits
            .iter()
            .enumerate()
            .fold(FieldElement::<F>::zero(), |acc, (i, b)| acc + b * mersenne::pow2::<F>(i));
        Ok(Self { value, bits })
    }

    /// `[r_{p_bits−1}]`, the sign bit of `r` read as a signed integer.
    pub fn msb(&self) -> &FieldElement<F> {
        &self.bits[mersenne::p_bits::<F>() - 1]
    }

    /// `[Trunc_d(r)]` from the bits (Liu et al. Theorem 3.1, Protocol 3.1
    /// step 2): shift down by `d` and fill the top `d` bits with the MSB.
    pub fn trunc_shift(&self, d: usize) -> FieldElement<F> {
        let p_bits = mersenne::p_bits::<F>();
        assert!(d < p_bits, "truncation by {} bits of an {}-bit value", d, p_bits);
        let mut acc = FieldElement::<F>::zero();
        for i in d..p_bits {
            acc = acc + &self.bits[i] * mersenne::pow2::<F>(i - d);
        }
        for i in (p_bits - d)..p_bits {
            acc = acc + self.msb() * mersenne::pow2::<F>(i);
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

    /// edaBits the pool holds when filled.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Fill the pool, op-depth 1's slice first. Over a Mersenne field these
    /// are the edaBits as made; over another prime field, the candidates the
    /// uniform-mask prefix kept.
    pub fn fill(&mut self, edabits: Vec<EdaBit<F>>) -> Result<()> {
        if edabits.len() != self.total {
            bail!("the edaBit pool needs {} edaBits, got {}", self.total, edabits.len());
        }
        self.edabits = edabits;
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
    use fields::{mersenne_31::Mersenne31Field, Mersenne127Field, Mersenne61Field};

    fn signs_of<F: ProtocolField>(r: u128) -> Vec<FieldElement<F>> {
        (0..mersenne::p_bits::<F>())
            .map(|i| if (r >> i) & 1 == 1 { FieldElement::<F>::one() } else { -FieldElement::<F>::one() })
            .collect()
    }

    /// Plaintext `Trunc_d` of a field element read as a signed integer.
    fn trunc_ref<F: ProtocolField>(r: u128, d: usize) -> u128 {
        let p = (1u128 << mersenne::p_bits::<F>()) - 1;
        if r <= (p - 1) / 2 { r >> d } else { (p - ((p - r) >> d)) % p }
    }

    fn check<F: ProtocolField>() {
        let p = (1u128 << mersenne::p_bits::<F>()) - 1;
        for r in [0u128, 1, 5, p / 2, p / 2 + 1, p - 2, p - 1] {
            let e = EdaBit::<F>::from_signs(&signs_of::<F>(r), mersenne::p_bits::<F>()).unwrap();
            assert_eq!(mersenne::canonical::<F>(&e.value), r % p, "value of r={r}");
            for i in 0..mersenne::p_bits::<F>() {
                assert_eq!(mersenne::canonical::<F>(&e.bits[i]), (r >> i) & 1, "bit {i} of r={r}");
            }
            assert_eq!(mersenne::canonical::<F>(e.msb()), r >> (mersenne::p_bits::<F>() - 1));
            for d in [1usize, 2, 7, 16, mersenne::p_bits::<F>() - 2] {
                // r = p reads as 0 in the field; Theorem 3.1 is stated for r < p.
                if r == p { continue; }
                assert_eq!(mersenne::canonical::<F>(&e.trunc_shift(d)), trunc_ref::<F>(r, d), "trunc_{d} of r={r}");
            }
        }
        assert!(EdaBit::<F>::from_signs(&signs_of::<F>(3)[..mersenne::p_bits::<F>() - 1], mersenne::p_bits::<F>()).is_err());
    }

    #[test]
    fn bits_and_truncation_match_integers() {
        check::<Mersenne61Field>();
        check::<Mersenne31Field>();
        check::<Mersenne127Field>();
    }

    #[test]
    fn pool_slices_by_depth() {
        type F = Mersenne61Field;
        let mut pool = EdaBitPool::<F>::plan(&[2, 0, 1]);
        assert_eq!(pool.total(), 3);
        assert!(pool.for_depth(1, 1).is_err(), "not filled yet");
        let edabits = |count: u128| -> Vec<EdaBit<F>> { (0..count).map(|r| EdaBit::from_signs(&signs_of::<F>(100 + r), 61).unwrap()).collect() };
        assert!(pool.fill(edabits(2)).is_err(), "one short");
        pool.fill(edabits(3)).unwrap();
        assert_eq!(mersenne::canonical::<F>(&pool.for_depth(1, 2).unwrap()[1].value), 101);
        assert_eq!(mersenne::canonical::<F>(&pool.for_depth(3, 1).unwrap()[0].value), 102);
        assert_eq!(pool.for_depth(1, 1).unwrap(), pool.for_depth(1, 1).unwrap(), "read, not drained");
        assert!(pool.for_depth(2, 1).is_err(), "depth 2 declared none");
        assert!(pool.for_depth(1, 3).is_err());
        assert!(pool.for_depth(4, 0).is_err());
        assert!(pool.for_depth(0, 0).is_err());
    }
}
