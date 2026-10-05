//! [`ProtocolField`] for the Mersenne-127 base field and its degree-2
//! extension.
//!
//! The base field is wide enough for the statistical checks (127 bits ≥ 60),
//! so verification runs in it directly. The ACSS DZK needs at least 240
//! bits, so `Ext` is the degree-2 extension (254 bits) and two base elements
//! pack into one `Ext` element.

use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsSubFieldOf},
};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::{mersenne_prime::MersennePrimeField, protocol_field::ProtocolField};

use super::{
    extension::{Fp2E, Mersenne127Degree2ExtensionField},
    field::{Mersenne127Field, M127},
};

type FpE = FieldElement<Mersenne127Field>;

/// One serialized base element: the canonical `u128`.
const LIMB_BYTES: usize = 16;

/// ASCII bytes one limb carries: the top byte stays `0x00`, so the limb is
/// below `2^120 < p` and the reduction on decoding is a no-op.
const LIMB_PAYLOAD: usize = LIMB_BYTES - 1;

/// A uniform element from 128 random bits. The fold maps 128 bits onto
/// `[0, p)` with a bias of about `2^−126`, far below every soundness level
/// the protocol uses.
fn from_two_words(lo: u64, hi: u64) -> FpE {
    FpE::new(M127(((hi as u128) << 64) | lo as u128))
}

fn limb_from_bytes_be(bytes: &[u8]) -> Result<FpE, ByteConversionError> {
    let bytes: [u8; LIMB_BYTES] = bytes.try_into().map_err(|_| ByteConversionError::FromBEBytesError)?;
    Ok(FpE::new(M127(u128::from_be_bytes(bytes))))
}

fn limb_from_bytes_le(bytes: &[u8]) -> Result<FpE, ByteConversionError> {
    let bytes: [u8; LIMB_BYTES] = bytes.try_into().map_err(|_| ByteConversionError::FromLEBytesError)?;
    Ok(FpE::new(M127(u128::from_le_bytes(bytes))))
}

/// `limbs` big-endian limbs of [`LIMB_PAYLOAD`] bytes each, the text
/// right-aligned across them: the last limb takes the last bytes. Shared by
/// the base field (one limb) and the extension (two).
fn encode_ascii_limbs(limbs: usize, input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.len() > LIMB_PAYLOAD * limbs {
        return None;
    }
    let mut padded = vec![0u8; LIMB_BYTES * limbs];
    let mut remaining = bytes;
    for limb in (0..limbs).rev() {
        let take = remaining.len().min(LIMB_PAYLOAD);
        if take == 0 {
            break;
        }
        let end = (limb + 1) * LIMB_BYTES;
        padded[end - take..end].copy_from_slice(&remaining[remaining.len() - take..]);
        remaining = &remaining[..remaining.len() - take];
    }
    Some(padded)
}

/// Inverse of [`encode_ascii_limbs`]: the limbs' payload bytes, concatenated,
/// with the encoding's left padding stripped.
fn decode_ascii_limbs(limbs: usize, bytes: &[u8]) -> String {
    let payload: Vec<u8> =
        (0..limbs).flat_map(|limb| bytes[limb * LIMB_BYTES + 1..(limb + 1) * LIMB_BYTES].iter().copied()).collect();
    let first_nonzero = payload.iter().position(|&b| b != 0).unwrap_or(payload.len());
    payload[first_nonzero..].iter().map(|&b| b as char).collect()
}

/// Both square roots in the base field. `p ≡ 3 (mod 4)`, so a root of a
/// residue `a` is `a^((p+1)/4) = a^(2^125)`: 125 squarings, no search for a
/// non-residue. A non-residue gives a candidate whose square is `−a`, which
/// the check rejects.
fn sqrt_base(a: &FpE) -> Option<(FpE, FpE)> {
    let mut root = *a;
    for _ in 0..125 {
        root = root.square();
    }
    (root.square() == *a).then(|| (root, -root))
}

/// Both square roots in the extension, by the complex method for
/// `i^2 = −1` (Adj and Rodríguez-Henríquez, "Square root computation over
/// even extension fields", Alg. 8). For `x = x0 + x1·i` with `x^2 = a`:
/// `x0^2 − x1^2 = a0`, `2·x0·x1 = a1`, and `x0^2 + x1^2 = √(a0^2 + a1^2)`
/// (the norm's root), so `x0^2 = (a0 ± √norm) / 2`.
fn sqrt_ext(a: &Fp2E) -> Option<(Fp2E, Fp2E)> {
    let [a0, a1] = *a.value();
    let half = FpE::from(2u64).inv().expect("2 is invertible");
    let root = if a1 == FpE::zero() {
        // A base element: a root of `a0`, or `i` times a root of `−a0`.
        match sqrt_base(&a0) {
            Some((r, _)) => Fp2E::new([r, FpE::zero()]),
            None => Fp2E::new([FpE::zero(), sqrt_base(&-a0)?.0]),
        }
    } else {
        let (norm_root, _) = sqrt_base(&(a0.square() + a1.square()))?;
        let (x0, _) = sqrt_base(&((a0 + norm_root) * half)).or_else(|| sqrt_base(&((a0 - norm_root) * half)))?;
        let x1 = a1 * (x0.double()).inv().ok()?;
        Fp2E::new([x0, x1])
    };
    (root.square() == *a).then(|| (root, -root))
}

/// [`ProtocolField`] for the base field, `p = 2^127 − 1`.
impl ProtocolField for Mersenne127Field {
    /// One 16-byte limb.
    const SER_BYTES: usize = LIMB_BYTES;

    const MAX_INPUT_PAYLOAD: usize = LIMB_PAYLOAD;

    /// 127 bits is too narrow for the cryptographic DZK challenges, so they
    /// lift into the 254-bit extension.
    type Ext = Mersenne127Degree2ExtensionField;

    /// Two base elements are the two coefficients of one Fp2 element.
    const CONV_RATIO: usize = 2;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        debug_assert!(chunk.len() <= Self::CONV_RATIO, "chunk wider than one Ext element");
        let coeff = |k: usize| chunk.get(k).copied().unwrap_or_else(FpE::zero);
        Fp2E::new([coeff(0), coeff(1)])
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        Fp2E::new(<Self as IsSubFieldOf<Mersenne127Degree2ExtensionField>>::embed(*elem.value()))
    }

    const MERSENNE_BITS: Option<usize> = Some(<Self as MersennePrimeField>::BITS);

    fn mersenne_canonical(elem: &FieldElement<Self>) -> Option<u128> {
        Some(<Self as MersennePrimeField>::to_canonical_u128(elem))
    }

    /// 127 bits is enough for the statistical checks: verification runs in
    /// the base field, at no extension cost.
    type StatisticalExt = Self;
    const STATISTICAL_DEGREE: usize = 1;

    fn statistical_coeff(elem: &FieldElement<Self>, _k: usize) -> FieldElement<Self> {
        *elem
    }

    fn from_statistical_coeffs(coeffs: &[FieldElement<Self>]) -> FieldElement<Self> {
        coeffs.first().copied().unwrap_or_else(FieldElement::zero)
    }

    fn rand() -> FieldElement<Self> {
        from_two_words(random::<u64>(), random::<u64>())
    }

    /// Two draws per element, low word first.
    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        let lo = rng.next_u64();
        from_two_words(lo, rng.next_u64())
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        sqrt_base(elem)
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().0.to_be_bytes().to_vec()
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().0.to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        limb_from_bytes_be(bytes)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        limb_from_bytes_le(bytes)
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_limbs(1, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_limbs(1, &<Self as ProtocolField>::to_bytes_be(elem))
    }
}

/// [`ProtocolField`] for the degree-2 extension: 254 bits, its own `Ext`.
impl ProtocolField for Mersenne127Degree2ExtensionField {
    /// Two 16-byte limbs, real part first.
    const SER_BYTES: usize = 2 * LIMB_BYTES;

    const MAX_INPUT_PAYLOAD: usize = 2 * LIMB_PAYLOAD;

    type Ext = Self;
    const CONV_RATIO: usize = 1;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        chunk.first().copied().unwrap_or_else(FieldElement::zero)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        *elem
    }

    type StatisticalExt = Self;
    const STATISTICAL_DEGREE: usize = 1;

    fn statistical_coeff(elem: &FieldElement<Self>, _k: usize) -> FieldElement<Self> {
        *elem
    }

    fn from_statistical_coeffs(coeffs: &[FieldElement<Self>]) -> FieldElement<Self> {
        coeffs.first().copied().unwrap_or_else(FieldElement::zero)
    }

    fn rand() -> FieldElement<Self> {
        Fp2E::new([Mersenne127Field::rand(), Mersenne127Field::rand()])
    }

    /// Four draws per element, real part first.
    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        let real = Mersenne127Field::from_rng(rng);
        Fp2E::new([real, Mersenne127Field::from_rng(rng)])
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        sqrt_ext(elem)
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().iter().flat_map(|c| c.value().0.to_be_bytes()).collect()
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().iter().flat_map(|c| c.value().0.to_le_bytes()).collect()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        if bytes.len() != Self::SER_BYTES {
            return Err(ByteConversionError::FromBEBytesError);
        }
        Ok(Fp2E::new([limb_from_bytes_be(&bytes[..LIMB_BYTES])?, limb_from_bytes_be(&bytes[LIMB_BYTES..])?]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        if bytes.len() != Self::SER_BYTES {
            return Err(ByteConversionError::FromLEBytesError);
        }
        Ok(Fp2E::new([limb_from_bytes_le(&bytes[..LIMB_BYTES])?, limb_from_bytes_le(&bytes[LIMB_BYTES..])?]))
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_limbs(2, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_limbs(2, &<Self as ProtocolField>::to_bytes_be(elem))
    }
}

impl MersennePrimeField for Mersenne127Field {
    const BITS: usize = 127;

    /// Every operation leaves the word fully reduced, so it is the integer.
    fn to_canonical_u128(elem: &FieldElement<Self>) -> u128 {
        elem.value().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    type F = Mersenne127Field;
    type E = Mersenne127Degree2ExtensionField;
    const P: u128 = super::super::field::MERSENNE_127_PRIME_FIELD_ORDER;

    fn fe(x: u128) -> FpE {
        FpE::new(M127(x))
    }

    fn canon(e: &FpE) -> u128 {
        e.value().0
    }

    /// `(a + b) mod p` without the field code.
    fn add_ref(a: u128, b: u128) -> u128 {
        let s = a + b; // a, b < 2^127
        if s >= P { s - P } else { s }
    }

    /// `(a · b) mod p` by double-and-add, independent of the limb product
    /// the field uses.
    fn mul_ref(a: u128, b: u128) -> u128 {
        let mut acc = 0u128;
        for i in (0..127).rev() {
            acc = add_ref(acc, acc);
            if (b >> i) & 1 == 1 {
                acc = add_ref(acc, a);
            }
        }
        acc
    }

    /// xorshift on two words: deterministic, reproducible failures.
    fn samples(count: usize, seed: u64) -> Vec<u128> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..count).map(|_| (((next() as u128) << 64) | next() as u128) % P).collect()
    }

    /// Values where a fold or a borrow happens.
    fn edges() -> Vec<u128> {
        vec![0, 1, 2, P - 1, P - 2, P / 2, P / 2 + 1, 1 << 64, (1 << 64) - 1, 1 << 126, (1 << 126) - 1]
    }

    #[test]
    fn arithmetic_agrees_with_the_reference() {
        let values: Vec<u128> = edges().into_iter().chain(samples(200, 0x5eed)).collect();
        for &a in &values {
            for &b in values.iter().step_by(7) {
                assert_eq!(canon(&(fe(a) + fe(b))), add_ref(a, b), "{a} + {b}");
                assert_eq!(canon(&(fe(a) - fe(b))), add_ref(a, P - b) % P, "{a} - {b}");
                assert_eq!(canon(&(fe(a) * fe(b))), mul_ref(a, b), "{a} * {b}");
            }
            assert_eq!(canon(&-fe(a)), (P - a) % P, "-{a}");
            assert_eq!(canon(&fe(a).square()), mul_ref(a, a), "{a}^2");
        }
    }

    #[test]
    fn reduction_edges() {
        assert_eq!(canon(&(fe(P - 1) + fe(1))), 0, "(p-1)+1 must read as 0");
        assert_eq!(canon(&(fe(P - 1) * fe(P - 1))), 1, "(-1)^2 = 1");
        assert_eq!(canon(&FpE::new(M127(P))), 0, "p reduces to 0");
        assert_eq!(canon(&FpE::new(M127(u128::MAX))), 1, "2^128 - 1 = 2p + 1");
        assert_eq!(canon(&-fe(0)), 0);
    }

    #[test]
    fn inverse() {
        for a in edges().into_iter().chain(samples(50, 0x1a7)).filter(|&a| a != 0) {
            assert_eq!(fe(a) * fe(a).inv().unwrap(), FpE::one(), "a={a}");
        }
        assert!(fe(0).inv().is_err());
    }

    #[test]
    fn base_sqrt() {
        for r in samples(50, 0x5a7) {
            let (a, b) = F::sqrt(&fe(r).square()).expect("a square has a root");
            assert!(canon(&a) == r || canon(&b) == r);
        }
        assert_eq!(F::sqrt(&fe(0)), Some((fe(0), fe(0))));
        // p ≡ 3 (mod 4): −1 is a non-residue.
        assert!(F::sqrt(&-FpE::one()).is_none());
    }

    #[test]
    fn extension_arithmetic() {
        let i = Fp2E::new([fe(0), fe(1)]);
        assert_eq!(i.square(), -Fp2E::one(), "i^2 = -1");
        for _ in 0..50 {
            let (a, b) = (E::rand(), E::rand());
            let [a0, a1] = *a.value();
            let [b0, b1] = *b.value();
            assert_eq!(a * b, Fp2E::new([a0 * b0 - a1 * b1, a0 * b1 + a1 * b0]), "schoolbook product");
            assert_eq!(a.square(), a * a);
            if a != Fp2E::zero() {
                assert_eq!(a * a.inv().unwrap(), Fp2E::one());
            }
        }
    }

    #[test]
    fn extension_sqrt() {
        for _ in 0..50 {
            let root = E::rand();
            let square = root * root;
            let (a, b) = E::sqrt(&square).expect("a square has a root");
            assert!(a == root || b == root);
        }
        // Every base element is a square in the extension, including −1.
        for x in [fe(0), fe(5), -fe(1), -fe(5)] {
            let a = Fp2E::new([x, fe(0)]);
            let (r, _) = E::sqrt(&a).expect("a base element is a square in Fp2");
            assert_eq!(r * r, a);
        }
    }

    #[test]
    fn bytes_round_trip() {
        let e = F::rand();
        assert_eq!(F::to_bytes_be(&e).len(), F::SER_BYTES);
        assert_eq!(F::from_bytes_be(&F::to_bytes_be(&e)).unwrap(), e);
        assert_eq!(F::from_bytes_le(&F::to_bytes_le(&e)).unwrap(), e);
        assert!(F::from_bytes_be(&[0u8; 15]).is_err());

        let x = E::rand();
        assert_eq!(E::to_bytes_be(&x).len(), E::SER_BYTES);
        assert_eq!(E::from_bytes_be(&E::to_bytes_be(&x)).unwrap(), x);
        assert_eq!(E::from_bytes_le(&E::to_bytes_le(&x)).unwrap(), x);
    }

    #[test]
    fn ascii_round_trip() {
        for line in ["", "a", "hello world", "ZZZZZZZZZZZZZZZ"] {
            assert_eq!(F::decode_ascii(&F::encode_ascii(line).unwrap()), line);
        }
        for line in ["", "a", "mountain grape desert sky", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"] {
            assert_eq!(E::decode_ascii(&E::encode_ascii(line).unwrap()), line);
        }
        assert!(F::encode_ascii(&"x".repeat(F::MAX_INPUT_PAYLOAD + 1)).is_none());
        assert!(E::encode_ascii(&"x".repeat(E::MAX_INPUT_PAYLOAD + 1)).is_none());
    }

    #[test]
    fn from_rng_is_deterministic_in_the_seed() {
        let draw = || {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            (0..4).map(|_| (F::from_rng(&mut rng), E::from_rng(&mut rng))).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn lift_is_injective_and_embed_is_a_homomorphism() {
        let base = [fe(3), fe(4)];
        let lifted = F::lift(&base);
        for k in 0..2 {
            let mut perturbed = base;
            perturbed[k] += FpE::one();
            assert_ne!(F::lift(&perturbed), lifted, "coefficient {k} did not move the packed value");
        }
        let (x, y) = (F::rand(), F::rand());
        assert_eq!(F::embed_ext(&(x + y)), F::embed_ext(&x) + F::embed_ext(&y));
        assert_eq!(F::embed_ext(&(x * y)), F::embed_ext(&x) * F::embed_ext(&y));
        assert_eq!(F::embed_ext(&fe(7)), Fp2E::from(7u64));
    }

    /// Packing evaluations coefficient-wise keeps them evaluations of a
    /// polynomial of the same degree over `Ext`, which the DZK relies on.
    #[test]
    fn lift_preserves_polynomial_degree() {
        use lambdaworks_math::polynomial::Polynomial;
        let degree = 3;
        let polys: Vec<Polynomial<FpE>> =
            (0..2).map(|_| Polynomial::new(&(0..=degree).map(|_| F::rand()).collect::<Vec<_>>())).collect();
        let points: Vec<u64> = (1..=8).collect();
        let packed: Vec<Fp2E> = points
            .iter()
            .map(|&pt| F::lift(&polys.iter().map(|p| p.evaluate(&FpE::from(pt))).collect::<Vec<_>>()))
            .collect();
        let eval_pts: Vec<Fp2E> = points.iter().map(|&pt| Fp2E::from(pt)).collect();
        let interpolated = Polynomial::interpolate(&eval_pts[..=degree], &packed[..=degree]).unwrap();
        assert_eq!(interpolated.degree(), degree);
        for i in (degree + 1)..points.len() {
            assert_eq!(interpolated.evaluate(&eval_pts[i]), packed[i], "point {i} left the polynomial");
        }
    }
}
