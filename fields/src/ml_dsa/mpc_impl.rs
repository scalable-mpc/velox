//! [`ProtocolField`] for the ML-DSA field and its degree-11 extension.
//!
//! The base field has 23 bits, so both soundness levels need an extension.
//! The verification checks run in the degree-3 extension (69 bits). The ACSS
//! DZK challenges live in the degree-11 extension (253 bits), and 11 base
//! elements pack into one of its elements.

use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion as LambdaByteConversion,
    unsigned_integer::{element::U256, traits::IsUnsignedInteger},
};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::protocol_field::ProtocolField;

use super::{
    extension::{ExtCoeffs, MlDsaDegree11ExtensionField, MlDsaDegree3ExtensionField},
    field::{pow_mod, MlDsaField, ML_DSA_PRIME_FIELD_ORDER},
};

type FqE = FieldElement<MlDsaField>;
type Fq11E = FieldElement<MlDsaDegree11ExtensionField>;

const Q: u32 = ML_DSA_PRIME_FIELD_ORDER;

/// One serialized base element: the canonical `u32`.
const BASE_BYTES: usize = 4;

/// ASCII bytes one base element carries: `2^16 < q`, so two bytes always
/// decode back unchanged, and three bytes may not.
const BASE_PAYLOAD: usize = 2;

/// One serialized extension element: the packed integer `Σ c_i·q^i < 2^253`.
const EXT_BYTES: usize = 32;

/// `2^248 < q^11`, so 31 bytes always decode back unchanged.
const EXT_PAYLOAD: usize = 31;

/// `q − 1 = 2^13 · 1023`. The extension's group order `q^11 − 1` has the
/// same 2-adic part, as `(q^11 − 1)/(q − 1) = 1 + q + … + q^10` is a sum of
/// 11 odd terms.
const TWO_ADICITY: usize = 13;

/// `5^1023`, a primitive `2^13`-th root of unity: 5 is the least quadratic
/// non-residue mod q. It generates the subgroup of order 2^13 of `F_q^*`
/// and of the extension's group, as both have 2-adic part 2^13.
const ROOT_OF_UNITY_2_13: u32 = pow_mod(5, 1023);

/// Both square roots of `a` by Tonelli–Shanks, in a field whose
/// multiplicative group has order `2^13 · odd_part` with `odd_part` odd, or
/// `None` for a non-residue. `root_of_unity` must generate the subgroup of
/// order 2^13.
fn sqrt_tonelli_shanks<E: IsField, T: IsUnsignedInteger>(
    a: &FieldElement<E>,
    odd_part: T,
    root_of_unity: FieldElement<E>,
) -> Option<(FieldElement<E>, FieldElement<E>)> {
    if *a == FieldElement::zero() {
        return Some((FieldElement::zero(), FieldElement::zero()));
    }
    // Invariant: x^2 = a·t, with t in the subgroup of order 2^order.
    let mut x = a.pow((odd_part >> 1) + T::from(1));
    let mut t = a.pow(odd_part);
    let (mut c, mut order) = (root_of_unity, TWO_ADICITY);
    while t != FieldElement::one() {
        // The least i with t^(2^i) = 1. For a non-residue, t has the full
        // order 2^13 at the start, and the search reaches `order`.
        let (mut i, mut power) = (0, t.clone());
        while power != FieldElement::one() {
            power = power.square();
            i += 1;
            if i == order {
                return None;
            }
        }
        let b = (0..order - i - 1).fold(c, |acc, _| acc.square());
        c = b.square();
        x = &x * &b;
        t = &t * &c;
        order = i;
    }
    Some((x.clone(), -x))
}

/// A uniform element of `F_q`: the top 23 bits of a draw, drawn again while
/// they are at least `q` (about once in a thousand). Exact, where reducing a
/// wider draw would bias it: shares and masks over this field must be
/// perfectly uniform.
fn uniform_element(mut draw: impl FnMut() -> u32) -> FqE {
    loop {
        let x = draw() >> 9;
        if x < Q {
            return FqE::from_raw(x);
        }
    }
}

/// `input` right-aligned in `width` big-endian bytes. At most `payload`
/// bytes fit, so the leading `width − payload` bytes stay zero, the integer
/// stays below the field's order and the round trip through the field is
/// exact.
fn encode_ascii_be(width: usize, payload: usize, input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.len() > payload {
        return None;
    }
    let mut padded = vec![0u8; width];
    padded[width - bytes.len()..].copy_from_slice(bytes);
    Some(padded)
}

/// Inverse of [`encode_ascii_be`]: the bytes with the left padding stripped.
fn decode_ascii_be(bytes: &[u8]) -> String {
    let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    bytes[first_nonzero..].iter().map(|&b| b as char).collect()
}

/// [`ProtocolField`] for the base field, `q = 8380417`.
impl ProtocolField for MlDsaField {
    /// One 4-byte limb.
    const SER_BYTES: usize = BASE_BYTES;

    const MAX_INPUT_PAYLOAD: usize = BASE_PAYLOAD;

    /// 23 bits is far too narrow for the cryptographic DZK challenges, so
    /// they lift into the 253-bit extension.
    type Ext = MlDsaDegree11ExtensionField;

    /// Eleven base elements are the eleven coefficients of one `Ext` element.
    const CONV_RATIO: usize = 11;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        debug_assert!(chunk.len() <= Self::CONV_RATIO, "chunk wider than one Ext element");
        Fq11E::new(ExtCoeffs(std::array::from_fn(|k| chunk.get(k).map_or(0, |c| *c.value()))))
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        Fq11E::from(*elem.value() as u64)
    }

    /// 23 bits is too narrow for the statistical checks; they run in the
    /// degree-3 extension, 69 bits.
    type StatisticalExt = MlDsaDegree3ExtensionField;
    const STATISTICAL_DEGREE: usize = 3;

    fn statistical_coeff(elem: &FieldElement<MlDsaDegree3ExtensionField>, k: usize) -> FieldElement<Self> {
        FqE::from_raw(elem.value().0[k])
    }

    fn from_statistical_coeffs(coeffs: &[FieldElement<Self>]) -> FieldElement<MlDsaDegree3ExtensionField> {
        debug_assert!(coeffs.len() <= 3, "more coefficients than the extension's degree");
        FieldElement::new(ExtCoeffs(std::array::from_fn(|k| coeffs.get(k).map_or(0, |c| *c.value()))))
    }

    fn rand() -> FieldElement<Self> {
        uniform_element(random::<u32>)
    }

    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        uniform_element(|| rng.next_u32())
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        sqrt_tonelli_shanks(elem, (Q as u64 - 1) >> TWO_ADICITY, FqE::from_raw(ROOT_OF_UNITY_2_13))
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().to_be_bytes().to_vec()
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        let bytes: [u8; BASE_BYTES] = bytes.try_into().map_err(|_| ByteConversionError::FromBEBytesError)?;
        Ok(FqE::new(u32::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        let bytes: [u8; BASE_BYTES] = bytes.try_into().map_err(|_| ByteConversionError::FromLEBytesError)?;
        Ok(FqE::new(u32::from_le_bytes(bytes)))
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_be(BASE_BYTES, BASE_PAYLOAD, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_be(&<Self as ProtocolField>::to_bytes_be(elem))
    }
}

/// [`ProtocolField`] for the degree-11 extension: 253 bits, its own `Ext`.
impl ProtocolField for MlDsaDegree11ExtensionField {
    /// The packed integer, which the ACSS also reads a 32-byte hash as.
    const SER_BYTES: usize = EXT_BYTES;

    const MAX_INPUT_PAYLOAD: usize = EXT_PAYLOAD;

    type Ext = Self;
    const CONV_RATIO: usize = 1;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        chunk.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        elem.clone()
    }

    type StatisticalExt = Self;
    const STATISTICAL_DEGREE: usize = 1;

    fn statistical_coeff(elem: &FieldElement<Self>, _k: usize) -> FieldElement<Self> {
        elem.clone()
    }

    fn from_statistical_coeffs(coeffs: &[FieldElement<Self>]) -> FieldElement<Self> {
        coeffs.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn rand() -> FieldElement<Self> {
        Fq11E::new(ExtCoeffs(std::array::from_fn(|_| *MlDsaField::rand().value())))
    }

    /// Eleven base draws, constant term first.
    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        Fq11E::new(ExtCoeffs(std::array::from_fn(|_| *MlDsaField::from_rng(rng).value())))
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        // q^11 − 1 = Σ (q − 1)·q^i, which is what the coefficients q − 1 pack to.
        let group_order = U256::from_bytes_be(&ExtCoeffs([Q - 1; 11]).to_bytes_be()).expect("32 bytes");
        sqrt_tonelli_shanks(elem, group_order >> TWO_ADICITY, Fq11E::from(ROOT_OF_UNITY_2_13 as u64))
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().to_bytes_be()
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        elem.value().to_bytes_le()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        ExtCoeffs::from_bytes_be(bytes).map(Fq11E::new)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        ExtCoeffs::from_bytes_le(bytes).map(Fq11E::new)
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_be(EXT_BYTES, EXT_PAYLOAD, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_be(&<Self as ProtocolField>::to_bytes_be(elem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_dsa::extension::MlDsaExtensionField;
    use rand_core::SeedableRng;

    type F = MlDsaField;
    type E = MlDsaDegree11ExtensionField;
    type Fq3E = FieldElement<MlDsaDegree3ExtensionField>;

    fn fe(x: u32) -> FqE {
        FqE::new(x)
    }

    /// xorshift: deterministic, reproducible failures.
    fn samples(count: usize, seed: u64) -> Vec<u32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % Q as u64) as u32
            })
            .collect()
    }

    /// Values where a wrap or a borrow happens.
    fn edges() -> Vec<u32> {
        vec![0, 1, 2, Q - 1, Q - 2, Q / 2, Q / 2 + 1, 1 << 22, (1 << 23) - (1 << 13)]
    }

    #[test]
    fn arithmetic_agrees_with_the_reference() {
        let q = Q as u64;
        let values: Vec<u32> = edges().into_iter().chain(samples(300, 0x5eed)).collect();
        for &a in &values {
            for &b in &values {
                let (a64, b64) = (a as u64, b as u64);
                assert_eq!(*(fe(a) + fe(b)).value() as u64, (a64 + b64) % q, "{a} + {b}");
                assert_eq!(*(fe(a) - fe(b)).value() as u64, (a64 + q - b64) % q, "{a} - {b}");
                assert_eq!(*(fe(a) * fe(b)).value() as u64, a64 * b64 % q, "{a} * {b}");
            }
            assert_eq!(*(-fe(a)).value() as u64, (q - a as u64) % q, "-{a}");
        }
        assert_eq!(fe(Q), fe(0), "q reduces to 0");
        assert_eq!(*FqE::from(u64::MAX).value() as u64, u64::MAX % q);
    }

    #[test]
    fn inverse() {
        for a in (1..Q).step_by(997).chain([Q - 1]) {
            assert_eq!(fe(a) * fe(a).inv().unwrap(), FqE::one(), "a={a}");
        }
        assert!(fe(0).inv().is_err());
    }

    #[test]
    fn base_sqrt() {
        // Euler's criterion decides every sampled element; the root must agree.
        for a in (0..Q).step_by(4099).chain(edges()) {
            let is_square = a == 0 || pow_mod(a, (Q as u64 - 1) / 2) == 1;
            match F::sqrt(&fe(a)) {
                Some((r, s)) => {
                    assert!(is_square, "{a} is not a square");
                    assert_eq!(r * r, fe(a));
                    assert_eq!(s, -r);
                }
                None => assert!(!is_square, "{a} is a square"),
            }
        }
        assert_eq!(pow_mod(ROOT_OF_UNITY_2_13, 1 << 12), Q - 1, "the root has order exactly 2^13");
    }

    /// x^D − 2 is irreducible: x^(q^D) = x in F_q[x]/(x^D − 2), and for
    /// prime D the only other condition (Rabin) is that it has no root,
    /// which is that 2 is not a D-th power.
    fn modulus_is_irreducible<const D: usize>() {
        let mut x = [0; D];
        x[1] = 1;
        let x = FieldElement::<MlDsaExtensionField<D>>::new(ExtCoeffs(x));
        assert_eq!(x.pow(D as u64), FieldElement::from(2u64), "x^{D} = 2");
        let frobenius_d = (0..D).fold(x.clone(), |acc, _| acc.pow(Q as u64));
        assert_eq!(frobenius_d, x, "x^(q^{D}) = x");
        assert_ne!(pow_mod(2, ((Q - 1) / D as u32) as u64), 1, "2 is a {D}-th power");
    }

    #[test]
    fn moduli_are_irreducible() {
        modulus_is_irreducible::<3>();
        modulus_is_irreducible::<11>();
    }

    /// Products against the schoolbook product reduced afterwards, and the
    /// inverse, for both degrees.
    fn extension_arithmetic<const D: usize>(seed: u64) {
        type Elem<const D: usize> = FieldElement<MlDsaExtensionField<D>>;
        let mut values = samples(4 * D * 20, seed).into_iter();
        let mut next = || Elem::<D>::new(ExtCoeffs(std::array::from_fn(|_| values.next().unwrap())));
        for _ in 0..20 {
            let (a, b, c) = (next(), next(), next());
            let mut wide = vec![0u64; 2 * D - 1];
            for i in 0..D {
                for j in 0..D {
                    wide[i + j] = (wide[i + j] + a.value().0[i] as u64 * b.value().0[j] as u64) % Q as u64;
                }
            }
            for k in (D..2 * D - 1).rev() {
                wide[k - D] = (wide[k - D] + 2 * wide[k]) % Q as u64;
            }
            let expected: [u32; D] = std::array::from_fn(|k| wide[k] as u32);
            assert_eq!((&a * &b).value().0, expected, "schoolbook product, D = {D}");
            assert_eq!(&(&a * &b) * &c, &a * &(&b * &c), "associativity, D = {D}");
            assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c), "distributivity, D = {D}");
            assert_eq!(&a * &a.inv().unwrap(), Elem::<D>::one(), "inverse, D = {D}");
        }
        assert!(Elem::<D>::zero().inv().is_err());
    }

    #[test]
    fn extension_arithmetic_both_degrees() {
        extension_arithmetic::<3>(0x3);
        extension_arithmetic::<11>(0xb);
    }

    #[test]
    fn extension_sqrt() {
        for _ in 0..20 {
            let root = E::rand();
            let (a, b) = E::sqrt(&(&root * &root)).expect("a square has a root");
            assert!(a == root || b == root);
        }
        // The degree is odd, so a base non-residue stays a non-residue.
        assert!(E::sqrt(&Fq11E::from(5u64)).is_none());
        assert_eq!(E::sqrt(&Fq11E::zero()), Some((Fq11E::zero(), Fq11E::zero())));
    }

    #[test]
    fn bytes_round_trip() {
        for _ in 0..20 {
            let e = F::rand();
            assert_eq!(F::to_bytes_be(&e).len(), F::SER_BYTES);
            assert_eq!(F::from_bytes_be(&F::to_bytes_be(&e)).unwrap(), e);
            assert_eq!(F::from_bytes_le(&F::to_bytes_le(&e)).unwrap(), e);

            let x = E::rand();
            assert_eq!(E::to_bytes_be(&x).len(), E::SER_BYTES);
            assert_eq!(E::from_bytes_be(&E::to_bytes_be(&x)).unwrap(), x);
            assert_eq!(E::from_bytes_le(&E::to_bytes_le(&x)).unwrap(), x);
        }
        assert!(F::from_bytes_be(&[0u8; 3]).is_err());
        assert!(E::from_bytes_be(&[0u8; 31]).is_err());
        // A hash is any 32 bytes: it decodes, reduced mod q^11.
        let top = E::from_bytes_be(&[0xff; 32]).unwrap();
        assert!(top.value().0.iter().all(|&c| c < Q));
        // The packing is the integer Σ c_i·q^i: x encodes q.
        let x = Fq11E::new(ExtCoeffs(std::array::from_fn(|k| (k == 1) as u32)));
        assert_eq!(E::to_bytes_be(&x)[28..], Q.to_be_bytes());
    }

    #[test]
    fn ascii_round_trip() {
        for line in ["", "a", "hi", "\x7f\x7f"] {
            assert_eq!(F::decode_ascii(&F::encode_ascii(line).unwrap()), line);
        }
        for line in ["", "a", "mountain grape desert sky", &"~".repeat(31)] {
            assert_eq!(E::decode_ascii(&E::encode_ascii(line).unwrap()), line);
        }
        assert!(F::encode_ascii(&"x".repeat(F::MAX_INPUT_PAYLOAD + 1)).is_none());
        assert!(E::encode_ascii(&"x".repeat(E::MAX_INPUT_PAYLOAD + 1)).is_none());
    }

    #[test]
    fn sampling_is_in_range_and_seed_determined() {
        let draw = || {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            (0..64).map(|_| (F::from_rng(&mut rng), E::from_rng(&mut rng))).collect::<Vec<_>>()
        };
        let (first, second) = (draw(), draw());
        assert_eq!(first, second);
        for (f, e) in first {
            assert!(*f.value() < Q);
            assert!(e.value().0.iter().all(|&c| c < Q));
        }
    }

    #[test]
    fn lift_is_injective_and_embed_is_a_homomorphism() {
        let base: Vec<FqE> = (1..=11).map(|k| fe(k)).collect();
        let lifted = F::lift(&base);
        for k in 0..11 {
            let mut perturbed = base.clone();
            perturbed[k] += FqE::one();
            assert_ne!(F::lift(&perturbed), lifted, "coefficient {k} did not move the packed value");
        }
        let (x, y) = (F::rand(), F::rand());
        assert_eq!(F::embed_ext(&(x + y)), F::embed_ext(&x) + F::embed_ext(&y));
        assert_eq!(F::embed_ext(&(x * y)), F::embed_ext(&x) * F::embed_ext(&y));
    }

    #[test]
    fn statistical_coefficients_round_trip() {
        let coeffs = [F::rand(), F::rand(), F::rand()];
        let elem = F::from_statistical_coeffs(&coeffs);
        for k in 0..3 {
            assert_eq!(F::statistical_coeff(&elem, k), coeffs[k]);
        }
        let (x, y) = (F::rand(), F::rand());
        let embed = |v: FqE| F::from_statistical_coeffs(&[v]);
        assert_eq!(embed(x) * embed(y), embed(x * y));
        assert_eq!(embed(x) + embed(y), embed(x + y));
        let _: Fq3E = embed(x);
    }

    /// Packing evaluations coefficient-wise keeps them evaluations of a
    /// polynomial of the same degree over `Ext`, which the DZK relies on.
    #[test]
    fn lift_preserves_polynomial_degree() {
        use lambdaworks_math::polynomial::Polynomial;
        let degree = 3;
        let polys: Vec<Polynomial<FqE>> =
            (0..11).map(|_| Polynomial::new(&(0..=degree).map(|_| F::rand()).collect::<Vec<_>>())).collect();
        let points: Vec<u64> = (1..=8).collect();
        let packed: Vec<Fq11E> = points
            .iter()
            .map(|&pt| F::lift(&polys.iter().map(|p| p.evaluate(&FqE::from(pt))).collect::<Vec<_>>()))
            .collect();
        let eval_pts: Vec<Fq11E> = points.iter().map(|&pt| Fq11E::from(pt)).collect();
        let interpolated = Polynomial::interpolate(&eval_pts[..=degree], &packed[..=degree]).unwrap();
        assert_eq!(interpolated.degree(), degree);
        for i in (degree + 1)..points.len() {
            assert_eq!(interpolated.evaluate(&eval_pts[i]), packed[i], "point {i} left the polynomial");
        }
    }
}
