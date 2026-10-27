//! ML-DSA (FIPS 204) in the clear: what every party computes alone, on
//! public values or on its own shares by linear maps.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128, Shake256,
};
use velox::{
    fields::{
        ml_dsa::{intt, ntt, ML_DSA_PRIME_FIELD_ORDER, RING_DEGREE},
        MlDsaField,
    },
    FieldElement,
};

/// An element of `F_q`.
pub type Fq = FieldElement<MlDsaField>;

/// An element of the ring `R_q = F_q[X]/(X^256 + 1)`: its coefficients, or
/// its values in the NTT domain.
pub type Poly = [Fq; RING_DEGREE];

fn zero_poly() -> Poly {
    std::array::from_fn(|_| Fq::zero())
}

/// The NTT of a copy of `p`.
fn to_ntt_domain(p: &Poly) -> Poly {
    let mut p_hat = p.clone();
    ntt(&mut p_hat);
    p_hat
}

/// `q = 2^23 − 2^13 + 1`.
pub const Q: u32 = ML_DSA_PRIME_FIELD_ORDER;

/// `d`: the bits that `Power2Round` drops from `t` (FIPS 204, Table 1).
pub const DROPPED_BITS: u32 = 13;

/// One ML-DSA parameter set (FIPS 204, Table 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterSet {
    /// The FIPS 204 name, for example `"ML-DSA-44"`.
    pub name: &'static str,
    /// The rows of the matrix `A`.
    pub k: usize,
    /// The columns of the matrix `A` (`ℓ` in FIPS 204).
    pub l: usize,
    /// `η`: the coefficients of the secrets `s` and `e` are in `[−η, η]`.
    pub eta: u32,
    /// `τ`: the number of ±1 coefficients of the challenge `c`.
    pub tau: usize,
    /// `β = τ·η`: a bound on `‖c·s‖∞` and `‖c·e‖∞`.
    pub beta: u32,
    /// `γ1`: the coefficients of the mask `y` are in `(−γ1, γ1]`.
    pub gamma1: u32,
    /// `γ2`: the low-order rounding range of `Decompose`.
    pub gamma2: u32,
    /// `ω`: the maximum number of 1 coefficients in the hint `h`.
    pub omega: usize,
    /// `λ`: the collision strength in bits; the challenge hash `c̃` has
    /// `λ/4` bytes.
    pub lambda: usize,
}

/// ML-DSA-44, security category 2: the one parameter set the app signs
/// with.
pub const ML_DSA_44: ParameterSet = ParameterSet {
    name: "ML-DSA-44",
    k: 4,
    l: 4,
    eta: 2,
    tau: 39,
    beta: 78,
    gamma1: 1 << 17,
    gamma2: (Q - 1) / 88,
    omega: 80,
    lambda: 128,
};

/// `ExpandA` (FIPS 204, Algorithm 32): the public matrix `A`, `k` rows of
/// `l` ring elements, in the NTT domain, from the seed `ρ`.
pub fn expand_a(rho: &[u8; 32], set: &ParameterSet) -> Vec<Vec<Poly>> {
    (0..set.k)
        .map(|row| {
            (0..set.l)
                .map(|column| {
                    // RejNTTPoly (Algorithm 30): SHAKE128 of ρ ‖ column ‖ row,
                    // read 3 bytes at a time; keep the 23-bit values below q.
                    let mut shake = Shake128::default();
                    shake.update(rho);
                    shake.update([column as u8, row as u8]);
                    let mut reader = shake.finalize_xof();
                    let mut values = Vec::with_capacity(RING_DEGREE);
                    let mut bytes = [0u8; 3];
                    while values.len() < RING_DEGREE {
                        reader.read(&mut bytes);
                        let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2] & 0x7f, 0]);
                        if value < Q {
                            values.push(Fq::from(u64::from(value)));
                        }
                    }
                    values.try_into().expect("exactly RING_DEGREE values")
                })
                .collect()
        })
        .collect()
}

/// `A·v` for `A` in the NTT domain and `v` in the coefficient domain; the
/// result is in the coefficient domain. Linear in `v`, so a party applies it
/// to its shares of `v`.
pub fn matrix_vector_product(a_hat: &[Vec<Poly>], v: &[Poly]) -> Vec<Poly> {
    let v_hat: Vec<Poly> = v.iter().map(to_ntt_domain).collect();
    a_hat
        .iter()
        .map(|row| {
            let mut sum = zero_poly();
            for (a, v) in row.iter().zip(&v_hat) {
                for ((s, a), v) in sum.iter_mut().zip(a).zip(v) {
                    *s = &*s + a * v;
                }
            }
            intt(&mut sum);
            sum
        })
        .collect()
}

/// `r mod± α`: the representative of `r mod α` in `(−α/2, α/2]` (FIPS 204,
/// §2.3).
fn centered_mod(r: u32, alpha: u32) -> i64 {
    let r0 = i64::from(r % alpha);
    if r0 > i64::from(alpha / 2) {
        r0 - i64::from(alpha)
    } else {
        r0
    }
}

/// `Power2Round` (FIPS 204, Algorithm 35): `r = r1·2^d + r0` with
/// `r0 = r mod± 2^d`, for `r` in `[0, q)`.
pub fn power2round(r: u32) -> (u32, i64) {
    let r0 = centered_mod(r, 1 << DROPPED_BITS);
    (((i64::from(r) - r0) >> DROPPED_BITS) as u32, r0)
}

/// `Decompose` (FIPS 204, Algorithm 36): `r = r1·2γ2 + r0` with
/// `r0 = r mod± 2γ2`, for `r` in `[0, q)`; the one value with
/// `r − r0 = q − 1` gives `r1 = 0` and `r0 − 1`.
pub fn decompose(r: u32, gamma2: u32) -> (u32, i64) {
    let alpha = 2 * gamma2;
    let r0 = centered_mod(r, alpha);
    if i64::from(r) - r0 == i64::from(Q - 1) {
        (0, r0 - 1)
    } else {
        (((i64::from(r) - r0) / i64::from(alpha)) as u32, r0)
    }
}

/// `HighBits` (FIPS 204, Algorithm 37).
pub fn high_bits(r: u32, gamma2: u32) -> u32 {
    decompose(r, gamma2).0
}

/// `LowBits` (FIPS 204, Algorithm 38).
pub fn low_bits(r: u32, gamma2: u32) -> i64 {
    decompose(r, gamma2).1
}

/// `MakeHint` (FIPS 204, Algorithm 39): whether adding `z` to `r` changes
/// the high bits. Both are in `[0, q)`.
pub fn make_hint(z: u32, r: u32, gamma2: u32) -> bool {
    high_bits(r, gamma2) != high_bits((r + z) % Q, gamma2)
}

/// `UseHint` (FIPS 204, Algorithm 40): the high bits of `r + z` from `r`
/// and the hint, when `‖z‖∞ ≤ γ2`.
pub fn use_hint(hint: bool, r: u32, gamma2: u32) -> u32 {
    let m = (Q - 1) / (2 * gamma2);
    let (r1, r0) = decompose(r, gamma2);
    match (hint, r0 > 0) {
        (false, _) => r1,
        (true, true) => (r1 + 1) % m,
        (true, false) => (r1 + m - 1) % m,
    }
}

/// `SampleInBall` (FIPS 204, Algorithm 29): the challenge `c` with `τ`
/// coefficients in `{−1, 1}` and the rest 0, from the hash `c̃`.
pub fn sample_in_ball(c_tilde: &[u8], tau: usize) -> Poly {
    let mut shake = Shake256::default();
    shake.update(c_tilde);
    let mut reader = shake.finalize_xof();
    let mut sign_bytes = [0u8; 8];
    reader.read(&mut sign_bytes);
    let signs = u64::from_le_bytes(sign_bytes);
    let mut c = zero_poly();
    for (k, i) in (RING_DEGREE - tau..RING_DEGREE).enumerate() {
        let mut j = [0u8; 1];
        loop {
            reader.read(&mut j);
            if usize::from(j[0]) <= i {
                break;
            }
        }
        let j = usize::from(j[0]);
        c[i] = c[j].clone();
        c[j] = if (signs >> k) & 1 == 1 { -Fq::one() } else { Fq::one() };
    }
    c
}

/// `w1Encode` (FIPS 204, Algorithm 28): `SimpleBitPack` of each ring element
/// of `w1`, with `bitlen((q − 1)/(2γ2) − 1)` bits per coefficient (6 for
/// ML-DSA-44), least significant bit first.
pub fn w1_encode(w1: &[[u32; RING_DEGREE]], gamma2: u32) -> Vec<u8> {
    let bits = u32::BITS - ((Q - 1) / (2 * gamma2) - 1).leading_zeros();
    pack_bits(w1.iter().flatten().copied(), bits)
}

/// `bits` bits of each value, least significant bit first, into bytes
/// (FIPS 204, `SimpleBitPack` and `BitPack`).
fn pack_bits(values: impl Iterator<Item = u32>, bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let (mut buffer, mut filled) = (0u64, 0);
    for value in values {
        buffer |= u64::from(value) << filled;
        filled += bits;
        while filled >= 8 {
            bytes.push(buffer as u8);
            buffer >>= 8;
            filled -= 8;
        }
    }
    bytes
}

/// The integer in `(−(q−1)/2, (q−1)/2]` that `x` represents.
pub fn centered(x: &Fq) -> i64 {
    let v = i64::from(x.representative());
    if v > i64::from((Q - 1) / 2) {
        v - i64::from(Q)
    } else {
        v
    }
}

/// `pkEncode` (FIPS 204, Algorithm 22): `ρ`, then the 10-bit coefficients
/// of `t1`. 1,312 bytes for ML-DSA-44.
pub fn pk_encode(rho: &[u8; 32], t1: &[[u32; RING_DEGREE]]) -> Vec<u8> {
    let bits = u32::BITS - ((Q - 1) >> DROPPED_BITS).leading_zeros();
    let mut bytes = rho.to_vec();
    bytes.extend(pack_bits(t1.iter().flatten().copied(), bits));
    bytes
}

/// `sigEncode` (FIPS 204, Algorithm 26): `c̃`, then `γ1 − z` for each
/// coefficient of `z` in `bitlen(2γ1 − 1)` bits, then `HintBitPack(h)`: the
/// positions of the ones of each ring element of `h`, and the running count
/// after each, in `ω + k` bytes. 2,420 bytes for ML-DSA-44. `None` when
/// the hint has more than `ω` ones: FIPS 204 gives no signature then.
pub fn sig_encode(c_tilde: &[u8], z: &[Poly], hint: &[[bool; RING_DEGREE]], set: &ParameterSet) -> Option<Vec<u8>> {
    if hint.iter().flatten().filter(|one| **one).count() > set.omega {
        return None;
    }
    let bits = u32::BITS - (2 * set.gamma1 - 1).leading_zeros();
    let mut bytes = c_tilde.to_vec();
    bytes.extend(pack_bits(
        z.iter().flatten().map(|x| (i64::from(set.gamma1) - centered(x)) as u32),
        bits,
    ));
    let mut positions = vec![0u8; set.omega + set.k];
    let mut count = 0;
    for (i, h) in hint.iter().enumerate() {
        for (j, _) in h.iter().enumerate().filter(|(_, one)| **one) {
            positions[count] = j as u8;
            count += 1;
        }
        positions[set.omega + i] = count as u8;
    }
    bytes.extend(positions);
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_dsa::{EncodedSignature, EncodedVerifyingKey, MlDsa44, Signature, VerifyingKey};

    /// SHAKE256 of the parts, `len` bytes (FIPS 204's `H`).
    fn shake256(parts: &[&[u8]], len: usize) -> Vec<u8> {
        let mut shake = Shake256::default();
        for part in parts {
            shake.update(part);
        }
        let mut out = vec![0u8; len];
        shake.finalize_xof().read(&mut out);
        out
    }

    /// A pseudo-random integer source for the reference signer.
    struct Coins(u64);
    impl Coins {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }
        /// A ring element with coefficients in `[low, high]`.
        fn poly(&mut self, low: i64, high: i64) -> Poly {
            std::array::from_fn(|_| {
                let v = low + self.next((high - low + 1) as u64) as i64;
                Fq::from(v.rem_euclid(i64::from(Q)) as u64)
            })
        }
    }

    fn ring_product(a: &Poly, b: &Poly) -> Poly {
        matrix_vector_product(&[vec![to_ntt_domain(a)]], std::slice::from_ref(b)).remove(0)
    }

    fn add_polys(a: &Poly, b: &Poly) -> Poly {
        std::array::from_fn(|i| &a[i] + &b[i])
    }

    fn infinity_norm(p: &[Poly]) -> i64 {
        p.iter().flatten().map(|x| centered(x).abs()).max().unwrap_or(0)
    }

    /// Quorus Algorithm 1 (ML-DSA with the extra error `e_w`), in the clear,
    /// on the functions of this module: key generation and one signature of
    /// `message`. Returns `(pk, signature)`.
    fn reference_sign(message: &[u8], coins: &mut Coins) -> (Vec<u8>, Vec<u8>) {
        let set = ML_DSA_44;
        let eta = i64::from(set.eta);
        let rho: [u8; 32] = std::array::from_fn(|_| coins.next(256) as u8);
        let a_hat = expand_a(&rho, &set);
        let s: Vec<Poly> = (0..set.l).map(|_| coins.poly(-eta, eta)).collect();
        let e: Vec<Poly> = (0..set.k).map(|_| coins.poly(-eta, eta)).collect();
        let t: Vec<Poly> = matrix_vector_product(&a_hat, &s).iter().zip(&e).map(|(a, e)| add_polys(a, e)).collect();
        let (t1, t0): (Vec<[u32; RING_DEGREE]>, Vec<Poly>) = t
            .iter()
            .map(|t| {
                let parts: Vec<(u32, i64)> = t.iter().map(|x| power2round(x.representative())).collect();
                (
                    std::array::from_fn(|i| parts[i].0),
                    std::array::from_fn(|i| Fq::from(parts[i].1.rem_euclid(i64::from(Q)) as u64)),
                )
            })
            .unzip();
        let pk = pk_encode(&rho, &t1);
        let tr = shake256(&[&pk], 64);
        let mu = shake256(&[&tr, &[0, 0], message], 64);
        for _ in 0..1000 {
            let gamma1 = i64::from(set.gamma1);
            let y: Vec<Poly> = (0..set.l).map(|_| coins.poly(1 - gamma1, gamma1)).collect();
            let e_w: Vec<Poly> = (0..set.k).map(|_| coins.poly(-eta, eta)).collect();
            let w: Vec<Poly> = matrix_vector_product(&a_hat, &y).iter().zip(&e_w).map(|(a, e)| add_polys(a, e)).collect();
            let w1: Vec<[u32; RING_DEGREE]> =
                w.iter().map(|w| std::array::from_fn(|i| high_bits(w[i].representative(), set.gamma2))).collect();
            let c_tilde = shake256(&[&mu, &w1_encode(&w1, set.gamma2)], set.lambda / 4);
            let c = sample_in_ball(&c_tilde, set.tau);
            let z: Vec<Poly> = y.iter().zip(&s).map(|(y, s)| add_polys(y, &ring_product(&c, s))).collect();
            let w0_bar: Vec<Poly> = w
                .iter()
                .zip(&e)
                .map(|(w, e)| {
                    let ce = ring_product(&c, e);
                    std::array::from_fn(|i| {
                        Fq::from(low_bits(w[i].representative(), set.gamma2).rem_euclid(i64::from(Q)) as u64) - &ce[i]
                    })
                })
                .collect();
            let beta = i64::from(set.beta);
            if infinity_norm(&z) >= gamma1 - beta || infinity_norm(&w0_bar) >= i64::from(set.gamma2) - beta {
                continue;
            }
            // h = MakeHint(−c·t0, A·z − c·t + c·t0), with e_w dropped as in
            // Quorus Algorithm 1.
            let ct0: Vec<Poly> = t0.iter().map(|t0| ring_product(&c, t0)).collect();
            if infinity_norm(&ct0) >= i64::from(set.gamma2) {
                continue;
            }
            let az = matrix_vector_product(&a_hat, &z);
            let hint: Vec<[bool; RING_DEGREE]> = (0..set.k)
                .map(|row| {
                    let ct = ring_product(&c, &t[row]);
                    std::array::from_fn(|i| {
                        let r = &az[row][i] - &ct[i] + &ct0[row][i];
                        make_hint((-&ct0[row][i]).representative(), r.representative(), set.gamma2)
                    })
                })
                .collect();
            if let Some(signature) = sig_encode(&c_tilde, &z, &hint, &set) {
                return (pk, signature);
            }
        }
        panic!("no passing attempt in 1000");
    }

    /// RustCrypto `ml-dsa` accepts the reference ML-DSA-44 signature: this
    /// checks `ExpandA`, the rounding, the hints, `SampleInBall` and the
    /// encodings against an independent implementation of FIPS 204.
    #[test]
    fn rustcrypto_accepts_the_reference_signature() {
        let mut coins = Coins(0x0dd_ba11);
        let message = b"threshold ML-DSA";
        let (pk, signature) = reference_sign(message, &mut coins);
        assert_eq!((pk.len(), signature.len()), (1312, 2420));
        let pk = EncodedVerifyingKey::<MlDsa44>::try_from(&pk[..]).expect("1,312 bytes");
        let signature = EncodedSignature::<MlDsa44>::try_from(&signature[..]).expect("2,420 bytes");
        let signature = Signature::<MlDsa44>::decode(&signature).expect("a well-formed signature");
        let key = VerifyingKey::<MlDsa44>::decode(&pk);
        assert!(key.verify_with_context(message, &[], &signature));
        assert!(!key.verify_with_context(b"another message", &[], &signature));
    }

    /// The challenge has exactly `τ` nonzero coefficients, each ±1, and the
    /// same hash gives the same challenge.
    #[test]
    fn sample_in_ball_has_tau_signs() {
        let c = sample_in_ball(&[3u8; 32], ML_DSA_44.tau);
        let nonzero: Vec<&Fq> = c.iter().filter(|x| **x != Fq::zero()).collect();
        assert_eq!(nonzero.len(), ML_DSA_44.tau);
        assert!(nonzero.iter().all(|x| **x == Fq::one() || **x == -Fq::one()));
        assert_eq!(c, sample_in_ball(&[3u8; 32], ML_DSA_44.tau));
        assert_ne!(c, sample_in_ball(&[4u8; 32], ML_DSA_44.tau));
    }

    /// 6 bits per coefficient, least significant bit first: 768 bytes for
    /// `k = 4`, and the first coefficients land where FIPS 204 puts them.
    #[test]
    fn w1_encode_packs_6_bits() {
        let mut w1 = [[0u32; RING_DEGREE]; 4];
        w1[0][0] = 43;
        w1[0][1] = 1;
        let bytes = w1_encode(&w1, ML_DSA_44.gamma2);
        assert_eq!(bytes.len(), 768);
        // 43 = 0b101011 in bits 0..6, then 1 in bits 6..12.
        assert_eq!(bytes[0], 0b0110_1011);
        assert_eq!(bytes[1], 0b0000_0000);
    }

    /// Over all of `[0, q)`: `Decompose` gives `r ≡ r1·2γ2 + r0`, `r1` in
    /// `[0, 44)` and `r0` in its range, and the high bits are what the MPC
    /// computes, `((−44·r + (q − 1)/2) mod q) mod 44` (Quorus, Protocol 8).
    /// `Power2Round` gives `r = r1·2^13 + r0`.
    #[test]
    fn rounding_over_all_of_f_q() {
        let gamma2 = ML_DSA_44.gamma2;
        let (q, alpha) = (i64::from(Q), i64::from(2 * gamma2));
        for r in 0..Q {
            let (r1, r0) = decompose(r, gamma2);
            assert!(r1 < 44 && -alpha / 2 - 1 <= r0 && r0 <= alpha / 2, "r = {r}");
            assert_eq!((i64::from(r1) * alpha + r0).rem_euclid(q), i64::from(r), "r = {r}");
            let mpc = ((u64::from(Q - 44) * u64::from(r) + u64::from((Q - 1) / 2)) % u64::from(Q)) % 44;
            assert_eq!(u64::from(r1), mpc, "r = {r}");
            let (t1, t0) = power2round(r);
            assert_eq!(i64::from(t1) * (1 << DROPPED_BITS) + t0, i64::from(r), "r = {r}");
            assert!(-(1 << 12) < t0 && t0 <= 1 << 12, "r = {r}");
        }
    }

    /// `UseHint(MakeHint(z, r), r) = HighBits(r + z)` for `‖z‖∞ ≤ γ2`
    /// (FIPS 204, Lemma 2), on random `r` and `z`.
    #[test]
    fn hints_recover_the_high_bits() {
        let gamma2 = ML_DSA_44.gamma2;
        let mut coins = Coins(0x5eed);
        for _ in 0..200_000 {
            let r = coins.next(u64::from(Q)) as u32;
            let z_signed = coins.next(2 * u64::from(gamma2) + 1) as i64 - i64::from(gamma2);
            let z = z_signed.rem_euclid(i64::from(Q)) as u32;
            let hint = make_hint(z, r, gamma2);
            assert_eq!(use_hint(hint, r, gamma2), high_bits((r + z) % Q, gamma2), "r = {r}, z = {z_signed}");
        }
    }

    /// The product in `R_q` by the schoolbook rule: `X^256 = −1`.
    fn negacyclic_product(a: &Poly, b: &Poly) -> Poly {
        let mut c = zero_poly();
        for i in 0..RING_DEGREE {
            for j in 0..RING_DEGREE {
                let term = &a[i] * &b[j];
                if i + j < RING_DEGREE {
                    c[i + j] = &c[i + j] + term;
                } else {
                    c[i + j - RING_DEGREE] = &c[i + j - RING_DEGREE] - term;
                }
            }
        }
        c
    }

    fn pseudo_random_poly(seed: u64) -> Poly {
        let mut state = seed | 1;
        std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            Fq::from(state % u64::from(Q))
        })
    }

    /// `A·v` agrees with the schoolbook products, for a 2 × 2 matrix.
    #[test]
    fn matrix_vector_product_is_the_ring_product() {
        let a: Vec<Vec<Poly>> = (0..2).map(|i| (0..2).map(|j| pseudo_random_poly(10 * i + j + 1)).collect()).collect();
        let v: Vec<Poly> = (0..2).map(|j| pseudo_random_poly(100 + j)).collect();
        let a_hat: Vec<Vec<Poly>> = a.iter().map(|row| row.iter().map(to_ntt_domain).collect()).collect();
        let got = matrix_vector_product(&a_hat, &v);
        for (i, row) in a.iter().enumerate() {
            let mut want = zero_poly();
            for (a, v) in row.iter().zip(&v) {
                let product = negacyclic_product(a, v);
                for (w, p) in want.iter_mut().zip(&product) {
                    *w = &*w + p;
                }
            }
            assert_eq!(got[i], want, "row {i}");
        }
    }

    /// `ExpandA` is deterministic, has the ML-DSA-44 shape, and its entries
    /// differ. (RustCrypto `ml-dsa` checks the values in A2d.)
    #[test]
    fn expand_a_shape() {
        let a = expand_a(&[7u8; 32], &ML_DSA_44);
        assert_eq!((a.len(), a[0].len()), (4, 4));
        assert_eq!(a, expand_a(&[7u8; 32], &ML_DSA_44));
        assert_ne!(a[0][1], a[1][0], "the seed orders column, then row");
        assert_ne!(a, expand_a(&[8u8; 32], &ML_DSA_44));
    }

    /// The relations between the parameters that the threshold protocol
    /// uses: `β = τ·η`, and `(q − 1)/(2γ2) = 44` exactly, so the high bits
    /// `w1` are `(−44·w + (q − 1)/2 mod q) mod 44` (Quorus, Protocol 8).
    #[test]
    fn parameter_relations() {
        let set = ML_DSA_44;
        assert_eq!(set.beta, set.tau as u32 * set.eta);
        assert_eq!((Q - 1) / (2 * set.gamma2), 44);
        assert_eq!((Q - 1) % (2 * set.gamma2), 0);
    }
}
