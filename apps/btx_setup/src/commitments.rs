//! A party's commitments to its shares: the shares lifted into the pairing
//! groups. This is the public side of the setup, computed locally from the
//! party's own shares and nothing else.
//!
//! Notation as in the crate root: `⟨x⟩_j` is party `j`'s degree-`t` Shamir
//! share of `x`; `g₁^x`, `g₂^x`, `g_T^x` are `x` lifted into G₁, G₂, G_T of
//! BLS12-381, with `e(g₁, g₂) = g_T`.
//!
//! From `⟨τ¹⟩_j … ⟨τ^{2B}⟩_j`, party `j` computes
//!
//!   - `g₂^{⟨τ^i⟩_j}` for `i ∈ [2B] \ {B+1}`. For `i ≤ B` these are the share
//!     commitments `v_j^i` a combiner checks `j`'s partial decryptions
//!     against; for `i ≥ B+2` they exist only so that `h_i` can be
//!     interpolated.
//!   - `e(g₁^{⟨τ^{B+1}⟩_j}, g₂) = g_T^{⟨τ^{B+1}⟩_j}` for the punctured power,
//!     so that `τ^{B+1}` only ever exists in G_T. In G₂ (or G₁) it would be
//!     the one element the encryption's security says must not exist.
//!
//! Everything downstream is public and needs no party's cooperation: Shamir
//! reconstruction is linear, so it commutes with lifting, and any `t+1`
//! parties' commitments give `h_i = g₂^{τ^i} = Σ_j λ_j · g₂^{⟨τ^i⟩_j}` and
//! `ek = g_T^{τ^{B+1}} = Π_j (g_T^{⟨τ^{B+1}⟩_j})^{λ_j}` with the Lagrange
//! coefficients `λ_j` at zero for the parties' evaluation points (`j + 1`,
//! the engine's convention).

use anyhow::{bail, Result};
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    curve::BLS12381Curve, field_extension::Degree12ExtensionField, pairing::BLS12381AtePairing,
    twist::BLS12381TwistCurve,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::short_weierstrass::traits::Compress;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use lambdaworks_math::traits::ByteConversion;
use rayon::prelude::*;
use velox::fields::BLS12381ScalarField;
use velox::FieldElement;

/// The scalar field the setup ran over. Lifting is not generic: the curve's
/// points are multiplied by exactly this field.
pub type Scalar = FieldElement<BLS12381ScalarField>;
pub type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;
pub type Gt = FieldElement<Degree12ExtensionField>;

/// Party `j`'s shares lifted into the groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitments {
    /// `g₂^{⟨τ^i⟩_j}` for `i = 1..=B`: the share commitments `v_j^i`.
    pub low: Vec<G2>,
    /// `g_T^{⟨τ^{B+1}⟩_j}`: the punctured power, in G_T only.
    pub middle: Gt,
    /// `g₂^{⟨τ^i⟩_j}` for `i = B+2..=2B`.
    pub high: Vec<G2>,
}

impl Commitments {
    /// Lift `shares[i-1] = ⟨τ^i⟩_j`, `i = 1..=2B`. `2B` G₂ exponentiations and
    /// one pairing, run across rayon's pool.
    pub fn lift(batch_size: usize, shares: &[Scalar]) -> Result<Self> {
        if shares.len() != 2 * batch_size {
            bail!(
                "lifting for batch size {} needs {} shares, got {}",
                batch_size,
                2 * batch_size,
                shares.len()
            );
        }
        let g2 = BLS12381TwistCurve::generator();
        let lift = |s: &Scalar| g2.operate_with_self(s.representative());
        Ok(Self {
            low: shares[..batch_size].par_iter().map(lift).collect(),
            middle: lift_to_gt(&shares[batch_size])?,
            high: shares[batch_size + 1..].par_iter().map(lift).collect(),
        })
    }

    /// `B`, recovered from the shape.
    pub fn batch_size(&self) -> usize {
        self.low.len()
    }
}

/// `g_T^s = e(g₁^s, g₂)`.
pub fn lift_to_gt(s: &Scalar) -> Result<Gt> {
    let g1 = BLS12381Curve::generator().operate_with_self(s.representative());
    BLS12381AtePairing::compute_batch(&[(&g1, &BLS12381TwistCurve::generator())])
        .map_err(|e| anyhow::anyhow!("pairing failed: {:?}", e))
}

/// Lower-case hex of lambdaworks' 96-byte compressed G₂ encoding.
pub fn g2_hex(p: &G2) -> String {
    hex(&BLS12381Curve::compress_g2_point(p))
}

/// Lower-case hex of an Fp12 element as its six Fp2 limbs, each in
/// lambdaworks' big-endian encoding: `Fp12 = Fp6[w]/(w² − v)`,
/// `Fp6 = Fp2[v]/(v³ − ξ)`, in the order `c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2`
/// — 576 bytes.
pub fn gt_hex(e: &Gt) -> String {
    let mut bytes = Vec::with_capacity(576);
    for fp6 in e.value() {
        for fp2 in fp6.value() {
            bytes.extend(fp2.to_bytes_be());
        }
    }
    hex(&bytes)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use velox::ProtocolField;

    #[test]
    fn lifted_shares_are_the_shares_in_the_exponent() {
        let batch_size = 3;
        let shares: Vec<Scalar> = (0..2 * batch_size).map(|_| BLS12381ScalarField::rand()).collect();
        let c = Commitments::lift(batch_size, &shares).unwrap();
        let g2 = BLS12381TwistCurve::generator();
        assert_eq!(c.batch_size(), batch_size);
        assert_eq!(c.low.len(), batch_size);
        assert_eq!(c.high.len(), batch_size - 1);
        for i in 0..batch_size {
            assert_eq!(c.low[i], g2.operate_with_self(shares[i].representative()), "v^{}", i + 1);
        }
        for i in 0..batch_size - 1 {
            assert_eq!(
                c.high[i],
                g2.operate_with_self(shares[batch_size + 1 + i].representative()),
                "power {}",
                batch_size + 2 + i
            );
        }
        assert_eq!(c.middle, lift_to_gt(&shares[batch_size]).unwrap());
    }

    #[test]
    fn lifting_commutes_with_reconstruction() {
        // Two parties' shares of x on a degree-1 line at points 1 and 2:
        // λ = (2, -1). The lifted shares combine to g₂^x and g_T^x.
        let x = BLS12381ScalarField::rand();
        let slope = BLS12381ScalarField::rand();
        let s1 = &x + &slope;
        let s2 = &x + &slope + &slope;
        let two = Scalar::from(2u64);
        let minus_one = Scalar::zero() - Scalar::one();

        let g2 = BLS12381TwistCurve::generator();
        let combined = g2
            .operate_with_self(s1.representative())
            .operate_with_self(two.representative())
            .operate_with(&g2.operate_with_self(s2.representative()).operate_with_self(minus_one.representative()));
        assert_eq!(combined, g2.operate_with_self(x.representative()));

        let combined_gt =
            lift_to_gt(&s1).unwrap().pow(two.representative()) * lift_to_gt(&s2).unwrap().pow(minus_one.representative());
        assert_eq!(combined_gt, lift_to_gt(&x).unwrap());
    }

    #[test]
    fn wrong_share_count_is_an_error() {
        let shares: Vec<Scalar> = (0..5).map(|_| BLS12381ScalarField::rand()).collect();
        assert!(Commitments::lift(3, &shares).is_err());
    }

    #[test]
    fn encodings_have_the_documented_widths() {
        let s = BLS12381ScalarField::rand();
        assert_eq!(g2_hex(&BLS12381TwistCurve::generator().operate_with_self(s.representative())).len(), 2 * 96);
        assert_eq!(gt_hex(&lift_to_gt(&s).unwrap()).len(), 2 * 576);
    }
}
