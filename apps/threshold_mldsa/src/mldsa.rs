//! ML-DSA (FIPS 204) in the clear: what every party computes alone, on
//! public values or on its own shares by linear maps.

use velox::fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER;

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

#[cfg(test)]
mod tests {
    use super::*;

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
