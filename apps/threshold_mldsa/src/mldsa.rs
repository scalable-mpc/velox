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
    /// The FIPS 204 name, for example `"ML-DSA-65"`.
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

/// ML-DSA-65, security category 3.
pub const ML_DSA_65: ParameterSet = ParameterSet {
    name: "ML-DSA-65",
    k: 6,
    l: 5,
    eta: 4,
    tau: 49,
    beta: 196,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 55,
    lambda: 192,
};

/// ML-DSA-87, security category 5.
pub const ML_DSA_87: ParameterSet = ParameterSet {
    name: "ML-DSA-87",
    k: 8,
    l: 7,
    eta: 2,
    tau: 60,
    beta: 120,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 75,
    lambda: 256,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The relations between the parameters that the threshold protocol
    /// uses: `β = τ·η`, and `(q − 1)/(2γ2) = 16`, so the high bits `w1` are
    /// the low 4 bits of `−16·w + (q − 1)/2` (Quorus, Protocol 6).
    #[test]
    fn parameter_relations() {
        for set in [ML_DSA_65, ML_DSA_87] {
            assert_eq!(set.beta, set.tau as u32 * set.eta, "{}", set.name);
            assert_eq!((Q - 1) / (2 * set.gamma2), 16, "{}", set.name);
            assert_eq!((Q - 1) % (2 * set.gamma2), 0, "{}", set.name);
        }
    }
}
