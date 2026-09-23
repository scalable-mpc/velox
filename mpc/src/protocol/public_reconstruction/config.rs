//! What a public reconstruction is told when it starts, and what it is for.

/// The degree of the sharings being reconstructed. Decides how many of the
/// `n − t` L1 shares carry information: a degree-`t` sharing has `t` points
/// of redundancy among them, which the L1 step checks; a degree-`2t` sharing
/// has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharingDegree {
    T,
    TwoT,
}

/// Who started the reconstruction, which is who gets the values back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconKind {
    /// The linear multiplication protocol: masked products, degree `2t`.
    Multiplication,
    /// An application's `DepthInput::MaskedMultiply`: products under the
    /// application's own mask, degree `2t`, handed back public.
    MaskedMultiplication,
    /// Random bit generation: the squares of random sharings, degree `t`.
    RandBit,
    /// An application's `DepthInput::Reveal`: degree `t`.
    Reveal,
}

/// The two options of a public reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReconConfig {
    pub degree: SharingDegree,
    /// Add a fresh degree-`2t` sharing of zero to every L1 message, so the
    /// polynomial a party recovers at L1 has uniformly random coefficients
    /// `1..2t` whatever the input sharings' polynomials looked like. Needed
    /// when those polynomials are not themselves uniformly random — a product
    /// `f_a · f_b` exposes the factor polynomials, hence honest parties'
    /// shares of `a` and `b`. Not needed, and not offered, for degree-`t`
    /// inputs: a degree-`2t` zero sharing would raise their degree, and a
    /// degree-`t` value whose polynomial is structured should be masked by
    /// the caller instead, as the ops layer does.
    pub privacy: bool,
}

impl ReconConfig {
    /// Degree-`t`, no zero term: what a fresh-masked value needs.
    pub const REVEAL: Self = Self { degree: SharingDegree::T, privacy: false };
    /// Degree-`t`, no zero term: the squares are outputs of a multiplication,
    /// so their polynomials are uniformly random already.
    pub const RAND_BIT: Self = Self { degree: SharingDegree::T, privacy: false };
    /// Degree-`2t` with the zero term: masked products.
    pub const MULTIPLICATION: Self = Self { degree: SharingDegree::TwoT, privacy: true };

    pub fn new(degree: SharingDegree, privacy: bool) -> Result<Self, String> {
        if privacy && degree == SharingDegree::T {
            return Err("a degree-2t zero term cannot be added to degree-t sharings; mask them instead".to_string());
        }
        Ok(Self { degree, privacy })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn privacy_needs_degree_2t() {
        assert!(ReconConfig::new(SharingDegree::T, true).is_err());
        assert_eq!(ReconConfig::new(SharingDegree::T, false).unwrap(), ReconConfig::REVEAL);
        assert_eq!(ReconConfig::new(SharingDegree::TwoT, true).unwrap(), ReconConfig::MULTIPLICATION);
        assert!(ReconConfig::new(SharingDegree::TwoT, false).is_ok());
    }
}
