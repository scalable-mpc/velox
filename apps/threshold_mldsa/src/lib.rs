//! Threshold ML-DSA signing on the Planner — issue 19, after Bienstock et
//! al., "Quorus: Threshold ML-DSA" (USENIX Security 2026, eprint 2025/1163).
//!
//! The parties hold Shamir shares of an ML-DSA secret key over the ML-DSA
//! field `F_q`, `q = 8380417`, and sign a message without reconstructing the
//! key. The signature is a standard FIPS 204 signature: any ML-DSA verifier
//! accepts it.
//!
//! - [`mldsa`] — ML-DSA in the clear: the parameter sets and the public
//!   arithmetic that every party computes alone.
pub mod mldsa;
