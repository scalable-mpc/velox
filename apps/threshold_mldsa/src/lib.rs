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
//! - [`small_coefficients`] — sharings of coefficients uniform in `[−η, η]`.
//!
//! # Key generation
//!
//! Each run makes a new key (Quorus Protocol 5, key generation), in four
//! op-depths:
//!
//! 1. and 2. The reject bits of the [`small_coefficients`] candidates: two
//!    multiplications.
//! 3. Open the reject bits and 256 random bits, the seed `ρ`. The first kept
//!    candidates are `[s]` (`l` ring elements) and `[e]` (`k` ring
//!    elements). Each party computes `[t] = A·[s] + [e]` on its shares.
//! 4. Open `t`. Each party computes the [`PublicKey`] from `ρ` and `t`.

pub mod mldsa;
pub mod small_coefficients;

use anyhow::{bail, Result};
use async_trait::async_trait;
use mldsa::{expand_a, matrix_vector_product, Fq, Poly, PublicKey, ML_DSA_44};
use small_coefficients::{candidates_for, Candidates, BITS_PER_CANDIDATE};
use velox::{
    fields::{ml_dsa::RING_DEGREE, MlDsaField},
    Op, OpDepthInput, OpParams, OpResult, OpType, PlannerApplication, PlannerCounts, RandomWireShares,
};

type F = MlDsaField;

/// The bits of the seed `ρ` of the matrix `A`.
const SEED_BITS: usize = 256;

/// The coefficients of `s` and `e`.
const SECRET_COEFFICIENTS: usize = (ML_DSA_44.l + ML_DSA_44.k) * RING_DEGREE;

/// Cuts a flat list of coefficients into ring elements of `RING_DEGREE` each.
fn into_polys(coefficients: &[Fq]) -> Vec<Poly> {
    coefficients.chunks_exact(RING_DEGREE).map(|c| c.to_vec().try_into().expect("RING_DEGREE coefficients")).collect()
}

pub struct ThresholdMlDsa {
    /// The candidates for `s` and `e`, from preprocessing on.
    candidates: Option<Candidates>,
    /// The bits of `ρ`, as sharings of `±1`, until they are opened.
    seed_bits: Vec<Fq>,
    /// The seed `ρ` of the matrix `A`, from op-depth 3 on.
    rho: [u8; 32],
    /// `[s]` and `[e]`, from op-depth 3 on.
    s: Vec<Poly>,
    e: Vec<Poly>,
    /// The public key, once `t` is open.
    pub public_key: Option<PublicKey>,
}

impl ThresholdMlDsa {
    pub fn new() -> Self {
        Self { candidates: None, seed_bits: Vec::new(), rho: [0; 32], s: Vec::new(), e: Vec::new(), public_key: None }
    }

    fn candidates(&self) -> &Candidates {
        self.candidates.as_ref().expect("the candidates are set in preprocessing")
    }

    /// Op-depth 3 is complete: `ρ`, `[s]`, `[e]`, then open `[t] = A·[s] + [e]`.
    fn open_t(&mut self, opened: Vec<Fq>) -> Result<OpDepthInput<F>> {
        let (reject, seed) = opened.split_at(opened.len() - SEED_BITS);
        for (i, bit) in seed.iter().enumerate() {
            self.rho[i / 8] |= u8::from(*bit == Fq::one()) << (i % 8);
        }
        let kept = self.candidates().kept(reject, SECRET_COEFFICIENTS)?;
        let mut polys = into_polys(&kept);
        self.e = polys.split_off(ML_DSA_44.l);
        self.s = polys;
        let t = matrix_vector_product(&expand_a(&self.rho, &ML_DSA_44), &self.s)
            .into_iter()
            .zip(&self.e)
            .flat_map(|(a_s, e)| a_s.into_iter().zip(e).map(|(a_s, e)| a_s + e))
            .collect();
        OpDepthInput::op(4, Op::Reveal { x: t })
    }
}

impl Default for ThresholdMlDsa {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlannerApplication<F> for ThresholdMlDsa {
    fn preprocessing_count(&self) -> PlannerCounts {
        let candidates = candidates_for(SECRET_COEFFICIENTS);
        PlannerCounts::new(
            vec![
                OpParams::new(OpType::Mul, candidates),
                OpParams::new(OpType::Mul, candidates),
                OpParams::new(OpType::Reveal, candidates + SEED_BITS),
                OpParams::new(OpType::Reveal, ML_DSA_44.k * RING_DEGREE),
            ],
            0,
        )
        .with_random_wires(BITS_PER_CANDIDATE * candidates + SEED_BITS, 0)
    }

    /// No party has inputs: the key is made in the run.
    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<Fq>) -> Result<OpDepthInput<F>> {
        log::warn!("ThresholdMlDsa: ignoring {} input sharings from party {}", shares.len(), party);
        Ok(OpDepthInput::Waiting)
    }

    /// The engine's random bits are sharings of `±1`; `(1 + [b]) / 2` is 0
    /// or 1. The seed bits stay `±1`: they are opened as they are.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        let mut bits = wires.bits;
        self.seed_bits = bits.split_off(bits.len() - SEED_BITS);
        let half = Fq::from(2u64).inv().expect("2 is invertible");
        let bits: Vec<Fq> = bits.iter().map(|b| (Fq::one() + b) * &half).collect();
        let candidates = Candidates::new(&bits);
        let (x, y) = candidates.low_product_operands();
        self.candidates = Some(candidates);
        log::info!("ThresholdMlDsa: key generation from {} candidates", x.len());
        OpDepthInput::op(1, Op::Mul { x, y })
    }

    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        match depth {
            1 => {
                let (x, y) = self.candidates().reject_operands(&result.shares()?);
                OpDepthInput::op(2, Op::Mul { x, y })
            }
            2 => {
                let mut x = result.shares()?;
                x.append(&mut self.seed_bits);
                OpDepthInput::op(3, Op::Reveal { x })
            }
            3 => self.open_t(result.public()?),
            4 => {
                let key = PublicKey::new(&self.rho, into_polys(&result.public()?));
                log::info!(
                    "ThresholdMlDsa: public key of {} bytes, tr = {}",
                    key.encoded.len(),
                    key.tr.iter().map(|b| format!("{:02x}", b)).collect::<String>()
                );
                self.public_key = Some(key);
                Ok(OpDepthInput::Done(Vec::new()))
            }
            depth => bail!("ThresholdMlDsa has no op-depth {}", depth),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mldsa::{centered, shake256};
    use velox::{Application, DepthInput, Planner};

    /// The application on the real Planner, over a plaintext engine: each
    /// share is its value. The random bits come from xorshift64.
    async fn generate_key(seed: u64) -> Planner<F, ThresholdMlDsa> {
        let mut planner = Planner::new(ThresholdMlDsa::new()).unwrap();
        let mut state = seed;
        let bits = (0..planner.random_wires().bits)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state & 1 == 1 { Fq::one() } else { -Fq::one() }
            })
            .collect();
        let mut next = planner.on_preprocessing_complete(RandomWireShares::new(bits, Vec::new())).await.unwrap();
        loop {
            next = match next {
                DepthInput::Done(out) => {
                    assert!(out.is_empty(), "key generation has no output wires");
                    return planner;
                }
                DepthInput::Multiply { depth, x, y } => {
                    planner.on_depth_complete(depth, x.iter().zip(&y).map(|(a, b)| a * b).collect()).await.unwrap()
                }
                DepthInput::Reveal { depth, values } => planner.on_reveal_complete(depth, values).await.unwrap(),
                other => panic!("unexpected {:?}", other),
            };
        }
    }

    /// The secrets have coefficients in `[−2, 2]`, every value of that range
    /// occurs, `t = A·s + e` with `A` from the opened seed, and the public key
    /// has 1,312 bytes with `tr = H(pk)`.
    #[tokio::test]
    async fn key_generation_makes_a_consistent_key() {
        let planner = generate_key(0x5eed).await;
        let app = planner.app();
        let key = app.public_key.as_ref().expect("a public key");
        assert_eq!((app.s.len(), app.e.len()), (ML_DSA_44.l, ML_DSA_44.k));
        let coefficients: Vec<i64> = app.s.iter().chain(&app.e).flatten().map(centered).collect();
        for value in -2..=2 {
            assert!(coefficients.contains(&value), "{value} occurs");
        }
        assert!(coefficients.iter().all(|c| c.abs() <= 2));
        let a_s = matrix_vector_product(&expand_a(&app.rho, &ML_DSA_44), &app.s);
        for ((t, a_s), e) in key.t.iter().zip(&a_s).zip(&app.e) {
            for ((t, a_s), e) in t.iter().zip(a_s).zip(e) {
                assert_eq!(*t, a_s + e);
            }
        }
        assert_eq!(&key.encoded[..32], &app.rho);
        assert_eq!(key.encoded.len(), 1312);
        assert_eq!(key.tr, shake256(&[&key.encoded], 64));
        let other = generate_key(0xbeef).await;
        assert_ne!(other.app().rho, app.rho, "the seed comes from the random bits");
    }
}
