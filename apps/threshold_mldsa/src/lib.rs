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
//! - [`preprocessing`] — the sharings of the signing attempts.
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
//!
//! # Preprocessing
//!
//! The same run prepares `B` signing attempts (Quorus Protocol 8, see
//! [`preprocessing`]). Their errors `e_w` are kept candidates of the same
//! op-depths 1 to 3, after `s` and `e`, and `[w] = A·[y] + [e_w]` is computed
//! with `[t]` in op-depth 3.
//!
//! 5. `[w1]` of all attempts: the Planner's `CanonicalMod` with `d = 44`.

pub mod mldsa;
pub mod preprocessing;
pub mod small_coefficients;

use anyhow::{bail, Result};
use async_trait::async_trait;
use mldsa::{expand_a, into_polys, matrix_vector_product, Fq, Poly, PublicKey, ML_DSA_44};
use preprocessing::{Attempt, PendingAttempts, W1_MODULUS, W_COEFFICIENTS_PER_ATTEMPT, Y_BITS_PER_ATTEMPT};
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

pub struct ThresholdMlDsa {
    /// `B`: the signing attempts that the run prepares.
    attempts: usize,
    /// The candidates for `s`, `e` and the `e_w` of each attempt, from
    /// preprocessing on.
    candidates: Option<Candidates>,
    /// The bits of `ρ`, as sharings of `±1`, until they are opened.
    seed_bits: Vec<Fq>,
    /// The bits of `y` of all attempts, as sharings of 0 or 1, until
    /// op-depth 3.
    y_bits: Vec<Fq>,
    /// The seed `ρ` of the matrix `A`, from op-depth 3 on.
    rho: [u8; 32],
    /// `[s]` and `[e]`, from op-depth 3 on.
    s: Vec<Poly>,
    e: Vec<Poly>,
    /// The public key, once `t` is open.
    pub public_key: Option<PublicKey>,
    /// The attempts from op-depth 3 until `[w1]` is known.
    pending: Option<PendingAttempts>,
    /// The prepared attempts, once `[w1]` is known.
    prepared: Vec<Attempt>,
}

impl ThresholdMlDsa {
    /// A run that makes a key and prepares `attempts` signing attempts.
    pub fn new(attempts: usize) -> Self {
        Self {
            attempts,
            candidates: None,
            seed_bits: Vec::new(),
            y_bits: Vec::new(),
            rho: [0; 32],
            s: Vec::new(),
            e: Vec::new(),
            public_key: None,
            pending: None,
            prepared: Vec::new(),
        }
    }

    /// The small coefficients of the run: `s`, `e`, and `e_w` of each
    /// attempt.
    fn small_coefficients(&self) -> usize {
        SECRET_COEFFICIENTS + self.attempts * W_COEFFICIENTS_PER_ATTEMPT
    }

    fn candidates(&self) -> &Candidates {
        self.candidates.as_ref().expect("the candidates are set in preprocessing")
    }

    /// Op-depth 3 is complete: `ρ`, `[s]`, `[e]` and the attempts' `[w]`,
    /// then open `[t] = A·[s] + [e]`.
    fn open_t(&mut self, opened: Vec<Fq>) -> Result<OpDepthInput<F>> {
        let (reject, seed) = opened.split_at(opened.len() - SEED_BITS);
        for (i, bit) in seed.iter().enumerate() {
            self.rho[i / 8] |= u8::from(*bit == Fq::one()) << (i % 8);
        }
        let mut kept = self.candidates().kept(reject, self.small_coefficients())?;
        let errors = kept.split_off(SECRET_COEFFICIENTS);
        let mut polys = into_polys(&kept);
        self.e = polys.split_off(ML_DSA_44.l);
        self.s = polys;
        let a_hat = expand_a(&self.rho, &ML_DSA_44);
        self.pending = Some(PendingAttempts::new(&a_hat, &std::mem::take(&mut self.y_bits), &errors));
        let t = matrix_vector_product(&a_hat, &self.s)
            .into_iter()
            .zip(&self.e)
            .flat_map(|(a_s, e)| a_s.into_iter().zip(e).map(|(a_s, e)| a_s + e))
            .collect();
        OpDepthInput::op(4, Op::Reveal { x: t })
    }
}

#[async_trait]
impl PlannerApplication<F> for ThresholdMlDsa {
    fn preprocessing_count(&self) -> PlannerCounts {
        let candidates = candidates_for(self.small_coefficients());
        PlannerCounts::new(
            vec![
                OpParams::new(OpType::Mul, candidates),
                OpParams::new(OpType::Mul, candidates),
                OpParams::new(OpType::Reveal, candidates + SEED_BITS),
                OpParams::new(OpType::Reveal, ML_DSA_44.k * RING_DEGREE),
                OpParams::new(OpType::CanonicalMod { d: W1_MODULUS }, self.attempts * W_COEFFICIENTS_PER_ATTEMPT),
            ],
            0,
        )
        .with_random_wires(BITS_PER_CANDIDATE * candidates + self.attempts * Y_BITS_PER_ATTEMPT + SEED_BITS, 0)
    }

    /// No party has inputs: the key is made in the run.
    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<Fq>) -> Result<OpDepthInput<F>> {
        log::warn!("ThresholdMlDsa: ignoring {} input sharings from party {}", shares.len(), party);
        Ok(OpDepthInput::Waiting)
    }

    /// The engine's random bits are sharings of `±1`; `(1 + [b]) / 2` is 0
    /// or 1. The seed bits stay `±1`: they are opened as they are. The bits
    /// are, in order: the candidates, `y` of each attempt, the seed.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        let mut bits = wires.bits;
        self.seed_bits = bits.split_off(bits.len() - SEED_BITS);
        let half = Fq::from(2u64).inv().expect("2 is invertible");
        let mut bits: Vec<Fq> = bits.iter().map(|b| (Fq::one() + b) * &half).collect();
        self.y_bits = bits.split_off(bits.len() - self.attempts * Y_BITS_PER_ATTEMPT);
        let candidates = Candidates::new(&bits);
        let (x, y) = candidates.low_product_operands();
        self.candidates = Some(candidates);
        log::info!("ThresholdMlDsa: key generation and {} attempts from {} candidates", self.attempts, x.len());
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
                let x = self.pending.as_ref().expect("set in op-depth 3").high_bits_input();
                OpDepthInput::op(5, Op::CanonicalMod { x, d: W1_MODULUS })
            }
            5 => {
                let pending = self.pending.take().expect("set in op-depth 3");
                self.prepared = pending.complete(&result.shares()?);
                log::info!("ThresholdMlDsa: {} signing attempts prepared", self.prepared.len());
                Ok(OpDepthInput::Done(Vec::new()))
            }
            depth => bail!("ThresholdMlDsa has no op-depth {}", depth),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mldsa::{centered, high_bits, shake256};
    use velox::{Application, DepthInput, Planner};

    /// The application on the real Planner, over a plaintext engine: each
    /// share is its value. The random bits come from xorshift64.
    async fn run(seed: u64, attempts: usize) -> Planner<F, ThresholdMlDsa> {
        let mut planner = Planner::new(ThresholdMlDsa::new(attempts)).unwrap();
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
                    assert!(out.is_empty(), "the run has no output wires");
                    return planner;
                }
                DepthInput::Multiply { depth, x, y } => {
                    planner.on_depth_complete(depth, x.iter().zip(&y).map(|(a, b)| a * b).collect()).await.unwrap()
                }
                DepthInput::Reveal { depth, values } => planner.on_reveal_complete(depth, values).await.unwrap(),
                DepthInput::MaskedMultiply { depth, x, y, mask } => {
                    let opened = x.iter().zip(&y).zip(&mask).map(|((a, b), m)| a * b + m).collect();
                    planner.on_reveal_complete(depth, opened).await.unwrap()
                }
                other => panic!("unexpected {:?}", other),
            };
        }
    }

    /// The secrets have coefficients in `[−2, 2]`, every value of that range
    /// occurs, `t = A·s + e` with `A` from the opened seed, and the public key
    /// has 1,312 bytes with `tr = H(pk)`.
    #[tokio::test]
    async fn key_generation_makes_a_consistent_key() {
        let planner = run(0x5eed, 1).await;
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
        let other = run(0xbeef, 1).await;
        assert_ne!(other.app().rho, app.rho, "the seed comes from the random bits");
    }

    /// Each attempt: `y` in `(−γ1, γ1]`, `w1 = HighBits(w)` for
    /// `w = w0 + 2γ2·w1`, and `w − A·y` (the error `e_w`) in `[−2, 2]`.
    #[tokio::test]
    async fn preprocessing_prepares_the_attempts() {
        let planner = run(0x7e57, 2).await;
        let app = planner.app();
        assert_eq!(app.prepared.len(), 2);
        let a_hat = expand_a(&app.rho, &ML_DSA_44);
        let (gamma1, alpha) = (i64::from(ML_DSA_44.gamma1), Fq::from(2 * u64::from(ML_DSA_44.gamma2)));
        for attempt in &app.prepared {
            assert!(attempt.y.iter().flatten().map(centered).all(|y| -gamma1 < y && y <= gamma1));
            let a_y = matrix_vector_product(&a_hat, &attempt.y);
            for ((w0, w1), a_y) in attempt.w0.iter().zip(&attempt.w1).zip(&a_y) {
                for ((w0, w1), a_y) in w0.iter().zip(w1).zip(a_y) {
                    let w = w0 + w1 * &alpha;
                    assert_eq!(u64::from(w1.representative()), u64::from(high_bits(w.representative(), ML_DSA_44.gamma2)));
                    assert!(centered(&(w - a_y)).abs() <= 2);
                }
            }
        }
        assert_ne!(app.prepared[0].y[0], app.prepared[1].y[0], "each attempt has its own y");
    }
}
