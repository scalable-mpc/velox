//! Threshold decryption of LWE ciphertexts on the Planner — issue 10, after
//! Zyskind, Zarchy, Leibovich and Peikert, "High-Throughput Universally
//! Composable Threshold FHE Decryption" (CCS 2025).
//!
//! # The scheme
//!
//! An LWE ciphertext `(a, b)` over `Z_q`, `q = 2^64`, encrypts a bit `μ`
//! under a binary secret key `s ∈ {0,1}^n`, `n = 1024`, as
//! `b = ⟨a, s⟩ + μ·2^63 + e mod q` with noise `|e| < 2^62`. Decryption
//! computes `b − ⟨a, s⟩ = μ·2^63 + e` and rounds the noise away. In the
//! threshold setting the key is secret-shared, and opening `b − ⟨a, s⟩`
//! would reveal `e`, which leaks the key. So the noise is removed inside the
//! MPC first, and only then is anything opened.
//!
//! # What the application computes
//!
//! Over Mersenne-127 (`p = 2^127 − 1`) field, which holds the inner product as an
//! integer with room for a statistical mask. For each ciphertext:
//!
//! 1. `[z] = b + Σ (q − a_i)·[s_i] + 2^62`, locally. `z ≡ μ·2^63 + e + 2^62
//!    (mod q)` with `e + 2^62 ∈ [0, 2^63)`, and `z < 2^75` as an integer.
//! 2. `[e'] = [z mod 2^63]`, by one of two methods ([`Method`]):
//!    - **carry**: the Planner's `Mod2m` — a masked reveal, then the carry
//!      tree's 6 multiplication rounds (88 multiplications) once `z` is known;
//!    - **table**: the paper's preprocessed lookup tables
//!      ([`lookup_tables`]) — 4 rounds (2351 multiplications) that need no
//!      `z`, then 2 reveals and no multiplication once `z` is known.
//! 3. `[Q] = ([z] − [e']) / 2^63`, locally: `Q = μ + 2·j` for the integer
//!    `j = ⌊z / 2^64⌋ < 2^11`, which depends on the key, so `Q` is not opened.
//! 4. `[w] = [Q] + 2·[R]`, `R` a 51-bit random value from the engine's random
//!    bits, which hides `j` up to a statistical distance of `2^12 / 2^52`.
//! 5. `[w]` is an output wire: the engine opens it after verification, and
//!    every party reads `μ = w mod 2`.
//!
//! The online latency logged at the end runs from step 1 — the moment the key
//! and, for **table**, the tables are ready — to the opened outputs.
//!
//! # Inputs
//!
//! Party 0 deals the key bits as its input sharings; the other parties deal
//! none. A dealer that knows the key is a test arrangement only: a deployment
//! needs a distributed key generation. The ciphertexts are public, and every
//! party reads the same file (see [`parse_ciphertexts`]).

pub mod lookup_tables;

use std::time::Instant;

use anyhow::{bail, Result};
use async_trait::async_trait;
use lookup_tables::LtRand;
use velox::{
    fields::{mersenne_127::M127, Mersenne127Field},
    FieldElement, MersennePrimeField, Op, OpDepthInput, OpParams, OpResult, OpType, PlannerApplication, PlannerCounts,
    RandomWireShares,
};

type F = Mersenne127Field;
type E = FieldElement<F>;

/// LWE dimension: the number of key bits.
pub const LWE_DIMENSION: usize = 1024;
/// `log₂ q − log₂ (plaintext modulus)`: the noise occupies the low 63 bits.
pub const NOISE_BITS: usize = 63;
/// Random bits behind the output mask `R`.
pub const OUTPUT_MASK_BITS: usize = 51;

/// How step 2 removes the noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// The Planner's `Mod2m`: the carry tree after `z` is known.
    Carry,
    /// The paper's lookup tables, prepared before `z` is known.
    Table,
}

/// A public LWE ciphertext: `n` values `a_i` and `b`, all mod `2^64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ciphertext {
    pub a: Vec<u64>,
    pub b: u64,
}

/// One ciphertext per line, `a_1 … a_n b` in decimal; blank lines and lines
/// starting with `#` are skipped.
pub fn parse_ciphertexts(text: &str) -> Result<Vec<Ciphertext>> {
    let mut ciphertexts = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split_whitespace()
            .map(|v| v.parse::<u64>().map_err(|e| anyhow::anyhow!("line {}: {:?}: {}", index + 1, v, e)))
            .collect::<Result<Vec<u64>>>()?;
        if values.len() != LWE_DIMENSION + 1 {
            bail!("line {}: {} values, a ciphertext has {}", index + 1, values.len(), LWE_DIMENSION + 1);
        }
        let (a, b) = values.split_at(LWE_DIMENSION);
        ciphertexts.push(Ciphertext { a: a.to_vec(), b: b[0] });
    }
    if ciphertexts.is_empty() {
        bail!("no ciphertexts");
    }
    Ok(ciphertexts)
}

pub struct FheDecrypt {
    ciphertexts: Vec<Ciphertext>,
    method: Method,
    /// Party 0's input: the key bits it deals. `None` at the other parties.
    key: Option<Vec<u8>>,
    /// `[s_i]`, once party 0's input sharing has terminated.
    key_shares: Option<Vec<E>>,
    /// The output masks' bits as 0/1 sharings, once preprocessing is done.
    output_mask_bits: Option<Vec<E>>,
    /// **table** only: the tables and their masks, from preprocessing on.
    tables: Option<LtRand>,
    tables_ready: bool,
    /// `[z]` per ciphertext, kept for step 3 once the noise is known.
    z: Vec<E>,
    online_start: Option<Instant>,
    /// The decrypted bits, once the outputs are opened.
    pub plaintexts: Option<Vec<u8>>,
}

impl FheDecrypt {
    pub fn new(ciphertexts: Vec<Ciphertext>, key: Option<Vec<u8>>, method: Method) -> Self {
        Self {
            ciphertexts,
            method,
            key,
            key_shares: None,
            output_mask_bits: None,
            tables: None,
            tables_ready: false,
            z: Vec::new(),
            online_start: None,
            plaintexts: None,
        }
    }

    /// The op-depths of **table**'s preparation, before its two reveals.
    fn preparation_rounds() -> usize {
        LtRand::preparation_widths().len()
    }

    /// Step 1, and the first op-depth that needs `[z]`, once the key sharings,
    /// the random bits and (for **table**) the tables are all in — they
    /// arrive in any order.
    fn start_when_ready(&mut self) -> Result<OpDepthInput<F>> {
        let ready = self.output_mask_bits.is_some() && (self.method == Method::Carry || self.tables_ready);
        let (Some(key), true, true) = (&self.key_shares, ready, self.z.is_empty()) else {
            return Ok(OpDepthInput::Waiting);
        };
        let offset = E::from(1u64 << (NOISE_BITS - 1));
        self.z = self
            .ciphertexts
            .iter()
            .map(|ct| {
                ct.a.iter().zip(key).fold(E::from(ct.b) + offset, |acc, (a, s)| acc + E::from(a.wrapping_neg()) * s)
            })
            .collect();
        self.online_start = Some(Instant::now());
        log::info!("FheDecrypt: removing the noise of {} ciphertexts ({:?})", self.z.len(), self.method);
        match self.method {
            Method::Carry => OpDepthInput::op(1, Op::Mod2m { x: self.z.clone(), m: NOISE_BITS }),
            Method::Table => {
                let x = self.tables.as_ref().expect("prepared").mask_inputs(&self.z);
                OpDepthInput::op(Self::preparation_rounds() + 1, Op::Reveal { x })
            }
        }
    }

    /// Steps 3 and 4: `[w] = ([z] − [e']) / 2^63 + 2·[R]`, the outputs.
    fn remove_noise(&self, noise: &[E]) -> Result<OpDepthInput<F>> {
        let inv_l = E::new(M127(1u128 << NOISE_BITS)).inv().expect("2^63 is invertible");
        let bits = self.output_mask_bits.as_ref().expect("the noise is removed after preprocessing");
        let w = self
            .z
            .iter()
            .zip(noise)
            .zip(bits.chunks(OUTPUT_MASK_BITS))
            .map(|((z, e), mask)| {
                let r = mask.iter().rev().fold(E::zero(), |acc, bit| acc.double() + bit);
                (z - e) * inv_l + r.double()
            })
            .collect();
        Ok(OpDepthInput::Done(w))
    }
}

#[async_trait]
impl PlannerApplication<F> for FheDecrypt {
    /// **carry**: one `Mod2m` op-depth. **table**: the 4 preparation `Mul`s,
    /// then 2 `Reveal`s.
    fn preprocessing_count(&self) -> PlannerCounts {
        let n = self.ciphertexts.len();
        match self.method {
            Method::Carry => PlannerCounts::new(vec![OpParams::new(OpType::Mod2m { m: NOISE_BITS }, n)], n)
                .with_random_wires(OUTPUT_MASK_BITS * n, 0),
            Method::Table => {
                let mut ops: Vec<OpParams> =
                    LtRand::preparation_widths().into_iter().map(|w| OpParams::new(OpType::Mul, w * n)).collect();
                ops.extend([OpParams::new(OpType::Reveal, n), OpParams::new(OpType::Reveal, n)]);
                PlannerCounts::new(ops, n).with_random_wires((lookup_tables::MASK_BITS + OUTPUT_MASK_BITS) * n, 0)
            }
        }
    }

    async fn inputs(&mut self) -> Vec<E> {
        self.key.iter().flatten().map(|&bit| E::from(bit as u64)).collect()
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<E>) -> Result<OpDepthInput<F>> {
        if party != 0 {
            log::warn!("FheDecrypt: ignoring {} input sharings from party {}; only party 0 deals the key", shares.len(), party);
            return Ok(OpDepthInput::Waiting);
        }
        if shares.len() != LWE_DIMENSION {
            bail!("party 0 dealt {} key sharings, the key has {} bits", shares.len(), LWE_DIMENSION);
        }
        self.key_shares = Some(shares);
        self.start_when_ready()
    }

    /// The engine's random bits are sharings of `±1`; `(1 + [b]) / 2` is 0/1.
    /// **table** takes the tables' masks off the front and starts preparing.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        let half = E::from(2u64).inv().expect("2 is invertible");
        let mut bits: Vec<E> = wires.bits.iter().map(|b| (E::one() + b) * half).collect();
        match self.method {
            Method::Carry => {
                self.output_mask_bits = Some(bits);
                self.start_when_ready()
            }
            Method::Table => {
                let output_bits = bits.split_off(lookup_tables::MASK_BITS * self.ciphertexts.len());
                self.output_mask_bits = Some(output_bits);
                let tables = LtRand::new(&bits);
                let (x, y) = tables.preparation_operands(0);
                self.tables = Some(tables);
                log::info!("FheDecrypt: preparing the lookup tables of {} ciphertexts", self.ciphertexts.len());
                OpDepthInput::op(1, Op::Mul { x, y })
            }
        }
    }

    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        let rounds = Self::preparation_rounds();
        match (self.method, depth) {
            (Method::Carry, 1) => self.remove_noise(&result.shares()?),
            // A preparation round; after the last, the tables are ready.
            (Method::Table, d) if d <= rounds => {
                let tables = self.tables.as_mut().expect("prepared from preprocessing");
                if !tables.preparation_complete(d - 1, &result.shares()?) {
                    let (x, y) = tables.preparation_operands(d);
                    return OpDepthInput::op(d + 1, Op::Mul { x, y });
                }
                self.tables_ready = true;
                log::info!("FheDecrypt: lookup tables ready");
                self.start_when_ready()
            }
            // `z'` is open: look up the signs, open the masked `y`.
            (Method::Table, d) if d == rounds + 1 => {
                let x = self.tables.as_mut().expect("prepared").look_up_signs(&result.public()?);
                OpDepthInput::op(d + 1, Op::Reveal { x })
            }
            // The masked `y` is open: look up `[u]`, which gives the noise.
            (Method::Table, d) if d == rounds + 2 => {
                let noise = self.tables.as_ref().expect("prepared").noise(&result.public()?);
                self.remove_noise(&noise)
            }
            (method, depth) => bail!("FheDecrypt ({:?}) has no op-depth {}", method, depth),
        }
    }

    /// Step 5: `μ = w mod 2`.
    async fn on_output(&mut self, outputs: Vec<E>) -> Result<()> {
        let plaintexts: Vec<u8> = outputs.iter().map(|w| (F::to_canonical_u128(w) & 1) as u8).collect();
        let latency = self.online_start.map(|t| t.elapsed().as_millis()).unwrap_or_default();
        log::info!("FheDecrypt: online latency {} ms for {} ciphertexts", latency, plaintexts.len());
        log::info!(
            "FheDecrypt: plaintexts {}",
            plaintexts.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" ")
        );
        self.plaintexts = Some(plaintexts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velox::{Application, DepthInput, Planner};

    /// xorshift64: deterministic, reproducible failures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// `b = ⟨a, s⟩ + μ·2^63 + e mod 2^64`.
    fn encrypt(rng: &mut Rng, key: &[u8], mu: u8, e: i64) -> Ciphertext {
        let a: Vec<u64> = (0..LWE_DIMENSION).map(|_| rng.next()).collect();
        let inner = a.iter().zip(key).fold(0u64, |acc, (a, &s)| acc.wrapping_add(a.wrapping_mul(s as u64)));
        let b = inner.wrapping_add((mu as u64) << 63).wrapping_add(e as u64);
        Ciphertext { a, b }
    }

    /// The application hosted by the real Planner, over a plaintext engine.
    /// The key arrives after preprocessing, so **table** prepares its tables
    /// before `[z]` exists, as in a run.
    async fn decrypt(ciphertexts: Vec<Ciphertext>, key: Vec<u8>, method: Method) -> Vec<u8> {
        let key_shares: Vec<E> = key.iter().map(|&s| E::from(s as u64)).collect();
        let mut planner = Planner::new(FheDecrypt::new(ciphertexts, Some(key), method)).unwrap();
        let mut rng = Rng(0x5eed);
        let bits = (0..planner.random_wires().bits).map(|_| if rng.next() & 1 == 1 { E::one() } else { -E::one() }).collect();
        let mut next = planner.on_preprocessing_complete(RandomWireShares::new(bits, Vec::new())).await.unwrap();
        assert_eq!(next.is_waiting(), method == Method::Carry, "only table starts before the key is in");
        let mut key_shares = Some(key_shares);
        if method == Method::Carry {
            next = planner.input_sharing_termination(0, key_shares.take().unwrap()).await.unwrap();
        }
        loop {
            next = match next {
                DepthInput::Done(out) => {
                    planner.on_output(out).await.unwrap();
                    return planner.app().plaintexts.clone().unwrap();
                }
                DepthInput::Multiply { depth, x, y } => {
                    planner.on_depth_complete(depth, x.iter().zip(&y).map(|(a, b)| a * b).collect()).await.unwrap()
                }
                DepthInput::Reveal { depth, values } => planner.on_reveal_complete(depth, values).await.unwrap(),
                // table's preparation is done and waits for the key.
                DepthInput::Waiting => planner.input_sharing_termination(0, key_shares.take().expect("the key once")).await.unwrap(),
                other => panic!("unexpected {:?}", other),
            };
        }
    }

    /// The noise at both ends of its range, and random ciphertexts.
    #[tokio::test]
    async fn decrypts_boundary_and_random_ciphertexts() {
        let mut rng = Rng(0x1234);
        let key: Vec<u8> = (0..LWE_DIMENSION).map(|_| (rng.next() & 1) as u8).collect();
        let mut cases: Vec<(u8, i64)> = [0u8, 1].iter().flat_map(|&mu| [(mu, -(1 << 62)), (mu, 0), (mu, (1 << 62) - 1)]).collect();
        cases.extend((0..20).map(|_| ((rng.next() & 1) as u8, (rng.next() as i64) >> 24)));
        let ciphertexts: Vec<Ciphertext> = cases.iter().map(|&(mu, e)| encrypt(&mut rng, &key, mu, e)).collect();
        let want: Vec<u8> = cases.iter().map(|&(mu, _)| mu).collect();
        assert_eq!(decrypt(ciphertexts.clone(), key.clone(), Method::Carry).await, want, "carry");
        assert_eq!(decrypt(ciphertexts, key, Method::Table).await, want, "table");
    }

    #[test]
    fn parses_ciphertexts() {
        let line = |b: u64| (1..=LWE_DIMENSION as u64).chain([b]).map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
        let text = format!("# two ciphertexts\n{}\n\n{}\n", line(7), line(8));
        let parsed = parse_ciphertexts(&text).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!((parsed[0].a[1023], parsed[0].b, parsed[1].b), (1024, 7, 8));
        assert!(parse_ciphertexts("1 2 3\n").is_err(), "too few values");
        assert!(parse_ciphertexts("").is_err(), "no ciphertexts");
    }

    /// Only party 0 deals the key; any other party deals nothing.
    #[tokio::test]
    async fn only_party_0_deals() {
        let ct = Ciphertext { a: vec![0; LWE_DIMENSION], b: 0 };
        assert_eq!(FheDecrypt::new(vec![ct.clone()], Some(vec![1; LWE_DIMENSION]), Method::Carry).inputs().await.len(), LWE_DIMENSION);
        assert!(FheDecrypt::new(vec![ct], None, Method::Table).inputs().await.is_empty());
    }
}
