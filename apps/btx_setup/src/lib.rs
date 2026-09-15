//! Setup generation for batched threshold encryption on the MPC engine.
//!
//! # Notation
//!
//! `⟨x⟩` is a degree-`t` Shamir sharing of the field element `x`, and `⟨x⟩_j`
//! is party `j`'s share of it. `g₁^x`, `g₂^x`, `g_T^x` are the exponent `x`
//! lifted into the pairing groups G₁, G₂, G_T. Everything this application
//! touches is a field element or a sharing of one; it never sees a group
//! element.
//!
//! # What the scheme needs
//!
//! BTX (Agarwal, Das, Gilkalaye, Rindal, Shoup — eprint 2026/754, Fig. 3) and
//! the simple BTE scheme (Policharla — eprint 2026/760, Fig. 1) share one
//! `KeyGen`: sample a secret `τ` in the BLS12-381 scalar field and give server
//! `j` the shares `sk_j = (⟨τ¹⟩_j, …, ⟨τ^B⟩_j)` of *every* power up to the batch
//! size `B`. The public material is then derived from those shares, outside
//! this application and without interaction:
//!
//!   - `v_j^i = g₂^{⟨τ^i⟩_j}` for `i ≤ B` — party `j` lifts its own shares.
//!     The combiner uses them to check a server's partial decryption before
//!     interpolating it in.
//!   - `h_i = g₂^{τ^i}` for `i ∈ [2B] \ {B+1}` — Lagrange interpolation in the
//!     exponent of any `t+1` parties' `g₂^{⟨τ^i⟩_j}`. The combiner pairs these
//!     with the batch to open each slot; the missing power `B+1` is the
//!     puncture the encryption's security rests on.
//!   - `ek = g_T^{τ^{B+1}}` — interpolated the same way from
//!     `e(g₁^{⟨τ^{B+1}⟩_j}, g₂)`, so that `τ^{B+1}` only ever appears in G_T.
//!
//! A DKG hands out `⟨τ⟩`. Shares `⟨τ²⟩, ⟨τ³⟩, …` of the *same* `τ`, each on its
//! own fresh polynomial, with nobody ever learning `τ`, are products of a
//! shared secret — that is the part that needs MPC, and it is all this
//! application does.
//!
//! # What the application computes
//!
//! Mapped onto the [`Application`] MPC engine interaction API:
//!
//!   - [`Application::inputs`]: each party `j` deals one uniformly random
//!     field element `r_j`, so the engine's input ACSS gives everyone `⟨r_j⟩`.
//!   - [`Application::input_sharing_termination`]: once all `n` dealers have
//!     terminated, `⟨τ⟩ = Σ_j ⟨r_j⟩`, summed locally. Any honest `r_j` in the
//!     sum makes `τ` uniform and unknown to the adversary, and the ACSS hides
//!     honest inputs, so a corrupt dealer choosing last cannot bias it.
//!   - [`Application::on_depth_complete`] fills a power table by doubling:
//!     after depth `k−1` the table holds `⟨τ¹⟩ … ⟨τ^{2^{k−1}}⟩`; depth `k`
//!     multiplies `⟨τ^{2^{k−1}}⟩` by `⟨τ¹⟩ … ⟨τ^m⟩`, `m = min(2^{k−1}, 2B − 2^{k−1})`,
//!     in a single batch. `⌈log₂ 2B⌉` depths and `2B − 1` multiplications in
//!     all — for `B = 512`, ten depths of 1, 2, 4, …, 512 gates.
//!   - The last depth returns [`DepthInput::Done`] with **no** output wires.
//!     Nothing is ever reconstructed: the product of this circuit is the
//!     sharings `⟨τ¹⟩_j … ⟨τ^{2B}⟩_j` the party still holds.
//!   - [`Application::on_output`] is the engine saying those sharings are
//!     verified and agreed on. The party then prints its shares, computes and
//!     prints its commitments to them — `g₂^{⟨τ^i⟩_j}` and, for the punctured
//!     power, `g_T^{⟨τ^{B+1}⟩_j}` (see [`commitments`]) — and is done. Nothing
//!     is written to disk. Interpolating `h_i` and `ek` from any `t+1` parties'
//!     commitments is a public computation that needs no party's secrets.
//!
//! The circuit waits for all `n` dealers, as the other applications do; the
//! engine does not yet pass its ACS-agreed dealer set to applications.

use std::collections::HashMap;

use anyhow::{bail, Result};
use async_trait::async_trait;
use velox::{Application, DepthInput, FieldElement, PreprocessingCounts, ProtocolField};

pub mod commitments;
pub use commitments::Commitments;

/// The `--field` name under which lifting into BLS12-381's groups applies.
pub const BLS381: &str = "bls381";

pub struct BtxSetup<F: ProtocolField> {
    pub num_nodes: usize,
    pub num_faults: usize,
    pub my_id: usize,
    /// `B`: the scheme's (maximum) batch size. The table runs to `2B`.
    pub batch_size: usize,

    /// The field's name as the binary's `--field` spells it. Commitments are
    /// computed only over [`BLS381`]; other fields are for benchmarking the
    /// circuit and stop at printing the shares.
    field: String,

    /// `⟨r_j⟩` from each dealer whose input ACSS has terminated — this party's
    /// own included, since its share of its own input arrives the same way.
    contributions: HashMap<usize, FieldElement<F>>,
    /// Set once `⟨τ⟩` has been written into the table, so late dealers are
    /// ignored rather than re-summed.
    tau_assembled: bool,
    /// Set once preprocessing has been handed over.
    preprocessing_done: bool,
    /// Set once depth 1 has been scheduled, so the circuit starts exactly once.
    circuit_started: bool,

    /// `powers[i]` is this party's share `⟨τ^i⟩_j`, `i = 1..=2B`. Index 0 is
    /// unused.
    powers: Vec<Option<FieldElement<F>>>,
    /// Highest depth whose results have been folded in. Depths run
    /// `1..=depth()`, and depth `k+1` cannot be scheduled before depth `k`
    /// completes, so results arrive in order.
    depths_completed: usize,
}

impl<F: ProtocolField> BtxSetup<F> {
    pub fn new(
        num_nodes: usize,
        num_faults: usize,
        my_id: usize,
        batch_size: usize,
        field: &str,
    ) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch size must be at least 1");
        }
        if my_id >= num_nodes {
            bail!("party id {} out of range for {} nodes", my_id, num_nodes);
        }
        Ok(Self {
            num_nodes,
            num_faults,
            my_id,
            batch_size,
            field: field.to_string(),
            contributions: HashMap::new(),
            tau_assembled: false,
            preprocessing_done: false,
            circuit_started: false,
            powers: vec![None; 2 * batch_size + 1],
            depths_completed: 0,
        })
    }

    /// Highest power the table holds: `2B`.
    pub fn max_power(&self) -> usize {
        2 * self.batch_size
    }

    /// Number of multiplication depths: `ceil(log2(2B))`.
    pub fn depth(&self) -> usize {
        self.gates_per_depth().len()
    }

    /// Gates at each depth, depth 1 first. Depth `k` extends the table from
    /// `2^{k-1}` to `min(2^k, 2B)` powers.
    pub fn gates_per_depth(&self) -> Vec<usize> {
        let max = self.max_power();
        let mut gates = Vec::new();
        let mut reach = 1;
        while reach < max {
            gates.push(reach.min(max - reach));
            reach *= 2;
        }
        gates
    }

    /// This party's share of `τ^i`, once the table holds it.
    pub fn power(&self, i: usize) -> Option<&FieldElement<F>> {
        self.powers.get(i).and_then(|p| p.as_ref())
    }

    /// This party's shares of `τ¹ … τ^{2B}` in order, or `None` while the table
    /// is still being filled.
    pub fn shares(&self) -> Option<Vec<FieldElement<F>>> {
        self.powers[1..].iter().cloned().collect()
    }

    /// This party's commitments to its shares, once the table is full and if
    /// the field is the BLS12-381 scalar field — `None` otherwise, since the
    /// curve's points are multiplied by that field and no other.
    pub fn commitments(&self) -> Result<Option<Commitments>> {
        let Some(shares) = self.shares() else {
            bail!("power table is not complete: {} depths of {} done", self.depths_completed, self.depth());
        };
        Self::lift(&self.field, self.batch_size, &shares)
    }

    /// The lifting behind [`commitments`](Self::commitments), as a function of
    /// its inputs so it can run off the engine's task.
    ///
    /// The shares are re-read through their bytes rather than cast, which is
    /// what makes this callable from the generic application: the bytes of a
    /// `bls381` element are the same whichever alias of the field it is
    /// typed as.
    fn lift(field: &str, batch_size: usize, shares: &[FieldElement<F>]) -> Result<Option<Commitments>> {
        if field != BLS381 {
            return Ok(None);
        }
        let scalars: Vec<commitments::Scalar> = shares
            .iter()
            .map(|s| {
                velox::fields::BLS12381ScalarField::from_bytes_be(&F::to_bytes_be(s))
                    .map_err(|e| anyhow::anyhow!("share is not a bls381 element: {:?}", e))
            })
            .collect::<Result<_>>()?;
        Commitments::lift(batch_size, &scalars).map(Some)
    }

    /// Sum every dealer's contribution into `⟨τ⟩` once all have terminated.
    fn try_assemble_tau(&mut self) {
        if self.tau_assembled || self.contributions.len() < self.num_nodes {
            return;
        }
        let tau = self
            .contributions
            .values()
            .fold(FieldElement::<F>::zero(), |acc, share| acc + share);
        self.powers[1] = Some(tau);
        self.tau_assembled = true;
        // Summed and not read again; the flag above short-circuits late dealers
        // before they could be inserted back into an emptied map.
        self.contributions.clear();
        log::info!("BtxSetup: assembled [tau] from {} dealers", self.num_nodes);
    }

    /// Schedule depth 1 as soon as both `⟨τ⟩` and the preprocessing material
    /// are in hand — the two arrive in either order.
    fn try_start_circuit(&mut self) -> Result<DepthInput<F>> {
        if self.circuit_started || !self.tau_assembled || !self.preprocessing_done {
            return Ok(DepthInput::Waiting);
        }
        self.circuit_started = true;
        self.schedule_depth(1)
    }

    /// The batch for depth `k`: `⟨τ^{2^{k−1}}⟩` against `⟨τ¹⟩ … ⟨τ^m⟩`. Past
    /// the last depth the table is full and the circuit is done — with no
    /// output wires, since the shares are the product.
    fn schedule_depth(&mut self, depth: usize) -> Result<DepthInput<F>> {
        let reach = 1usize << (depth - 1);
        let max = self.max_power();
        if reach >= max {
            log::info!(
                "BtxSetup: power table complete through tau^{} after {} depths",
                max,
                self.depths_completed
            );
            return Ok(DepthInput::Done(Vec::new()));
        }
        let m = reach.min(max - reach);
        let Some(top) = self.power(reach).cloned() else {
            bail!("scheduling depth {}: tau^{} is not in the table", depth, reach);
        };
        let y: Vec<FieldElement<F>> = (1..=m)
            .map(|i| {
                self.power(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("scheduling depth {}: tau^{} is not in the table", depth, i))
            })
            .collect::<Result<_>>()?;
        let x = vec![top; m];
        log::info!(
            "BtxSetup: depth {} multiplies tau^{} by tau^1..tau^{} ({} gates)",
            depth,
            reach,
            m,
            m
        );
        DepthInput::multiply(depth, x, y)
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for BtxSetup<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        // the circuit is multiplications only and reconstructs nothing, so both rand_bits, and outputs fields are zeros. 
        let counts = PreprocessingCounts::new(self.gates_per_depth(), 0, 0);
        log::info!(
            "BtxSetup::preprocessing_count -> batch size {}, {} multiplications over {} depths",
            self.batch_size,
            counts.mult_gates(),
            counts.depth()
        );
        counts
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        vec![F::rand()]
    }

    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        if self.tau_assembled {
            log::debug!("BtxSetup: ignoring input sharing from party {}, tau already assembled", party);
            return Ok(DepthInput::Waiting);
        }
        let Some(share) = shares.into_iter().next() else {
            bail!("party {} dealt no input; every dealer contributes one element to tau", party);
        };
        self.contributions.insert(party, share);
        log::info!(
            "BtxSetup: contribution from party {} in ({} of {} dealers)",
            party,
            self.contributions.len(),
            self.num_nodes
        );
        self.try_assemble_tau();
        self.try_start_circuit()
    }

    async fn on_preprocessing_complete(
        &mut self,
        _rand_bit_sharings: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!(
            "BtxSetup: preprocessing complete"
        );
        self.preprocessing_done = true;
        self.try_start_circuit()
    }

    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        if depth <= self.depths_completed {
            log::debug!("BtxSetup: ignoring replayed completion of depth {}", depth);
            return Ok(DepthInput::Waiting);
        }
        if depth != self.depths_completed + 1 {
            bail!(
                "depth {} completed before depth {}; this circuit's depths are sequential",
                depth,
                self.depths_completed + 1
            );
        }
        let reach = 1usize << (depth - 1);
        let expected = reach.min(self.max_power() - reach);
        if results.len() != expected {
            bail!(
                "depth {} returned {} results, expected {}",
                depth,
                results.len(),
                expected
            );
        }
        for (offset, share) in results.into_iter().enumerate() {
            self.powers[reach + 1 + offset] = Some(share);
        }
        self.depths_completed = depth;
        self.schedule_depth(depth + 1)
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        if !outputs.is_empty() {
            bail!("BtxSetup declared no output wires but {} were reconstructed", outputs.len());
        }
        let Some(shares) = self.shares() else {
            bail!("power table is not complete: {} depths of {} done", self.depths_completed, self.depth());
        };
        let (j, b) = (self.my_id, self.batch_size);
        log::info!(
            "BtxSetup: run verified. Party {}'s shares <tau^i>_{} for i = 1..={} (indices 1..={} are sk_{}), as big-endian hex:",
            j, j, 2 * b, b, j
        );
        for (index, share) in shares.iter().enumerate() {
            log::info!("BtxSetup: <tau^{}>_{} = {}", index + 1, j, commitments::hex(&F::to_bytes_be(share)));
        }

        let Some(c) = velox::fields::rayon_async({
            let field = self.field.clone();
            move || Self::lift(&field, b, &shares)
        })
        .await?
        else {
            log::info!(
                "BtxSetup: field {:?} is not the BLS12-381 scalar field; commitments are not computed. Done.",
                self.field
            );
            return Ok(());
        };

        log::info!("BtxSetup: party {}'s commitments g2^(<tau^i>_{}) for i in [2B] \\ {{B+1}}, compressed G2 as hex:", j, j);
        for (index, p) in c.low.iter().enumerate() {
            log::info!("BtxSetup: v_{}^{} = g2^(<tau^{}>_{}) = {}", j, index + 1, index + 1, j, commitments::g2_hex(p));
        }
        for (index, p) in c.high.iter().enumerate() {
            let i = b + 2 + index;
            log::info!("BtxSetup: g2^(<tau^{}>_{}) = {}", i, j, commitments::g2_hex(p));
        }
        log::info!(
            "BtxSetup: punctured power in GT only: e(g1^(<tau^{}>_{}), g2) = {}",
            b + 1, j, commitments::gt_hex(&c.middle)
        );
        log::info!(
            "BtxSetup: the public keys follow by interpolation in the exponent of any t+1 = {} parties' commitments",
            self.num_faults + 1
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests exercise the application at one concrete field; the generic
    /// parameter is what the engine binds, not something the tests vary.
    type F = velox::fields::BLS12381ScalarField;

    const NUM_NODES: usize = 4;
    const NUM_FAULTS: usize = 1;

    /// A stand-in engine. Linear operations on Shamir sharings are the same
    /// operations on the secrets, so feeding each dealer's secret `r_j` where
    /// the engine would feed `⟨r_j⟩`, and multiplying batches in the clear,
    /// makes the table directly comparable to the powers of `τ = Σ r_j`.
    struct Harness {
        app: BtxSetup<F>,
        tau: FieldElement<F>,
        /// Batch sizes the application scheduled, depth by depth.
        batches: Vec<usize>,
    }

    impl Harness {
        fn new(batch_size: usize) -> Self {
            Self {
                app: BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, batch_size, BLS381).unwrap(),
                tau: FieldElement::<F>::zero(),
                batches: Vec::new(),
            }
        }

        async fn deliver_inputs(&mut self) -> DepthInput<F> {
            let mut last = DepthInput::Waiting;
            for party in 0..NUM_NODES {
                let r = F::rand();
                self.tau = &self.tau + &r;
                last = self.app.input_sharing_termination(party, vec![r]).await.unwrap();
            }
            last
        }

        async fn deliver_preprocessing(&mut self) -> DepthInput<F> {
            self.app.on_preprocessing_complete(Vec::new()).await.unwrap()
        }

        /// Run every depth the application schedules; returns the output wires
        /// it hands back, which must be none.
        async fn run(&mut self, mut depth_input: DepthInput<F>) -> Vec<FieldElement<F>> {
            loop {
                match depth_input {
                    DepthInput::Done(outputs) => return outputs,
                    DepthInput::Waiting => panic!("the application stalled with nothing scheduled"),
                    DepthInput::Multiply { depth, x, y } => {
                        self.batches.push(x.len());
                        let results: Vec<FieldElement<F>> =
                            x.into_iter().zip(y.into_iter()).map(|(x, y)| x * y).collect();
                        depth_input = self.app.on_depth_complete(depth, results).await.unwrap();
                    }
                }
            }
        }

        fn expected_powers(&self) -> Vec<FieldElement<F>> {
            let mut powers = Vec::new();
            let mut acc = FieldElement::<F>::one();
            for _ in 0..self.app.max_power() {
                acc = &acc * &self.tau;
                powers.push(acc.clone());
            }
            powers
        }
    }

    #[test]
    fn schedule_is_log_depth_and_covers_every_power() {
        for (batch_size, gates) in [
            (1, vec![1]),
            (2, vec![1, 2]),
            (3, vec![1, 2, 2]),
            (4, vec![1, 2, 4]),
            (5, vec![1, 2, 4, 2]),
            (16, vec![1, 2, 4, 8, 16]),
            (512, vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        ] {
            let app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, batch_size, BLS381).unwrap();
            assert_eq!(app.gates_per_depth(), gates, "B = {}", batch_size);
            assert_eq!(app.gates_per_depth().iter().sum::<usize>(), 2 * batch_size - 1, "B = {}", batch_size);
            assert_eq!(app.depth(), ((2 * batch_size) as f64).log2().ceil() as usize, "B = {}", batch_size);
            let counts = app.preprocessing_count();
            assert_eq!(counts.gates_per_depth, gates);
            assert_eq!((counts.rand_bits, counts.output), (0, 0));
        }
    }

    #[tokio::test]
    async fn table_holds_the_powers_of_the_summed_tau() {
        for batch_size in [1, 2, 3, 16] {
            let mut h = Harness::new(batch_size);
            // Inputs first, then preprocessing: the second arrival starts the circuit.
            assert!(h.deliver_inputs().await.is_waiting());
            let first = h.deliver_preprocessing().await;
            let outputs = h.run(first).await;
            assert!(outputs.is_empty(), "B = {}: the circuit declares no output wires", batch_size);
            assert_eq!(h.batches, h.app.gates_per_depth(), "B = {}", batch_size);
            assert_eq!(h.app.shares().unwrap(), h.expected_powers(), "B = {}", batch_size);
        }
    }

    #[tokio::test]
    async fn starts_whichever_of_inputs_and_preprocessing_arrives_last() {
        let mut h = Harness::new(4);
        assert!(h.deliver_preprocessing().await.is_waiting());
        let first = h.deliver_inputs().await;
        assert!(matches!(first, DepthInput::Multiply { depth: 1, .. }));
        h.run(first).await;
        assert_eq!(h.app.shares().unwrap(), h.expected_powers());
    }

    #[tokio::test]
    async fn replayed_and_late_events_are_ignored() {
        let mut h = Harness::new(4);
        h.deliver_inputs().await;
        let first = h.deliver_preprocessing().await;
        let DepthInput::Multiply { depth, x, y } = first else { panic!("depth 1 not scheduled") };
        let results: Vec<FieldElement<F>> = x.iter().zip(y.iter()).map(|(x, y)| x * y).collect();
        let second = h.app.on_depth_complete(depth, results.clone()).await.unwrap();
        assert!(matches!(second, DepthInput::Multiply { depth: 2, .. }));

        // A replayed termination of depth 1 schedules nothing.
        assert!(h.app.on_depth_complete(depth, results).await.unwrap().is_waiting());
        // A late dealer after tau was assembled changes nothing.
        assert!(h.app.input_sharing_termination(1, vec![F::rand()]).await.unwrap().is_waiting());
        // Preprocessing arriving twice does not restart the circuit.
        assert!(h.app.on_preprocessing_complete(Vec::new()).await.unwrap().is_waiting());

        h.run(second).await;
        assert_eq!(h.app.shares().unwrap(), h.expected_powers());
    }

    #[tokio::test]
    async fn wrong_batch_shape_is_an_error() {
        let mut h = Harness::new(4);
        h.deliver_inputs().await;
        let DepthInput::Multiply { depth, x, .. } = h.deliver_preprocessing().await else {
            panic!("depth 1 not scheduled")
        };
        // Depth 1 has one gate; hand back two results.
        let mut results = x.clone();
        results.push(F::rand());
        assert!(h.app.on_depth_complete(depth, results).await.is_err());
        // Depth 3 before depth 2 is not this circuit.
        assert!(h.app.on_depth_complete(3, vec![F::rand()]).await.is_err());
    }

    #[tokio::test]
    async fn commitments_are_the_table_in_the_exponent() {
        use lambdaworks_math::cyclic_group::IsGroup;
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
        let g2 = lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve::generator();

        let mut h = Harness::new(2);
        assert!(h.app.commitments().is_err(), "no commitments before the table is full");
        h.deliver_inputs().await;
        let first = h.deliver_preprocessing().await;
        h.run(first).await;

        let c = h.app.commitments().unwrap().expect("bls381 lifts");
        let powers = h.expected_powers();
        assert_eq!(c.low.len(), 2);
        assert_eq!(c.high.len(), 1);
        assert_eq!(c.low[0], g2.operate_with_self(powers[0].representative()));
        assert_eq!(c.low[1], g2.operate_with_self(powers[1].representative()));
        assert_eq!(c.high[0], g2.operate_with_self(powers[3].representative()));
        assert_eq!(c.middle, commitments::lift_to_gt(&powers[2]).unwrap());

        // on_output prints and lifts; it must accept the empty output and
        // reject wires this circuit never declared.
        h.app.on_output(Vec::new()).await.unwrap();
        assert!(h.app.on_output(vec![F::rand()]).await.is_err());
    }

    #[tokio::test]
    async fn other_fields_stop_at_the_shares() {
        let mut h = Harness::new(2);
        h.app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 2, "stark252").unwrap();
        h.deliver_inputs().await;
        let first = h.deliver_preprocessing().await;
        h.run(first).await;
        assert!(h.app.commitments().unwrap().is_none());
        h.app.on_output(Vec::new()).await.unwrap();
    }
}
