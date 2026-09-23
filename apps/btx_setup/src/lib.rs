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
//! # The indexed variant
//!
//! Policharla's indexed scheme (eprint 2026/760, §5, Fig. 2), selected by an
//! index radius `δ ≤ B`, shrinks a server's secret key from `B` shares to
//! `2δ − 1`: with `D = {−(δ−1), …, δ−1}` and secrets `τ, sk, β`, server `j`
//! holds `σ_j^d = ⟨sk·τ^d⟩_j` for `d ∈ D`, and the public material adds
//!
//!   - `A_i = g₁^{sk⁻¹·τ^i}` for `i ∈ [B]` — interpolated from the parties'
//!     `g₁^{⟨sk⁻¹·τ^i⟩_j}`;
//!   - `v_β = g₂^β` — interpolated from the parties' `g₂^{⟨β⟩_j}`;
//!   - `v_j^d = g₂^{β·σ_j^d}` — party `j` computes `σ_j^d · v_β` once `v_β`
//!     is public, so `β` is never multiplied in MPC.
//!
//! `h_i` and `ek` are as in the base scheme, so the power table runs to `2B`
//! either way. Negative powers of `τ` are avoided by sampling
//! `s = sk·τ^{−(δ−1)}` instead of `sk`, and *defining* `sk := s·τ^{δ−1}`:
//! with `s` and `τ` uniform and independent, so are `sk` and `τ`. Then
//! `σ^d = s·τ^{d+δ−1}` needs only the powers `τ⁰ … τ^{2δ−2}` the table
//! already has, and the one inversion left is `sk⁻¹`: reveal `u = r·sk` for a
//! random `⟨r⟩` — uniform, so it says nothing about `sk` — and set
//! `⟨sk⁻¹⟩ = u⁻¹·⟨r⟩` locally. `⟨r·sk⟩` is a multiplication output, whose
//! sharing polynomial the degree reduction makes fresh, which is what
//! [`Op::Reveal`]'s contract asks.
//!
//! # What the application computes
//!
//! Mapped onto the Planner's [`PlannerApplication`] API; nothing
//! Mersenne-specific, so it runs over BLS12-381:
//!
//!   - [`PlannerApplication::preprocessing_count`] asks for random sharings:
//!     `⟨τ⟩`, and for the indexed variant also `⟨s⟩ = ⟨σ^{−(δ−1)}⟩`, `⟨r⟩`,
//!     `⟨β⟩`. They come out of the preprocessing pool of random double sharings
//!     used for multiplication, extracted from whichever `n−t` dealers the ACS
//!     agreed on. At least `t+1` of those are honest, so each secret is uniform
//!     and unknown to any `t`-coalition — the guarantee the multiplication
//!     masks already rest on — and no party waits on any particular dealer. The
//!     circuit has no inputs of its own.
//!   - The circuit is a list of [`Wire`] products, laid out once in
//!     [`BtxSetup::new`] into op-depths: each `Mul` op-depth runs every product
//!     whose factors exist, and the one `Reveal` runs as soon as `⟨r·sk⟩` does.
//!     Powers are filled by doubling — `τ^i = τ^h · τ^{i−h}` with `h` the largest
//!     power of two below `i` — so the base scheme is `⌈log₂ 2B⌉` depths and
//!     `2B − 1` multiplications (for `B = 512`, ten depths of 1, 2, 4, …, 512
//!     gates). The indexed variant adds `2δ − 2` products for `σ`, two for
//!     `r·sk = (r·s)·τ^{δ−1}` (one at `δ = 1`), `B` for `sk⁻¹·τ^i`, and the
//!     reveal: one depth more than the base scheme for roughly `δ ≤ B/2`, two
//!     beyond.
//!   - The last depth returns [`OpDepthInput::Done`] with **no** output wires.
//!     Nothing but `r·sk` is ever reconstructed: the product of this circuit is
//!     the sharings the party still holds.
//!   - [`PlannerApplication::on_output`] is the engine saying the operations of
//!     the online phase have been verified and at least `t+1` honest parties
//!     terminated the protocol successfully. The party then outputs its shares,
//!     computes and prints its commitments to them (see [`commitments`]), and is
//!     done. Interpolating `h_i`, `ek`, `A_i` and `v_β` from any `t+1` parties'
//!     commitments is a public computation that needs no party's secrets.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use velox::{FieldElement, Op, OpDepthInput, OpParams, OpResult, OpType, PlannerApplication, PlannerCounts, ProtocolField, RandomWireShares};

pub mod commitments;
pub use commitments::{Commitments, IndexedCommitments};

/// The `--field` name under which lifting into BLS12-381's groups applies.
pub const BLS381: &str = "bls381";

/// A sharing the circuit computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wire {
    /// `⟨τ^i⟩`, `i = 1..=2B`. `τ¹` is a random sharing.
    Tau(usize),
    /// `σ^d = ⟨sk·τ^d⟩`, `d ∈ D`. `σ^{−(δ−1)}` is the random sharing `⟨s⟩`; the
    /// others are `σ^{−(δ−1)} · τ^{d+δ−1}`.
    Sigma(isize),
    /// `⟨r⟩`, the random mask of the inversion.
    Mask,
    /// `⟨r·s⟩`, one depth in, so that `r·sk` waits only on `τ^{δ−1}`.
    MaskedS,
    /// `⟨r·sk⟩ = ⟨r·s⟩ · τ^{δ−1}`, the one value the circuit reveals.
    MaskedSk,
    /// `⟨sk⁻¹⟩ = (r·sk)⁻¹ · ⟨r⟩`, local once `r·sk` is public.
    SkInv,
    /// `⟨sk⁻¹·τ^i⟩`, `i ∈ [B]`.
    A(usize),
    /// `⟨β⟩`, a random sharing.
    Beta,
}

/// `out = x · y`: one multiplication.
#[derive(Clone, Copy, Debug)]
struct Gate {
    out: Wire,
    x: Wire,
    y: Wire,
}

/// One op-depth of the circuit.
#[derive(Clone, Debug)]
enum Step {
    Mul(Vec<Gate>),
    /// Reveal [`Wire::MaskedSk`] and derive [`Wire::SkInv`] from it.
    RevealMaskedSk,
}

/// Party `j`'s outputs of the indexed variant, beyond the powers of `τ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedShares<F: ProtocolField> {
    /// `σ_j^d = ⟨sk·τ^d⟩_j` for `d = −(δ−1)..=δ−1` in order: the secret key.
    pub sigma: Vec<FieldElement<F>>,
    /// `⟨sk⁻¹·τ^i⟩_j` for `i = 1..=B`.
    pub a: Vec<FieldElement<F>>,
    /// `⟨β⟩_j`.
    pub beta: FieldElement<F>,
}

pub struct BtxSetup<F: ProtocolField> {
    pub num_nodes: usize,
    pub num_faults: usize,
    pub my_id: usize,
    /// `B`: the scheme's (maximum) batch size. The table runs to `2B`.
    pub batch_size: usize,
    /// `δ`, the index radius of the indexed variant; `None` for the base
    /// scheme.
    pub delta: Option<usize>,

    /// The field's name as the binary's `--field` spells it. Commitments are
    /// computed only over [`BLS381`]; other fields are for benchmarking the
    /// circuit and stop at printing the shares.
    field: String,

    /// The circuit, op-depth 1 first.
    steps: Vec<Step>,

    circuit_started: bool,

    /// This party's share of every wire computed so far.
    wires: HashMap<Wire, FieldElement<F>>,
    /// Highest depth whose results have been folded in. Depth `k+1` cannot be
    /// scheduled before depth `k` completes, so results arrive in order.
    depths_completed: usize,
}

impl<F: ProtocolField> BtxSetup<F> {
    pub fn new(
        num_nodes: usize,
        num_faults: usize,
        my_id: usize,
        batch_size: usize,
        delta: Option<usize>,
        field: &str,
    ) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch size must be at least 1");
        }
        if my_id >= num_nodes {
            bail!("party id {} out of range for {} nodes", my_id, num_nodes);
        }
        if let Some(delta) = delta {
            if delta == 0 || delta > batch_size {
                bail!("index radius delta = {} is outside 1..={}", delta, batch_size);
            }
        }

        // Every product, then laid out into op-depths.
        let mut gates: Vec<Gate> = (2..=2 * batch_size)
            .map(|i| {
                let h = 1usize << (usize::BITS - 1 - (i - 1).leading_zeros());
                Gate { out: Wire::Tau(i), x: Wire::Tau(h), y: Wire::Tau(i - h) }
            })
            .collect();
        let mut ready = HashSet::from([Wire::Tau(1)]);
        if let Some(delta) = delta {
            let delta = delta as isize;
            let s = Wire::Sigma(1 - delta);
            gates.extend((2 - delta..delta).map(|d| Gate {
                out: Wire::Sigma(d),
                x: s,
                y: Wire::Tau((d + delta - 1) as usize),
            }));
            if delta == 1 {
                gates.push(Gate { out: Wire::MaskedSk, x: Wire::Mask, y: s });
            } else {
                gates.push(Gate { out: Wire::MaskedS, x: Wire::Mask, y: s });
                gates.push(Gate { out: Wire::MaskedSk, x: Wire::MaskedS, y: Wire::Tau(delta as usize - 1) });
            }
            gates.extend((1..=batch_size).map(|i| Gate { out: Wire::A(i), x: Wire::SkInv, y: Wire::Tau(i) }));
            ready.extend([s, Wire::Mask, Wire::Beta]);
        }
        let mut steps = Vec::new();
        while !gates.is_empty() {
            if ready.contains(&Wire::MaskedSk) && !ready.contains(&Wire::SkInv) {
                steps.push(Step::RevealMaskedSk);
                ready.insert(Wire::SkInv);
                continue;
            }
            let (now, later): (Vec<Gate>, Vec<Gate>) =
                gates.into_iter().partition(|g| ready.contains(&g.x) && ready.contains(&g.y));
            ready.extend(now.iter().map(|g| g.out));
            steps.push(Step::Mul(now));
            gates = later;
        }

        Ok(Self {
            num_nodes,
            num_faults,
            my_id,
            batch_size,
            delta,
            field: field.to_string(),
            steps,
            circuit_started: false,
            wires: HashMap::new(),
            depths_completed: 0,
        })
    }

    /// Highest power the table holds: `2B`.
    pub fn max_power(&self) -> usize {
        2 * self.batch_size
    }

    /// Number of op-depths.
    pub fn depth(&self) -> usize {
        self.steps.len()
    }

    /// Each op-depth's op type and width, depth 1 first.
    pub fn ops(&self) -> Vec<OpParams> {
        self.steps
            .iter()
            .map(|step| match step {
                Step::Mul(gates) => OpParams::new(OpType::Mul, gates.len()),
                Step::RevealMaskedSk => OpParams::new(OpType::Reveal, 1),
            })
            .collect()
    }

    /// The random sharings the circuit starts from, in the order the engine
    /// hands them over.
    fn random_wires(&self) -> Vec<Wire> {
        match self.delta {
            None => vec![Wire::Tau(1)],
            Some(delta) => vec![Wire::Tau(1), Wire::Sigma(1 - delta as isize), Wire::Mask, Wire::Beta],
        }
    }

    /// This party's share of `wire`, once computed.
    fn wire(&self, wire: Wire) -> Result<FieldElement<F>> {
        self.wires.get(&wire).cloned().ok_or_else(|| anyhow!("{:?} is not computed yet", wire))
    }

    /// This party's shares of `τ¹ … τ^{2B}` in order, or `None` while the
    /// circuit is still running.
    pub fn shares(&self) -> Option<Vec<FieldElement<F>>> {
        if self.depths_completed < self.depth() {
            return None;
        }
        (1..=self.max_power()).map(|i| self.wires.get(&Wire::Tau(i)).cloned()).collect()
    }

    /// This party's indexed-variant outputs, or `None` for the base scheme or
    /// while the circuit is still running.
    pub fn indexed_shares(&self) -> Option<IndexedShares<F>> {
        let delta = self.delta? as isize;
        if self.depths_completed < self.depth() {
            return None;
        }
        Some(IndexedShares {
            sigma: (1 - delta..delta).map(|d| self.wires.get(&Wire::Sigma(d)).cloned()).collect::<Option<_>>()?,
            a: (1..=self.batch_size).map(|i| self.wires.get(&Wire::A(i)).cloned()).collect::<Option<_>>()?,
            beta: self.wires.get(&Wire::Beta).cloned()?,
        })
    }

    /// This party's commitments to its shares, once the circuit is done and
    /// if the field is the BLS12-381 scalar field — `None` otherwise, since the
    /// curve's points are multiplied by that field and no other.
    pub fn commitments(&self) -> Result<Option<Commitments>> {
        let Some(shares) = self.shares() else {
            bail!("the circuit is not complete: {} depths of {} done", self.depths_completed, self.depth());
        };
        Self::lift(&self.field, self.batch_size, &shares, self.indexed_shares())
    }

    /// The lifting behind [`commitments`](Self::commitments), as a function of
    /// its inputs so it can run off the engine's task.
    ///
    /// The shares are re-read through their bytes rather than cast, which is
    /// what makes this callable from the generic application: the bytes of a
    /// `bls381` element are the same whichever alias of the field it is
    /// typed as.
    fn lift(
        field: &str,
        batch_size: usize,
        shares: &[FieldElement<F>],
        indexed: Option<IndexedShares<F>>,
    ) -> Result<Option<Commitments>> {
        if field != BLS381 {
            return Ok(None);
        }
        let to_scalar = |s: &FieldElement<F>| {
            velox::fields::BLS12381ScalarField::from_bytes_be(&F::to_bytes_be(s))
                .map_err(|e| anyhow!("share is not a bls381 element: {:?}", e))
        };
        let scalars: Vec<commitments::Scalar> = shares.iter().map(to_scalar).collect::<Result<_>>()?;
        let indexed = match indexed {
            None => None,
            Some(ix) => Some((ix.a.iter().map(to_scalar).collect::<Result<Vec<_>>>()?, to_scalar(&ix.beta)?)),
        };
        Commitments::lift(batch_size, &scalars, indexed.as_ref().map(|(a, beta)| (a.as_slice(), beta))).map(Some)
    }

    /// Start the circuit exactly once, from the random sharings delivered.
    fn start_circuit(&mut self, sharings: Vec<FieldElement<F>>) -> Result<OpDepthInput<F>> {
        if self.circuit_started {
            log::debug!("BtxSetup: ignoring a repeated preprocessing handover");
            return Ok(OpDepthInput::Waiting);
        }
        let wires = self.random_wires();
        if sharings.len() != wires.len() {
            bail!("BtxSetup asked for {} random sharings and got {}", wires.len(), sharings.len());
        }
        self.circuit_started = true;
        self.wires.extend(wires.into_iter().zip(sharings));
        self.schedule_depth(1)
    }

    /// The op for depth `depth`, read from the wires computed so far. Past the
    /// last depth the circuit is done — with no output wires, since the shares
    /// are the product.
    fn schedule_depth(&self, depth: usize) -> Result<OpDepthInput<F>> {
        let Some(step) = self.steps.get(depth - 1) else {
            log::info!("BtxSetup: circuit complete after {} op-depths", self.depths_completed);
            return Ok(OpDepthInput::Done(Vec::new()));
        };
        let op = match step {
            Step::Mul(gates) => {
                let (x, y) = gates
                    .iter()
                    .map(|g| Ok((self.wire(g.x)?, self.wire(g.y)?)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                log::info!("BtxSetup: depth {} multiplies {} gates", depth, gates.len());
                Op::Mul { x, y }
            }
            Step::RevealMaskedSk => {
                log::info!("BtxSetup: depth {} reveals r*sk to invert sk", depth);
                Op::Reveal { x: vec![self.wire(Wire::MaskedSk)?] }
            }
        };
        OpDepthInput::op(depth, op)
    }
}

#[async_trait]
impl<F: ProtocolField> PlannerApplication<F> for BtxSetup<F> {
    fn preprocessing_count(&self) -> PlannerCounts {
        // The circuit reconstructs nothing, so there are no output wires; the
        // random sharings are the secrets and the mask, and there are no random
        // bits.
        let ops = self.ops();
        let multiplications: usize = ops.iter().filter(|p| p.op_type == OpType::Mul).map(|p| p.elements).sum();
        let counts = PlannerCounts::new(ops, 0).with_random_wires(0, self.random_wires().len());
        log::info!(
            "BtxSetup::preprocessing_count -> batch size {}, delta {:?}, {} multiplications over {} op-depths",
            self.batch_size,
            self.delta,
            multiplications,
            counts.depth()
        );
        counts
    }

    /// This circuit has no inputs, so the engine runs no input ACSS and this
    /// never fires; if it does, the shares belong to nothing here.
    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<FieldElement<F>>,
    ) -> Result<OpDepthInput<F>> {
        log::warn!(
            "BtxSetup: ignoring {} input sharings from party {}; this circuit takes no inputs",
            shares.len(),
            party
        );
        Ok(OpDepthInput::Waiting)
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        log::info!(
            "BtxSetup: preprocessing complete ({} random sharings, {} random bits)",
            wires.sharings.len(),
            wires.bits.len()
        );
        self.start_circuit(wires.sharings)
    }

    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        if depth <= self.depths_completed {
            log::debug!("BtxSetup: ignoring replayed completion of depth {}", depth);
            return Ok(OpDepthInput::Waiting);
        }
        if depth != self.depths_completed + 1 || depth > self.depth() {
            bail!(
                "depth {} completed before depth {}; this circuit's {} depths are sequential",
                depth,
                self.depths_completed + 1,
                self.depth()
            );
        }
        match self.steps[depth - 1].clone() {
            Step::Mul(gates) => {
                let results = result.shares()?;
                if results.len() != gates.len() {
                    bail!("depth {} returned {} results, expected {}", depth, results.len(), gates.len());
                }
                self.wires.extend(gates.iter().map(|g| g.out).zip(results));
            }
            Step::RevealMaskedSk => {
                let values = result.public()?;
                let [u] = values.as_slice() else {
                    bail!("depth {} revealed {} values, expected r*sk alone", depth, values.len());
                };
                let u_inv = u.inv().map_err(|_| anyhow!("r*sk revealed as 0, so r or sk is 0; rerun the setup"))?;
                self.wires.insert(Wire::SkInv, u_inv * self.wire(Wire::Mask)?);
            }
        }
        self.depths_completed = depth;
        self.schedule_depth(depth + 1)
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        if !outputs.is_empty() {
            bail!("BtxSetup declared no output wires but {} were reconstructed", outputs.len());
        }
        let Some(shares) = self.shares() else {
            bail!("the circuit is not complete: {} depths of {} done", self.depths_completed, self.depth());
        };
        let indexed = self.indexed_shares();
        let (j, b) = (self.my_id, self.batch_size);
        let hex = |s: &FieldElement<F>| commitments::hex(&F::to_bytes_be(s));
        match indexed {
            None => log::info!(
                "BtxSetup: run verified. Party {}'s shares <tau^i>_{} for i = 1..={} (indices 1..={} are sk_{}), as big-endian hex:",
                j, j, 2 * b, b, j
            ),
            Some(_) => log::info!(
                "BtxSetup: run verified. Party {}'s shares <tau^i>_{} for i = 1..={}, from which h_i and ek follow, as big-endian hex:",
                j, j, 2 * b
            ),
        }
        for (index, share) in shares.iter().enumerate() {
            log::info!("BtxSetup: <tau^{}>_{} = {}", index + 1, j, hex(share));
        }
        if let (Some(ix), Some(delta)) = (&indexed, self.delta) {
            log::info!("BtxSetup: party {}'s secret key sk_{} = {{sigma_{}^d = <sk*tau^d>_{}}} for |d| < {}:", j, j, j, j, delta);
            for (d, share) in (1 - delta as isize..).zip(&ix.sigma) {
                log::info!("BtxSetup: sigma_{}^{} = {}", j, d, hex(share));
            }
            for (index, share) in ix.a.iter().enumerate() {
                log::info!("BtxSetup: <sk^-1*tau^{}>_{} = {}", index + 1, j, hex(share));
            }
            log::info!("BtxSetup: <beta>_{} = {}", j, hex(&ix.beta));
        }

        let Some(c) = velox::fields::rayon_async({
            let field = self.field.clone();
            move || Self::lift(&field, b, &shares, indexed)
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
            match c.indexed {
                None => log::info!("BtxSetup: v_{}^{} = g2^(<tau^{}>_{}) = {}", j, index + 1, index + 1, j, commitments::g2_hex(p)),
                Some(_) => log::info!("BtxSetup: g2^(<tau^{}>_{}) = {}", index + 1, j, commitments::g2_hex(p)),
            }
        }
        for (index, p) in c.high.iter().enumerate() {
            let i = b + 2 + index;
            log::info!("BtxSetup: g2^(<tau^{}>_{}) = {}", i, j, commitments::g2_hex(p));
        }
        log::info!(
            "BtxSetup: punctured power in GT only: e(g1^(<tau^{}>_{}), g2) = {}",
            b + 1, j, commitments::gt_hex(&c.middle)
        );
        if let Some(ix) = &c.indexed {
            log::info!("BtxSetup: party {}'s commitments g1^(<sk^-1*tau^i>_{}) for i in [B], interpolating to A_i, compressed G1 as hex:", j, j);
            for (index, p) in ix.a.iter().enumerate() {
                log::info!("BtxSetup: g1^(<sk^-1*tau^{}>_{}) = {}", index + 1, j, commitments::g1_hex(p));
            }
            log::info!("BtxSetup: g2^(<beta>_{}) = {}", j, commitments::g2_hex(&ix.beta));
            log::info!(
                "BtxSetup: once v_beta = g2^beta is interpolated, party {} publishes v_{}^d = sigma_{}^d * v_beta for |d| < {}",
                j, j, j, self.delta.unwrap_or_default()
            );
        }
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
    /// operations on the secrets, so feeding each secret itself where the
    /// engine would feed its sharing, multiplying batches in the clear and
    /// revealing a value as itself makes every wire directly comparable to
    /// what it should hold.
    struct Harness {
        app: BtxSetup<F>,
        /// The random sharings handed over: `τ`, then for the indexed variant
        /// `s = sk·τ^{−(δ−1)}`, `r`, `β`.
        secrets: Vec<FieldElement<F>>,
        /// Each op-depth the application scheduled.
        ops: Vec<OpParams>,
    }

    impl Harness {
        fn new(batch_size: usize, delta: Option<usize>) -> Self {
            let app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, batch_size, delta, BLS381).unwrap();
            let secrets = (0..app.random_wires().len()).map(|_| F::rand()).collect();
            Self { app, secrets, ops: Vec::new() }
        }

        fn tau(&self) -> FieldElement<F> {
            self.secrets[0].clone()
        }

        /// The engine's handover: the random sharings asked for.
        async fn deliver_preprocessing(&mut self) -> OpDepthInput<F> {
            self.app
                .on_preprocessing_complete(RandomWireShares::new(Vec::new(), self.secrets.clone()))
                .await
                .unwrap()
        }

        /// Run every depth the application schedules; returns the output wires
        /// it hands back, which must be none.
        async fn run(&mut self, mut depth_input: OpDepthInput<F>) -> Vec<FieldElement<F>> {
            loop {
                let (depth, result) = match depth_input {
                    OpDepthInput::Done(outputs) => return outputs,
                    OpDepthInput::Waiting => panic!("the application stalled with nothing scheduled"),
                    OpDepthInput::Op { depth, op: Op::Mul { x, y } } => {
                        self.ops.push(OpParams::new(OpType::Mul, x.len()));
                        (depth, OpResult::Shares(x.into_iter().zip(y).map(|(x, y)| x * y).collect()))
                    }
                    OpDepthInput::Op { depth, op: Op::Reveal { x } } => {
                        self.ops.push(OpParams::new(OpType::Reveal, x.len()));
                        (depth, OpResult::Public(x))
                    }
                    other => panic!("the setup only multiplies and reveals, got {:?}", other),
                };
                depth_input = self.app.on_depth_complete(depth, result).await.unwrap();
            }
        }

        /// `τ¹ … τ^{max}`.
        fn powers_of_tau(&self, max: usize) -> Vec<FieldElement<F>> {
            let tau = self.tau();
            let mut acc = FieldElement::<F>::one();
            (0..max).map(|_| { acc = &acc * &tau; acc.clone() }).collect()
        }
    }

    fn mul_ops(gates: &[usize]) -> Vec<OpParams> {
        gates.iter().map(|g| OpParams::new(OpType::Mul, *g)).collect()
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
            let app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, batch_size, None, BLS381).unwrap();
            assert_eq!(app.ops(), mul_ops(&gates), "B = {}", batch_size);
            assert_eq!(gates.iter().sum::<usize>(), 2 * batch_size - 1, "B = {}", batch_size);
            assert_eq!(app.depth(), ((2 * batch_size) as f64).log2().ceil() as usize, "B = {}", batch_size);
            let counts = app.preprocessing_count();
            assert_eq!(counts.ops, app.ops());
            assert_eq!(counts.output, 0);
        }
    }

    #[test]
    fn indexed_schedule_adds_one_reveal_and_at_most_two_depths() {
        for (batch_size, delta) in [(1, 1), (2, 1), (2, 2), (4, 3), (16, 1), (16, 4), (16, 9), (16, 16), (100, 37)] {
            let app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, batch_size, Some(delta), BLS381).unwrap();
            let ops = app.ops();
            let reveals: Vec<&OpParams> = ops.iter().filter(|p| p.op_type == OpType::Reveal).collect();
            assert_eq!(reveals, vec![&OpParams::new(OpType::Reveal, 1)], "B = {}, delta = {}", batch_size, delta);
            let multiplications: usize = ops.iter().filter(|p| p.op_type == OpType::Mul).map(|p| p.elements).sum();
            let masking = if delta == 1 { 1 } else { 2 };
            assert_eq!(multiplications, (2 * batch_size - 1) + (2 * delta - 2) + masking + batch_size);
            let base = ((2 * batch_size) as f64).log2().ceil() as usize;
            assert!(app.depth() <= base + 2, "B = {}, delta = {}: {} depths", batch_size, delta, app.depth());
        }
        // Small δ: the reveal slots in among the doublings and the A_i ride the
        // last one, so the variant costs just the reveal's depth.
        let app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 16, Some(2), BLS381).unwrap();
        assert_eq!(app.depth(), 6);
    }

    #[test]
    fn delta_out_of_range_is_an_error() {
        assert!(BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 4, Some(0), BLS381).is_err());
        assert!(BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 4, Some(5), BLS381).is_err());
        assert!(BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 4, Some(4), BLS381).is_ok());
    }

    #[tokio::test]
    async fn table_holds_the_powers_of_tau() {
        for batch_size in [1, 2, 3, 16] {
            let mut h = Harness::new(batch_size, None);
            let first = h.deliver_preprocessing().await;
            assert!(matches!(first, OpDepthInput::Op { depth: 1, op: Op::Mul { .. } }), "B = {}: starts on the handover", batch_size);
            let outputs = h.run(first).await;
            assert!(outputs.is_empty(), "B = {}: the circuit declares no output wires", batch_size);
            assert_eq!(h.ops, h.app.ops(), "B = {}", batch_size);
            assert_eq!(h.app.shares().unwrap(), h.powers_of_tau(2 * batch_size), "B = {}", batch_size);
            assert!(h.app.indexed_shares().is_none());
        }
    }

    #[tokio::test]
    async fn indexed_outputs_are_the_schemes_secrets() {
        for (batch_size, delta) in [(1, 1), (3, 2), (8, 3), (16, 16)] {
            let mut h = Harness::new(batch_size, Some(delta));
            let first = h.deliver_preprocessing().await;
            assert!(h.run(first).await.is_empty());
            assert_eq!(h.ops, h.app.ops(), "B = {}, delta = {}", batch_size, delta);
            assert_eq!(h.app.shares().unwrap(), h.powers_of_tau(2 * batch_size));

            // sk = s·τ^{δ−1}; σ^d = sk·τ^d, a_i = sk⁻¹·τ^i.
            let tau = h.tau();
            let tau_inv = tau.inv().unwrap();
            let sk = &h.secrets[1] * tau.pow(delta as u64 - 1);
            let ix = h.app.indexed_shares().unwrap();
            assert_eq!(ix.sigma.len(), 2 * delta - 1);
            for (d, sigma) in (1 - delta as isize..).zip(&ix.sigma) {
                let tau_d = if d >= 0 { tau.pow(d as u64) } else { tau_inv.pow((-d) as u64) };
                assert_eq!(*sigma, &sk * tau_d, "B = {}, delta = {}: sigma^{}", batch_size, delta, d);
            }
            let sk_inv = sk.inv().unwrap();
            assert_eq!(ix.a, h.powers_of_tau(batch_size).iter().map(|t| &sk_inv * t).collect::<Vec<_>>());
            assert_eq!(ix.beta, h.secrets[3]);
        }
    }

    #[tokio::test]
    async fn wrong_number_of_random_sharings_is_an_error() {
        for (delta, sharings) in [(None, 0), (None, 2), (Some(2), 1), (Some(2), 5)] {
            let mut h = Harness::new(4, delta);
            let wires = RandomWireShares::new(Vec::new(), (0..sharings).map(|_| F::rand()).collect());
            assert!(h.app.on_preprocessing_complete(wires).await.is_err(), "{:?}: {} sharings", delta, sharings);
        }
        for (delta, sharings) in [(None, 1), (Some(2), 4)] {
            let counts = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 4, delta, BLS381).unwrap().preprocessing_count();
            assert_eq!((counts.rand_bits, counts.sharings), (0, sharings));
        }
    }

    #[tokio::test]
    async fn replayed_and_stray_events_are_ignored() {
        let mut h = Harness::new(4, None);
        let first = h.deliver_preprocessing().await;
        let OpDepthInput::Op { depth, op: Op::Mul { x, y } } = first else { panic!("depth 1 not scheduled") };
        let results: Vec<FieldElement<F>> = x.iter().zip(y.iter()).map(|(x, y)| x * y).collect();
        let second = h.app.on_depth_complete(depth, OpResult::Shares(results.clone())).await.unwrap();
        assert!(matches!(second, OpDepthInput::Op { depth: 2, .. }));

        // A replayed termination of depth 1 schedules nothing.
        assert!(h.app.on_depth_complete(depth, OpResult::Shares(results)).await.unwrap().is_waiting());
        // An input sharing belongs to nothing in this circuit.
        assert!(h.app.input_sharing_termination(1, vec![F::rand()]).await.unwrap().is_waiting());
        // Preprocessing arriving twice does not restart the circuit.
        assert!(h.deliver_preprocessing().await.is_waiting());

        h.run(second).await;
        assert_eq!(h.app.shares().unwrap(), h.powers_of_tau(8));
    }

    #[tokio::test]
    async fn wrong_batch_shape_is_an_error() {
        let mut h = Harness::new(4, None);
        let OpDepthInput::Op { depth, op: Op::Mul { x, .. } } = h.deliver_preprocessing().await else {
            panic!("depth 1 not scheduled")
        };
        // Depth 1 has one gate; hand back two results.
        let mut results = x.clone();
        results.push(F::rand());
        assert!(h.app.on_depth_complete(depth, OpResult::Shares(results)).await.is_err());
        // Depth 3 before depth 2 is not this circuit.
        assert!(h.app.on_depth_complete(3, OpResult::Shares(vec![F::rand()])).await.is_err());
        // A public result where shares belong is not this circuit either.
        assert!(h.app.on_depth_complete(1, OpResult::Public(vec![F::rand()])).await.is_err());
    }

    #[tokio::test]
    async fn a_zero_reveal_is_an_error() {
        let mut h = Harness::new(2, Some(1));
        let OpDepthInput::Op { depth: 1, op: Op::Mul { x, y } } = h.deliver_preprocessing().await else {
            panic!("depth 1 not scheduled")
        };
        let products = x.into_iter().zip(y).map(|(x, y)| x * y).collect();
        let reveal = h.app.on_depth_complete(1, OpResult::Shares(products)).await.unwrap();
        assert!(matches!(reveal, OpDepthInput::Op { depth: 2, op: Op::Reveal { .. } }));
        assert!(h.app.on_depth_complete(2, OpResult::Public(vec![FieldElement::zero()])).await.is_err());
    }

    #[tokio::test]
    async fn commitments_are_the_table_in_the_exponent() {
        use lambdaworks_math::cyclic_group::IsGroup;
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
        let g2 = lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve::generator();

        let mut h = Harness::new(2, None);
        assert!(h.app.commitments().is_err(), "no commitments before the table is full");
        let first = h.deliver_preprocessing().await;
        h.run(first).await;

        let c = h.app.commitments().unwrap().expect("bls381 lifts");
        let powers = h.powers_of_tau(4);
        assert_eq!(c.low.len(), 2);
        assert_eq!(c.high.len(), 1);
        assert_eq!(c.low[0], g2.operate_with_self(powers[0].representative()));
        assert_eq!(c.low[1], g2.operate_with_self(powers[1].representative()));
        assert_eq!(c.high[0], g2.operate_with_self(powers[3].representative()));
        assert_eq!(c.middle, commitments::lift_to_gt(&powers[2]).unwrap());
        assert!(c.indexed.is_none());

        // on_output prints and lifts; it must accept the empty output and
        // reject wires this circuit never declared.
        h.app.on_output(Vec::new()).await.unwrap();
        assert!(h.app.on_output(vec![F::rand()]).await.is_err());
    }

    #[tokio::test]
    async fn indexed_commitments_lift_a_and_beta() {
        use lambdaworks_math::cyclic_group::IsGroup;
        use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, twist::BLS12381TwistCurve};
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

        let mut h = Harness::new(3, Some(2));
        let first = h.deliver_preprocessing().await;
        h.run(first).await;
        let ix = h.app.indexed_shares().unwrap();
        let c = h.app.commitments().unwrap().expect("bls381 lifts").indexed.expect("indexed commitments");
        let g1 = BLS12381Curve::generator();
        assert_eq!(c.a, ix.a.iter().map(|a| g1.operate_with_self(a.representative())).collect::<Vec<_>>());
        assert_eq!(c.beta, BLS12381TwistCurve::generator().operate_with_self(ix.beta.representative()));
        h.app.on_output(Vec::new()).await.unwrap();
    }

    /// The setup hosted by the Planner over BLS12-381, driven by a plaintext
    /// engine: the random sharings are passed through, each multiply depth is
    /// one engine multiplication, each reveal one engine reveal.
    #[tokio::test]
    async fn the_setup_runs_through_the_planner_over_bls381() {
        use velox::{Application, DepthInput, Planner};
        for delta in [None, Some(3)] {
            let h = Harness::new(5, delta);
            let mut planner = Planner::new(BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 5, delta, BLS381).unwrap()).unwrap();
            assert_eq!(planner.random_wires(), velox::RandomWires::new(0, h.secrets.len()));
            let mut next = planner.on_preprocessing_complete(RandomWireShares::new(Vec::new(), h.secrets.clone())).await.unwrap();
            let mut depths = 0;
            let outputs = loop {
                next = match next {
                    DepthInput::Done(outputs) => break outputs,
                    DepthInput::Multiply { depth, x, y } => {
                        depths += 1;
                        let products = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect();
                        planner.on_depth_complete(depth, products).await.unwrap()
                    }
                    DepthInput::Reveal { depth, values } => {
                        depths += 1;
                        planner.on_reveal_complete(depth, values).await.unwrap()
                    }
                    other => panic!("the setup only multiplies and reveals, got {:?}", other),
                };
            };
            assert!(outputs.is_empty());
            assert_eq!(depths, planner.app().depth(), "{:?}", delta);
            assert_eq!(planner.app().shares().unwrap(), h.powers_of_tau(10));
            assert_eq!(planner.app().indexed_shares().is_some(), delta.is_some());
        }
    }

    #[tokio::test]
    async fn other_fields_stop_at_the_shares() {
        let mut h = Harness::new(2, Some(2));
        h.app = BtxSetup::<F>::new(NUM_NODES, NUM_FAULTS, 0, 2, Some(2), "stark252").unwrap();
        let first = h.deliver_preprocessing().await;
        h.run(first).await;
        assert!(h.app.commitments().unwrap().is_none());
        h.app.on_output(Vec::new()).await.unwrap();
    }
}
