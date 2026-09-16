//! Application interface for the MPC engine.
//!
//! This crate defines the [`Application`] trait — the guideline any application
//! implements to run on top of the MPC engine — together with the minimal
//! protocol data-model types its hooks exchange (see [`types`]).
//!
//! To build an application, implement [`Application`] and return the
//! appropriate [`types::DepthInput`] from each hook. Concrete applications live
//! in their own crates under `apps/` and depend on this one; it carries only the
//! interface between them and the engine, plus a no-op [`DefaultApplication`]
//! for running the base protocol on its own.
//!
//! # Division of labour
//!
//! The engine owns the protocol; the application owns the circuit. Concretely,
//! the application says *what to multiply* and *what to reveal* and the engine
//! decides *how*: it numbers the depths, draws the preprocessing each batch
//! consumes, picks the multiplication protocol, runs the public
//! reconstruction, and verifies the resulting tuples and revealed values. An
//! application that needed to know `2t+1` to portion its own preprocessing was
//! doing the engine's job, and doing it once per application.
//!
//! The engine's vocabulary is deliberately small — a multiplication batch, a
//! reveal batch, a masked multiplication batch — so that richer operations
//! (comparison, truncation) are built *above* this trait, as sequences of
//! those batches, rather than as engine features.

use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;
use rand::random;

pub mod types;
pub use types::*;

/// What the engine consumes evaluating an application's circuit.
///
/// Raw demand, in the application's own terms; the engine converts it into
/// ACSS/Sh2t batch sizes in `init_rand_sh`, adding what verification and the
/// common coin draw. Everything here is spent *by the engine* — masks and
/// zero-sharings per multiplication, masks per output wire — and the
/// application never sees it. What the application consumes itself, as
/// wires, is declared separately in [`RandomWires`].
#[derive(Default, Clone, Debug)]
pub struct PreprocessingCounts {
    /// Number of multiplication gates at each depth, depth 1 first.
    ///
    /// A profile rather than a total, because the engine reserves preprocessing material 
    /// for each depth as a
    /// *fixed slice* of the preprocessing pool, computed from this vector. Every
    /// party derives the same table from the same circuit, so depth `d` binds to
    /// the same random sharings everywhere however the depths happen to be
    /// scheduled locally — out of order, or fast-forwarded past a depth whose
    /// reconstruction already arrived. Handing out masks in the order batches
    /// happen to be scheduled would make that binding depend on local timing,
    /// and two parties masking a gate differently do not reconstruct.
    ///
    /// A depth may run *fewer* gates than it declares — it then uses a prefix of
    /// its slice, which is still the same prefix everywhere — but never more.
    ///
    /// A depth that carries a [`Reveal`](DepthInput::Reveal) declares `0`
    /// here: a reveal consumes no preprocessing.
    pub gates_per_depth: Vec<usize>,
    /// Number of [`MaskedMultiply`](DepthInput::MaskedMultiply) gates at each
    /// depth, depth 1 first, under the same fixed-slice rule as
    /// [`gates_per_depth`](Self::gates_per_depth). Shorter than that vector
    /// means the remaining depths have none; empty (the default) means the
    /// circuit has no masked multiplications at all.
    ///
    /// A masked gate is charged a share of the `2t` zero-sharings a batch
    /// draws but no mask, since the application supplies the mask. A depth
    /// declares gates of one kind only — one batch per depth, see
    /// [`DepthInput`].
    pub masked_gates_per_depth: Vec<usize>,
    /// Number of output wires to be reconstructed, each of which needs a mask.
    pub output: usize,
}

impl PreprocessingCounts {
    pub fn new(gates_per_depth: Vec<usize>, output: usize) -> Self {
        Self {
            gates_per_depth,
            masked_gates_per_depth: Vec::new(),
            output,
        }
    }

    /// Declare masked multiplication gates per depth; see
    /// [`masked_gates_per_depth`](Self::masked_gates_per_depth).
    pub fn with_masked_gates(mut self, masked_gates_per_depth: Vec<usize>) -> Self {
        self.masked_gates_per_depth = masked_gates_per_depth;
        self
    }

    /// Total number of multiplication gates across every depth, plain and
    /// masked: every one of them produces a tuple the verification phase
    /// checks, which is what this total sizes.
    pub fn mult_gates(&self) -> usize {
        self.plain_gates() + self.masked_gates_per_depth.iter().sum::<usize>()
    }

    /// Gates that draw a mask from the engine's pool — the plain ones.
    pub fn plain_gates(&self) -> usize {
        self.gates_per_depth.iter().sum()
    }

    /// Number of multiplication depths — protocol rounds, not gate depth.
    /// Linear gates are evaluated locally and do not count.
    pub fn depth(&self) -> usize {
        self.gates_per_depth.len().max(self.masked_gates_per_depth.len())
    }

    /// Number of plain gates depth `depth` declared, counting depths from 1;
    /// `None` past the declared depths.
    pub fn gates_at_depth(&self, depth: usize) -> Option<usize> {
        Self::at_depth(&self.gates_per_depth, self.depth(), depth)
    }

    /// Number of masked gates depth `depth` declared, counting depths from 1;
    /// `Some(0)` for a declared depth the shorter vector does not reach.
    pub fn masked_gates_at_depth(&self, depth: usize) -> Option<usize> {
        Self::at_depth(&self.masked_gates_per_depth, self.depth(), depth)
    }

    fn at_depth(profile: &[usize], depths: usize, depth: usize) -> Option<usize> {
        let index = depth.checked_sub(1)?;
        if index >= depths {
            return None;
        }
        Some(profile.get(index).copied().unwrap_or(0))
    }
}

/// Random wires an application's circuit consumes.
///
/// These are values the application reads *as sharings* — a random bit to
/// switch on, a random secret to raise to powers — as opposed to the
/// material in [`PreprocessingCounts`] that the engine spends on the
/// application's behalf and never hands over. Both are produced in the
/// preprocessing phase from the same ACS-agreed dealers, and both come with
/// the same guarantee: degree-`t`, uniformly random, unknown to any
/// `t`-coalition, and available without waiting on any particular dealer.
///
/// The engine delivers them in [`RandomWireShares`], field for field.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct RandomWires {
    /// Random bits, delivered as sharings of `±1`.
    pub bits: usize,
    /// Sharings of uniformly random field elements.
    pub sharings: usize,
}

impl RandomWires {
    pub fn new(bits: usize, sharings: usize) -> Self {
        Self { bits, sharings }
    }
}

/// Application-level interface for the MPC protocol.
///
/// The base MPC `Context<A>` drives the event loop and protocol phases
/// (preprocessing, multiplication, verification, output reconstruction). At
/// phase boundaries it calls into the application via these hooks, and acts on
/// the [`DepthInput`] each one returns.
///
/// Each application (anonymous broadcast, arithmetic circuits, decision trees,
/// …) implements this trait to define application-specific behavior.
///
/// # Extending
/// Add new hook methods with default implementations so existing applications
/// continue to compile without changes. Do not add a hook the engine does not
/// call: three such hooks accumulated here before, and every application had to
/// implement them to satisfy the trait even though none could ever fire.
#[async_trait]
pub trait Application<F: ProtocolField>: Send + 'static {
    /// How much preprocessing the circuit needs.
    ///
    /// Takes `&self` and must be pure, and must return the same answer at every
    /// party: the engine reads it once, at the start of preprocessing, and lays
    /// out the per-depth preprocessing slices from it. Two parties disagreeing
    /// on `gates_per_depth` would bind different random sharings to the same
    /// gate.
    fn preprocessing_count(&self) -> PreprocessingCounts;

    /// The random wires the circuit consumes. Same contract as
    /// [`preprocessing_count`](Application::preprocessing_count): pure, read
    /// once at the start of preprocessing, the same answer at every party.
    /// The default asks for none, so a circuit without random wires need not
    /// mention them.
    fn random_wires(&self) -> RandomWires {
        RandomWires::default()
    }

    /// The secrets this party contributes to the circuit's input wires, in wire
    /// order.
    ///
    /// Parties that deal no inputs return an empty `Vec`; the default
    /// implementation does exactly that, so applications without input sharing
    /// need not override it.
    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        Vec::new()
    }

    /// A party's input ACSS has terminated, so its `shares` are now available.
    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;

    /// Preprocessing is complete. `wires` are the random wires the application
    /// asked for in [`random_wires`](Application::random_wires); the
    /// multiplication masks stay with the engine, which draws them per batch.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>>;

    /// Every multiplication at `depth` has completed; `results` are the output
    /// sharings, in the order the operands were given.
    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;

    /// A [`Reveal`](DepthInput::Reveal) or
    /// [`MaskedMultiply`](DepthInput::MaskedMultiply) batch at `depth` has
    /// completed; `values` are the reconstructed field elements, in operand
    /// order. They are public — every party holds the same vector.
    ///
    /// The default is an error, not [`DepthInput::Waiting`]: an application
    /// that scheduled a reveal and does not handle its result has a bug, and
    /// `Waiting` would turn that into the silent hang the `Err` path exists to
    /// surface. Applications that never reveal need not override it.
    async fn on_reveal_complete(
        &mut self,
        depth: usize,
        values: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        let _ = values;
        anyhow::bail!(
            "a reveal batch completed at depth {} but this application does not implement on_reveal_complete",
            depth
        )
    }

    /// The protocol has terminated: every multiplication the circuit ran has
    /// been verified, and the parties have agreed that at least `t+1` of them
    /// reconstructed the output. `outputs` are the reconstructed values of the
    /// wires the application returned in [`DepthInput::Done`], in that order —
    /// empty if it returned none.
    ///
    /// This is the only point at which an application knows its run was
    /// verified *and* agreed on, so it is where an application whose real
    /// product is the sharings it still holds — a key-generation setup, say —
    /// persists them. Nothing the application holds should be treated as
    /// final before this fires. The default does nothing, so applications
    /// that only care about the reconstructed output need not override it.
    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let _ = outputs;
        Ok(())
    }
}

/// Default no-op application for running the base MPC protocol
/// without application-specific logic (e.g. benchmarking preprocessing).
pub struct DefaultApplication<F: ProtocolField>(PhantomData<F>);

impl<F: ProtocolField> Default for DefaultApplication<F> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<F: ProtocolField> DefaultApplication<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for DefaultApplication<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        let counts = PreprocessingCounts::new(vec![1000; 10], 100);
        log::info!(
            "DefaultApplication::preprocessing_count -> mult_gates={}, depth={}, output={}",
            counts.mult_gates(),
            counts.depth(),
            counts.output
        );
        counts
    }

    fn random_wires(&self) -> RandomWires {
        RandomWires::new(10000, 0)
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        (0..1000)
            .map(|_| FieldElement::<F>::from(random::<u64>()))
            .collect()
    }

    async fn input_sharing_termination(
        &mut self,
        _party: usize,
        _shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        Ok(DepthInput::Waiting)
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        log::info!(
            "DefaultApplication: preprocessing complete with {} random bits and {} random sharings",
            wires.bits.len(),
            wires.sharings.len()
        );
        Ok(DepthInput::Waiting)
    }

    async fn on_depth_complete(
        &mut self,
        depth: usize,
        _results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!("DefaultApplication: depth {} complete", depth);
        Ok(DepthInput::Waiting)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Masked gates count towards the tuples verification sizes, not towards
    /// the masks the pool provides.
    #[test]
    fn masked_gates_are_tuples_but_not_masks() {
        let counts = PreprocessingCounts::new(vec![3, 0, 5], 0).with_masked_gates(vec![0, 4]);
        assert_eq!(counts.plain_gates(), 8);
        assert_eq!(counts.mult_gates(), 12);
        assert_eq!(PreprocessingCounts::new(vec![3], 0).mult_gates(), 3);
    }

    /// The depth count covers both profiles, and each lookup fills the shorter
    /// profile with zeros up to that count.
    #[test]
    fn depth_spans_both_profiles() {
        let counts = PreprocessingCounts::new(vec![3, 0], 0).with_masked_gates(vec![0, 4, 6]);
        assert_eq!(counts.depth(), 3);
        assert_eq!(counts.gates_at_depth(1), Some(3));
        assert_eq!(counts.gates_at_depth(3), Some(0), "declared by the masked profile only");
        assert_eq!(counts.masked_gates_at_depth(1), Some(0));
        assert_eq!(counts.masked_gates_at_depth(3), Some(6));
        assert_eq!(counts.gates_at_depth(4), None);
        assert_eq!(counts.masked_gates_at_depth(4), None);
        assert_eq!(counts.gates_at_depth(0), None);
        assert_eq!(counts.masked_gates_at_depth(0), None);
    }

    /// Without masked gates nothing about the old accessors changes.
    #[test]
    fn plain_profile_is_unchanged() {
        let counts = PreprocessingCounts::new(vec![1, 2], 7);
        assert_eq!(counts.depth(), 2);
        assert_eq!(counts.gates_at_depth(2), Some(2));
        assert_eq!(counts.gates_at_depth(3), None);
        assert_eq!(counts.masked_gates_at_depth(2), Some(0));
        assert_eq!(counts.output, 7);
    }
}
