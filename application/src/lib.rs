//! Application interface for the MPC engine.
//!
//! This crate defines the [`Application`] trait — the guideline any application
//! implements to run on top of the MPC engine — together with the minimal
//! protocol data-model types its hooks exchange (see [`types`]).
//!
//! To build an application, implement [`Application`] and return the
//! appropriate [`types::DepthInput`] from each hook. The crate ships one
//! concrete application, [`anonymous_broadcast::AnonymousBroadcast`], plus a
//! no-op [`DefaultApplication`] for running the base protocol on its own.
//!
//! # Division of labour
//!
//! The engine owns the protocol; the application owns the circuit. Concretely,
//! the application says *what to multiply* and the engine decides *how*: it
//! numbers the depths, draws the preprocessing each batch consumes, picks the
//! multiplication protocol, and verifies the resulting tuples. An application
//! that needed to know `2t+1` to portion its own preprocessing was doing the
//! engine's job, and doing it once per application.

use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;
use rand::random;

pub mod types;
pub use types::*;

pub mod anonymous_broadcast;
pub use anonymous_broadcast::AnonymousBroadcast;

/// How much preprocessing an application's circuit consumes.
///
/// Raw demand, in the application's own terms; the engine converts it into
/// ACSS/Sh2t batch sizes in `init_rand_sh`, adding what verification and the
/// common coin draw.
#[derive(Default, Clone, Debug)]
pub struct PreprocessingCounts {
    /// Number of multiplication gates at each depth, depth 1 first.
    ///
    /// A profile rather than a total, because the engine reserves each depth a
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
    pub gates_per_depth: Vec<usize>,
    /// Number of random bit sharings the circuit consumes.
    pub rand_bits: usize,
    /// Number of output wires to be reconstructed, each of which needs a mask.
    pub output: usize,
}

impl PreprocessingCounts {
    pub fn new(gates_per_depth: Vec<usize>, rand_bits: usize, output: usize) -> Self {
        Self {
            gates_per_depth,
            rand_bits,
            output,
        }
    }

    /// Total number of multiplication gates across every depth.
    pub fn mult_gates(&self) -> usize {
        self.gates_per_depth.iter().sum()
    }

    /// Number of multiplication depths — protocol rounds, not gate depth.
    /// Linear gates are evaluated locally and do not count.
    pub fn depth(&self) -> usize {
        self.gates_per_depth.len()
    }

    /// Number of gates depth `depth` declared, counting depths from 1.
    pub fn gates_at_depth(&self, depth: usize) -> Option<usize> {
        depth
            .checked_sub(1)
            .and_then(|index| self.gates_per_depth.get(index))
            .copied()
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

    /// Preprocessing is complete. `rand_bit_sharings` are the random bits the
    /// application asked for; the multiplication masks stay with the engine,
    /// which draws them per batch.
    async fn on_preprocessing_complete(
        &mut self,
        rand_bit_sharings: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;

    /// Every multiplication at `depth` has completed; `results` are the output
    /// sharings, in the order the operands were given.
    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;
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
        let counts = PreprocessingCounts::new(vec![1000; 10], 10000, 100);
        log::info!(
            "DefaultApplication::preprocessing_count -> mult_gates={}, depth={}, rand_bits={}, output={}",
            counts.mult_gates(),
            counts.depth(),
            counts.rand_bits,
            counts.output
        );
        counts
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

    async fn on_preprocessing_complete(
        &mut self,
        rand_bit_sharings: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!(
            "DefaultApplication: preprocessing complete with {} random bits",
            rand_bit_sharings.len()
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
