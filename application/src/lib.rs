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

use anyhow::Result;
use async_trait::async_trait;
use rand::random;

pub mod types;
pub use types::*;

pub mod anonymous_broadcast;
pub use anonymous_broadcast::AnonymousBroadcast;

/// Application-facing field element. Same underlying type as the engine's
/// `LargeField`; aliased here so application code reads in the field vocabulary.
pub type SmallField = fields::LargeField;

#[derive(Default, Clone)]
pub struct PreprocessingCounts{
    pub simd_mult: usize,
    pub depth: usize,

    pub net_route: Option<NetworkRoutingPreprocessingCounts>,
    pub nr_depth: usize,

    pub rand_bits: usize,

    pub output: usize
}

impl PreprocessingCounts{
    pub fn new_w_nr(simd_mult: usize, depth: usize, rand_bits: usize, output: usize)-> Self{
        PreprocessingCounts {
            simd_mult,
            depth,
            net_route: None,
            nr_depth: 0,
            rand_bits,
            output
        }
    }

    pub fn new(simd_mult: usize, depth: usize, nr: NetworkRoutingPreprocessingCounts, rand_bits: usize, nr_depth: usize, output: usize) -> Self{
        PreprocessingCounts {
            simd_mult,
            depth,
            net_route: Some(nr),
            rand_bits,
            nr_depth,
            output
        }
    }
}

// This needs to be in better detail.
#[derive(Clone)]
pub struct NetworkRoutingPreprocessingCounts{
    // The pattern is as follows. random_ is the first part, followed by the degree, followed by
    // first half values and second half values.
    // degree-3t/2 to 2t transformation
    pub random_2t_rand_zero: usize,
    pub random_2t_zero_rand_dual: usize,

    // select, reverse select, routing_l1, routing_l2, routing_l3
    pub random_2t_rand_rand: usize,
    // Reverse transformation to degree-3t/2 sharings
    pub random_3t_2_rand_x: usize,
    pub random_3t_2_rand_dual: usize,

    // Two selects and degree-2t to degree-3t/2 transformation
    pub random_3t_zero_zero: usize
}

impl NetworkRoutingPreprocessingCounts {
    pub fn new(
        random_2t_rand_zero: usize,
        random_2t_zero_rand_dual: usize,
        random_2t_rand_rand: usize,
        random_3t_2_rand_x: usize,
        random_3t_2_rand_dual: usize,
        random_3t_zero_zero: usize,
    ) -> Self {
        Self {
            random_2t_rand_zero,
            random_2t_zero_rand_dual,
            random_2t_rand_rand,
            random_3t_2_rand_x,
            random_3t_2_rand_dual,
            random_3t_zero_zero,
        }
    }
}

impl Default for NetworkRoutingPreprocessingCounts {
    fn default() -> Self {
        Self::new(0, 0, 0, 0, 0, 0)
    }
}

/// Actual preprocessed sharings for the network-routing pipeline. Field names
/// mirror `NetworkRoutingPreprocessingCounts`; `usize` counts there become
/// `Vec<SmallField>` here, and the two dual buckets become a pair of vectors
/// (`.0` = first-half encoding, `.1` = dual companion encoding).
#[derive(Clone, Default)]
pub struct NetworkRoutingPreprocessing {
    pub random_2t_rand_zero: Vec<SmallField>,
    pub random_2t_zero_rand_dual: (Vec<SmallField>, Vec<SmallField>),

    pub random_2t_rand_rand: Vec<SmallField>,

    pub random_3t_2_rand_x: Vec<SmallField>,
    pub random_3t_2_rand_dual: (Vec<SmallField>, Vec<SmallField>),

    pub random_3t_zero_zero: Vec<SmallField>,
}

impl NetworkRoutingPreprocessing {
    pub fn new(
        random_2t_rand_zero: Vec<SmallField>,
        random_2t_zero_rand_dual: (Vec<SmallField>, Vec<SmallField>),
        random_2t_rand_rand: Vec<SmallField>,
        random_3t_2_rand_x: Vec<SmallField>,
        random_3t_2_rand_dual: (Vec<SmallField>, Vec<SmallField>),
        random_3t_zero_zero: Vec<SmallField>,
    ) -> Self {
        Self {
            random_2t_rand_zero,
            random_2t_zero_rand_dual,
            random_2t_rand_rand,
            random_3t_2_rand_x,
            random_3t_2_rand_dual,
            random_3t_zero_zero,
        }
    }
}

/// Application-level interface for the MPC protocol.
///
/// The base MPC `Context<A>` drives the event loop and protocol phases
/// (preprocessing, multiplication, verification, routing). At phase
/// boundaries it calls into the application via these hooks.
///
/// Each application (anonymous broadcast, decision trees, etc.)
/// implements this trait to define application-specific behavior.
///
/// # Extending
/// Add new hook methods with default implementations so existing
/// applications continue to compile without changes.
#[async_trait]
pub trait Application: Send + 'static {
    /// Specifies how much preprocessing the circuit needs.
    /// The first value is the number of multiplication gates the protocol has (SIMD gates + Network Routing gates) and the second parameter is the number of random bits needed by the circuit.
    /// TODO: The application should return the number of bits and gates crudely and the MPC implementation should account for the packing factor.
    fn preprocessing_count(&mut self)-> PreprocessingCounts;

    /// Return the inputs this party wants to secret-share into the MPC.
    ///
    /// The outer `Vec` is one entry per sharing (one polynomial per batch);
    /// each inner `Vec` holds the secrets packed at the first-half evaluation
    /// points of that polynomial.
    ///
    /// Parties that are not acting as dealers for any input should return an
    /// empty `Vec`. The default implementation does exactly that, so
    /// applications without input sharing need not override it.
    async fn inputs(&mut self) -> Vec<Vec<SmallField>>;

    /// On terminating a party's input ACSS, this function is invoked to inform the application that there are shares available for use.
    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<SmallField>) -> DepthInput;
    /// Called after preprocessing completes (random sharings generated).
    ///
    /// Returns `Some(MultInput)` to kick off the first multiplication depth,
    /// or `None` if no circuit evaluation is needed.
    async fn on_preprocessing_complete(
        &mut self,
        rand_sharings_mult: Vec<SmallField>,
        rand_sharings_3t: Vec<SmallField>,
        rand_bit_sharings: Vec<SmallField>,
        network_routing_preprocessing: Option<NetworkRoutingPreprocessing>
    ) -> DepthInput;

    /// Called after all multiplications at a given depth complete.
    ///
    /// `depth` is the circuit depth that was just evaluated.
    /// `results` contains the output sharings from that depth.
    /// The results encode values in two different sets of possible locations.
    ///
    /// Returns `Some(MultInput)` for the next depth, or `None` when the
    /// circuit is complete.
    /// If it is the last depth, then it returns the set of output sharings for output reconstruction.
    async fn on_multiplication_complete(
        &mut self,
        depth: usize,
        results: (Vec<SmallField>, Vec<SmallField>),
    ) -> DepthInput;

    async fn on_network_routing_complete(
        &mut self,
        depth: usize,
        results: (Vec<SmallField>, Vec<SmallField>),
    ) -> DepthInput;

    /// Called after multiplication verification succeeds.
    async fn on_verification_complete(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when the protocol terminates (success or failure).
    async fn on_protocol_complete(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Default no-op application for running the base MPC protocol
/// without application-specific logic (e.g., benchmarking preprocessing).
pub struct DefaultApplication;

#[async_trait]
impl Application for DefaultApplication {

    fn preprocessing_count(&mut self)-> PreprocessingCounts{
        let counts = PreprocessingCounts::new_w_nr(10000,10, 10000, 100);
        log::info!(
            "DefaultApplication::preprocessing_count -> simd_mult={}, rand_bits={}, output={};",
            counts.simd_mult,
            counts.rand_bits,
            counts.output
        );
        counts
    }

    async fn input_sharing_termination(&mut self, _party: usize, _shares: Vec<SmallField>)-> DepthInput{
        return DepthInput { mult: None, network_routing: None };
    }

    async fn inputs(&mut self)-> Vec<Vec<SmallField>>{

        let k = 1000;
        (0..k)
            .map(|_| {
                (0..1)
                    .map(|_| SmallField::from(random::<u64>()))
                    .collect()
            })
            .collect()
    }

    async fn on_preprocessing_complete(
        &mut self,
        rand_sharings_mult: Vec<SmallField>,
        rand_sharings_3t: Vec<SmallField>,
        rand_bit_sharings: Vec<SmallField>,
        network_routing_preprocessing: Option<NetworkRoutingPreprocessing>,
    ) -> DepthInput {
        log::info!("DefaultApplication: preprocessing complete with {} {} {} counts",
            rand_sharings_mult.len(),
            rand_sharings_3t.len(),
            rand_bit_sharings.len()
        );
        match network_routing_preprocessing.as_ref() {
            Some(nr) => log::info!(
                "DefaultApplication: network routing sharings received -> \
                 random_2t_rand_zero={}, random_2t_zero_rand_dual=({}, {}), random_2t_rand_rand={}, \
                 random_3t_2_rand_x={}, random_3t_2_rand_dual=({}, {}), random_3t_zero_zero={}",
                nr.random_2t_rand_zero.len(),
                nr.random_2t_zero_rand_dual.0.len(), nr.random_2t_zero_rand_dual.1.len(),
                nr.random_2t_rand_rand.len(),
                nr.random_3t_2_rand_x.len(),
                nr.random_3t_2_rand_dual.0.len(), nr.random_3t_2_rand_dual.1.len(),
                nr.random_3t_zero_zero.len(),
            ),
            None => log::info!("DefaultApplication: no network routing preprocessing received"),
        }
        DepthInput { mult: None, network_routing: None }
    }

    async fn on_multiplication_complete(
        &mut self,
        depth: usize,
        _results: (Vec<SmallField>, Vec<SmallField>),
    ) -> DepthInput {
        log::info!("DefaultApplication: depth {} complete", depth);
        DepthInput { mult: None, network_routing: None }
    }

    async fn on_network_routing_complete(
        &mut self,
        depth: usize,
        _results: (Vec<SmallField>, Vec<SmallField>)
    )-> DepthInput{
        log::info!("DefaultApplication: network routing {} complete", depth);
        DepthInput { mult: None, network_routing: None }
    }
}
