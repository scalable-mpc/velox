//! Per-batch state of a public reconstruction. One of these hangs off each
//! depth's `SingleDepthState`, so a depth's reconstruction is found where the
//! rest of the depth's state is.
//!
//! Memory discipline, in one place for every consumer:
//! - the raw per-party share buffers are taken into the interpolation that
//!   consumes them the moment the threshold is crossed, not at termination;
//! - the claim flags (`l1_started`, `l2_started`) are set *before* the yielded
//!   interpolation, so a message handled during the job returns early instead
//!   of pushing into a buffer that is gone;
//! - the reconstructed values are moved out at termination, not cloned;
//! - what stays after termination is bookkeeping only — the flags, the
//!   counts and the hash votes — so a late message is still deduplicated.

use std::collections::HashSet;

use crypto::hash::Hash;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;
use types::Replica;

use super::config::{ReconConfig, ReconKind};

pub struct ReconState<F: ProtocolField> {
    /// Set by this party's own `init`; a peer's message may create the state
    /// first. Interpolation and termination wait for it — the degree, the
    /// padding and the consumer are only known once this party has started.
    pub kind: Option<ReconKind>,
    pub config: Option<ReconConfig>,
    /// Zero shares appended to round the batch up to whole chunks; that many
    /// values are trimmed at the end. Known to the owner only.
    pub padding: Option<usize>,
    /// Chunks of `2t+1` values the batch was split into. Set by whichever of
    /// own `init` or a peer's message comes first.
    pub num_chunks: usize,

    /// L1: senders' evaluation points, and per chunk their shares of this
    /// party's point on the chunk polynomial.
    pub l1_shares: (Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>),
    pub recv_l1: usize,
    pub l1_started: bool,

    /// L2: senders' evaluation points, and per chunk the points they
    /// reconstructed at L1.
    pub l2_shares: (Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>),
    pub recv_l2: usize,
    pub l2_started: bool,
    /// The publicly reconstructed values, padded, until termination moves
    /// them out to the consumer.
    pub values: Vec<FieldElement<F>>,
    /// Set once `values` has been filled — `values` may legitimately be empty
    /// after termination moved it out, so emptiness is not the signal.
    pub values_ready: bool,

    pub hash_votes: HashSet<Hash>,
    pub hash_voters: Vec<Replica>,

    pub terminated: bool,
}

impl<F: ProtocolField> ReconState<F> {
    pub fn new() -> Self {
        Self {
            kind: None,
            config: None,
            padding: None,
            num_chunks: 0,
            l1_shares: (Vec::new(), Vec::new()),
            recv_l1: 0,
            l1_started: false,
            l2_shares: (Vec::new(), Vec::new()),
            recv_l2: 0,
            l2_started: false,
            values: Vec::new(),
            values_ready: false,
            hash_votes: HashSet::new(),
            hash_voters: Vec::new(),
            terminated: false,
        }
    }

    /// Size the per-chunk buffers, if not sized yet. A no-op once the batch is
    /// past the point where the buffers are read, so a late message cannot
    /// re-allocate what was deliberately freed.
    pub fn init_chunks(&mut self, num_chunks: usize) {
        if self.num_chunks == 0 {
            self.num_chunks = num_chunks;
        }
        if !self.l1_started {
            while self.l1_shares.1.len() < self.num_chunks {
                self.l1_shares.1.push(Vec::new());
            }
        }
        if !self.l2_started {
            while self.l2_shares.1.len() < self.num_chunks {
                self.l2_shares.1.push(Vec::new());
            }
        }
    }

    pub fn own_init_done(&self) -> bool {
        self.config.is_some()
    }

    /// An L1 share is live until the L1 interpolation claims the buffers.
    pub fn accepts_l1(&self) -> bool {
        !self.terminated && !self.l1_started
    }

    /// An L2 point is live until the L2 interpolation claims the buffers.
    pub fn accepts_l2(&self) -> bool {
        !self.terminated && !self.l2_started
    }
}

impl<F: ProtocolField> Default for ReconState<F> {
    fn default() -> Self {
        Self::new()
    }
}
