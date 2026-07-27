use std::collections::{HashMap, HashSet, VecDeque};

use crypto::hash::Hash;
use fields::LargeField;
use types::Replica;

/// State for the public reconstruction of the squared random sharings, run as a
/// two-level linear protocol so each party sends O(1) field elements per value
/// instead of broadcasting every share.
pub struct RandBitReconState{
    /// Zero shares appended to round the batch up to a multiple of 2t+1; the
    /// same number of reconstructed values is trimmed at the end.
    pub padding: Option<usize>,
    /// Number of chunks of 2t+1 values the batch was split into.
    pub num_chunks: usize,

    /// L1: evaluation points of the senders, and per chunk their shares of this
    /// party's point on the chunk polynomial.
    pub l1_shares: (Vec<LargeField>, Vec<Vec<LargeField>>),
    pub recv_share_count_l1: usize,
    /// Claimed by whoever crosses the L1 threshold first, so a later message
    /// arriving while the interpolation is in flight does not redo it.
    pub l1_reconstruction_started: bool,
    /// This party's point on each chunk polynomial, recovered from L1.
    pub l1_reconstructed: Vec<LargeField>,

    /// L2: evaluation points of the senders, and per chunk the points they
    /// reconstructed at L1.
    pub l2_shares: (Vec<LargeField>, Vec<Vec<LargeField>>),
    pub recv_share_count_l2: usize,
    /// Same claim flag for the L2 interpolation.
    pub l2_reconstruction_started: bool,
    /// The publicly reconstructed values, recovered from L2.
    pub l2_reconstructed: Vec<LargeField>,

    /// Hash agreement over the reconstructed values.
    pub recv_hash_set: HashSet<Hash>,
    pub recv_hash_msgs: Vec<Replica>,

    pub terminated: bool,
}

impl RandBitReconState{
    pub fn new() -> Self{
        RandBitReconState{
            padding: None,
            num_chunks: 0,
            l1_shares: (Vec::new(), Vec::new()),
            recv_share_count_l1: 0,
            l1_reconstruction_started: false,
            l1_reconstructed: Vec::new(),
            l2_shares: (Vec::new(), Vec::new()),
            recv_share_count_l2: 0,
            l2_reconstruction_started: false,
            l2_reconstructed: Vec::new(),
            recv_hash_set: HashSet::new(),
            recv_hash_msgs: Vec::new(),
            terminated: false,
        }
    }

    /// Size the per-chunk share vectors once the batch has been chunked. Called
    /// both when this party starts its own reconstruction and when the first
    /// message of a round arrives, whichever happens first.
    pub fn init_chunks(&mut self, num_chunks: usize){
        if self.num_chunks == num_chunks{
            return;
        }
        self.num_chunks = num_chunks;
        self.l1_shares.1 = vec![Vec::new(); num_chunks];
        self.l2_shares.1 = vec![Vec::new(); num_chunks];
    }
}

/// Engine-side state for the two phases that feed the application: random bit
/// generation and input sharing. The circuit itself — wires, wire pairs and the
/// per-depth multiplication results — belongs to the application now.
pub struct MixCircuitState{
    pub rand_bit_inp_shares: Vec<LargeField>,
    pub rand_bit_recon_shares: HashMap<usize, Vec<LargeField>>,

    pub rand_bit_inverse_recon_values: Vec<LargeField>,
    pub rand_bit_sharings: VecDeque<LargeField>,
    pub rand_bit_reconstruction: HashMap<usize, Vec<LargeField>>,
    /// Public reconstruction of the squared sharings, run as a linear protocol.
    pub rand_bit_recon_state: RandBitReconState,

    pub input_acss_shares: HashMap<Replica, HashMap<usize,Vec<LargeField>>>,
    /// Set once the input sharings have been handed to the application, so the
    /// handover happens exactly once.
    pub input_sharings_forwarded: bool,
}

impl MixCircuitState{
    pub fn new() -> Self {
        MixCircuitState{
            rand_bit_inp_shares: Vec::new(),
            rand_bit_recon_shares: HashMap::new(),

            rand_bit_inverse_recon_values: Vec::new(),
            rand_bit_sharings: VecDeque::new(),
            rand_bit_reconstruction: HashMap::default(),
            rand_bit_recon_state: RandBitReconState::new(),

            input_acss_shares: HashMap::default(),
            input_sharings_forwarded: false,
        }
    }
}