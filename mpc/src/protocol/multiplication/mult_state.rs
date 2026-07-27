use std::{collections::{HashMap, HashSet}};

use crypto::hash::Hash;
use fields::LargeField;
use types::Replica;

pub struct MultState{
    pub depth_share_map: HashMap<usize, SingleDepthState>,
    pub output_layer: OutputLayerState,
}

pub struct SingleDepthState{
    // Each party sends one share from each group. This map is sorted group wise
    pub l1_shares: (Vec<LargeField>,Vec<Vec<LargeField>>),
    pub l1_shares_reconstructed: Vec<LargeField>,
    pub l2_shares: (Vec<LargeField>,Vec<Vec<LargeField>>),
    pub l2_shares_reconstructed: Vec<LargeField>,

    pub util_rand_sharings: Vec<LargeField>,
    
    pub two_levels: bool,
    pub padding_shares: usize,
    // TODO: replace these with mutexes
    pub recv_share_count_l1: usize,
    pub recv_share_count_l2: usize,

    /// Claimed by whichever L1/quadratic share crosses the `n-t` threshold, so a
    /// later share arriving after the interpolation neither redoes it nor tries
    /// to push into the buffers `clear_l1_shares` has already freed.
    pub l1_reconstruction_done: bool,
    /// Same claim for the L2 interpolation of the linear protocol.
    pub l2_reconstruction_done: bool,

    pub recv_hash_set: HashSet<Hash>,
    pub recv_hash_msgs: Vec<Replica>,

    pub depth_terminated: bool,
}

impl SingleDepthState{
    pub fn new(two_levels: bool) -> Self {
        SingleDepthState{
            l1_shares: (Vec::new(),Vec::new()),
            l1_shares_reconstructed: Vec::new(),
            
            l2_shares: (Vec::new(),Vec::new()),
            l2_shares_reconstructed: Vec::new(),
            
            util_rand_sharings: Vec::new(),

            two_levels,
            padding_shares: 0,

            recv_share_count_l1: 0,
            recv_share_count_l2: 0,

            l1_reconstruction_done: false,
            l2_reconstruction_done: false,

            recv_hash_set: HashSet::new(),
            recv_hash_msgs: Vec::new(),

            depth_terminated: false,
        }
    }

    /// Frees the per-party share buffers once this depth has terminated.
    ///
    /// By the time a depth terminates, the reconstructed secrets have already
    /// been cloned into the next depth's inputs, so the raw L1/L2 shares sent by
    /// other parties (the bulk of this struct's memory, O(n) `LargeField`s per
    /// group) are dead. The lightweight termination bookkeeping
    /// (`depth_terminated`, the receive counts, and the hash-vote sets) is kept
    /// so that any late-arriving share for this depth is still deduped and
    /// cannot re-trigger reconstruction or termination.
    pub fn clear_shares(&mut self) {
        self.clear_l1_shares();
        self.l1_shares_reconstructed = Vec::new();
        self.clear_l2_shares();
        self.l2_shares_reconstructed = Vec::new();
        self.util_rand_sharings = Vec::new();
    }

    /// Frees the raw L1 (or quadratic) shares the other parties sent.
    ///
    /// These are only ever read by the interpolation that runs the moment the
    /// `n-t`-th share lands; from then on the depth carries the interpolated
    /// `l1_shares_reconstructed` instead, so the O(n) shares per group are dead
    /// well before the depth terminates. Callers must set
    /// `l1_reconstruction_done` first — the handlers key their "drop this late
    /// share" check off that flag, and without it a late share would index the
    /// emptied `l1_shares.1` out of bounds.
    pub fn clear_l1_shares(&mut self) {
        self.l1_shares = (Vec::new(), Vec::new());
    }

    /// Frees the raw L2 shares. Same argument as `clear_l1_shares`: the L2
    /// interpolation happens once, on the `n-t`-th share, and its output lives
    /// in `l2_shares_reconstructed`. Guarded by `l2_reconstruction_done`.
    pub fn clear_l2_shares(&mut self) {
        self.l2_shares = (Vec::new(), Vec::new());
    }

    /// Size the per-group share vectors, if they are not sized yet.
    ///
    /// A `HashZMsg` from a party that is ahead of us creates this depth's entry
    /// before we know how many groups the depth holds, leaving `l1_shares.1` /
    /// `l2_shares.1` empty; a later share message would then index them out of
    /// bounds (L1) or silently drop every share (L2, which zips). Sizing here
    /// keeps that entry usable.
    ///
    /// Deliberately a no-op once the depth has terminated: a terminated depth's
    /// buffers were freed on purpose and late shares are dropped by the handlers,
    /// so re-allocating the group vectors would only leak them again.
    pub fn ensure_groups(&mut self, tot_groups: usize) {
        if self.depth_terminated {
            return;
        }
        while self.l1_shares.1.len() < tot_groups {
            self.l1_shares.1.push(Vec::new());
        }
        while self.l2_shares.1.len() < tot_groups {
            self.l2_shares.1.push(Vec::new());
        }
    }
}

pub struct OutputLayerState{
    pub output_shares: Option<(LargeField, Vec<LargeField>)>,

    pub output_wire_shares: HashMap<usize, (LargeField,Vec<LargeField>)>,
    pub reconstructed_masked_outputs: Option<Vec<LargeField>>,

    // CTRBC outputs
    pub broadcasted_masked_outputs: HashMap<Replica,Vec<u8>>,
    pub acs_output: Vec<Replica>,

    pub random_mask_shares: HashMap<usize, (LargeField,Vec<LargeField>)>,
}

impl OutputLayerState{
    pub fn new() -> Self {
        OutputLayerState{
            output_shares: None,

            output_wire_shares: HashMap::default(),
            reconstructed_masked_outputs: None,

            broadcasted_masked_outputs: HashMap::default(),
            acs_output: Vec::new(),

            random_mask_shares: HashMap::default()
        }
    }
}

impl MultState{
    pub fn new() -> Self {
        MultState{
            depth_share_map: HashMap::new(),
            output_layer: OutputLayerState::new()   
        }
    }

    pub fn get_single_depth_state(&mut self, depth: usize, two_levels: bool, tot_groups_in_level: usize) -> &mut SingleDepthState {
        let state = self.depth_share_map
            .entry(depth)
            .or_insert_with(|| SingleDepthState::new(two_levels));
        // For each group, we will have a vector of pairs (x,y) for each party.
        // Done through `ensure_groups` rather than at construction so an entry a
        // `HashZMsg` created ahead of us gets sized too.
        state.ensure_groups(tot_groups_in_level);
        state
    }
}