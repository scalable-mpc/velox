use std::collections::HashMap;

use super::{ex_compr_state::ExComprState, StatisticalElement};
use lambdaworks_math::field::element::FieldElement;
use fields::ProtocolField;

pub struct VerificationState<F: ProtocolField>{
    // A vector of multiplication tuples (a,b,a*b) to be verified at each depth
    pub mult_tuples: HashMap<usize, (Vec<FieldElement<F>>, Vec<FieldElement<F>>, Vec<FieldElement<F>>)>,
    pub ex_compr_state: HashMap<usize, ExComprState<F>>,
    // Prepare a beaver triple as a random mask for verification, over `K`
    pub random_mask: (Option<StatisticalElement<F>>,Option<StatisticalElement<F>>,Option<StatisticalElement<F>>),
    // indices, x_shares, y_shares, z_shares; the shares are over `K`
    pub output_verf_reconstruction_shares: (Vec<FieldElement<F>>, Vec<StatisticalElement<F>>, Vec<StatisticalElement<F>>, Vec<StatisticalElement<F>>),

    /// Set once this party's own circuit has finished and `delinearize_mult_tuples`
    /// has drawn the verification mask. Until then `mult_tuples` is still being
    /// filled, and delinearizing it would fold a partial tuple sequence.
    pub delinearization_ready: bool,
    /// Set once the tuple sequence has actually been delinearized, so the step
    /// runs exactly once.
    pub delinearized: bool,

    /// Every value an application revealed, by engine depth: the sharings
    /// that were opened and the public values that came back. The
    /// verification phase checks a coin-weighted combination of
    /// `[v_i] − v_i` before the output is unmasked.
    pub revealed: HashMap<usize, (Vec<FieldElement<F>>, Vec<FieldElement<F>>)>,
    /// Set once this party has folded its own reveals into `[Δ]` and broadcast
    /// its share; until then shares from faster parties are only collected.
    pub reveal_check_sent: bool,
    /// Set once `[Δ]` has been opened, so the check runs exactly once.
    pub reveal_check_done: bool,
    /// The shares of `[Δ]` received so far: evaluation points and shares
    /// over `K`.
    pub reveal_check_shares: (Vec<FieldElement<F>>, Vec<StatisticalElement<F>>),

    /// The two halves of verification run side by side; the output waits
    /// for both. `verification_finished` makes the handover run once.
    pub tuples_verified: bool,
    pub reveals_verified: bool,
    pub verification_finished: bool,
}

impl<F: ProtocolField> VerificationState<F>{
    pub fn new() -> Self {
        VerificationState{
            mult_tuples: HashMap::new(),
            ex_compr_state: HashMap::new(),
            random_mask: (None, None, None),
            output_verf_reconstruction_shares: (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            delinearization_ready: false,
            delinearized: false,
            revealed: HashMap::new(),
            reveal_check_sent: false,
            reveal_check_done: false,
            reveal_check_shares: (Vec::new(), Vec::new()),
            tuples_verified: false,
            reveals_verified: false,
            verification_finished: false,
        }
    }

    /// Move every verified depth's `(a, b, a·b)` triple out of `mult_tuples`, in
    /// ascending depth order so all parties delinearize the same sequence.
    ///
    /// Takes rather than clones, and drops the map afterwards: delinearization is
    /// the only reader of `mult_tuples` (see `verify_coin_toss_deserialization`),
    /// it runs once, and nothing writes to the map after it — `add_mult_inputs`
    /// and `add_mult_output_shares` are both gated on `is_verified_depth`, and no
    /// verified depth is still running by the time this is called.
    pub fn take_verified_tuples(&mut self, is_verified: impl Fn(usize) -> bool)
        -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>, Vec<FieldElement<F>>)
    {
        let mut verified_depths: Vec<usize> = self.mult_tuples.keys()
            .copied()
            .filter(|depth| is_verified(*depth))
            .collect();
        verified_depths.sort();

        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        let mut mult_values = Vec::new();
        for depth in verified_depths{
            let Some(tuples) = self.mult_tuples.get_mut(&depth) else { continue };
            x_values.append(&mut tuples.0);
            y_values.append(&mut tuples.1);
            mult_values.append(&mut tuples.2);
        }
        self.mult_tuples.clear();
        (x_values, y_values, mult_values)
    }

    // Function to add a multiplication tuple for verification
    pub fn add_mult_inputs(&mut self, depth: usize, a_shares: Vec<FieldElement<F>>, b_shares: Vec<FieldElement<F>>,) {
        let entry = self.mult_tuples.entry(depth).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
        entry.0.extend(a_shares); // Add the shares of 'a' to the first vector
        entry.1.extend(b_shares); // Add the shares of 'b' to the second vector
    }

    pub fn add_mult_output_shares(&mut self, depth: usize, output_shares: Vec<FieldElement<F>>) {
        // For each multiplication tuple at this depth, we will assign the output share
        let entry = self.mult_tuples.entry(depth).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
        entry.2.extend(output_shares); // Add the shares of the output to the third vector
    }
}