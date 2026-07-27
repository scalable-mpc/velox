use std::collections::HashMap;

use fields::LargeField;

use super::ex_compr_state::ExComprState;

pub struct VerificationState{
    // A vector of multiplication tuples (a,b,a*b) to be verified at each depth
    pub mult_tuples: HashMap<usize, (Vec<LargeField>, Vec<LargeField>, Vec<LargeField>)>,
    pub ex_compr_state: HashMap<usize, ExComprState>,
    // Prepare a beaver triple as a random mask for verification
    pub random_mask: (Option<LargeField>,Option<LargeField>,Option<LargeField>),
    // indices, x_shares, y_shares, z_shares
    pub output_verf_reconstruction_shares: (Vec<LargeField>, Vec<LargeField>, Vec<LargeField>, Vec<LargeField>),

    /// Set once this party's own circuit has finished and `delinearize_mult_tuples`
    /// has drawn the verification mask. Until then `mult_tuples` is still being
    /// filled, and delinearizing it would fold a partial tuple sequence.
    pub delinearization_ready: bool,
    /// Set once the tuple sequence has actually been delinearized, so the step
    /// runs exactly once.
    pub delinearized: bool,
}

impl VerificationState{
    pub fn new() -> Self {
        VerificationState{
            mult_tuples: HashMap::new(),
            ex_compr_state: HashMap::new(),
            random_mask: (None, None, None),
            output_verf_reconstruction_shares: (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            delinearization_ready: false,
            delinearized: false,
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
        -> (Vec<LargeField>, Vec<LargeField>, Vec<LargeField>)
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
    pub fn add_mult_inputs(&mut self, depth: usize, a_shares: Vec<LargeField>, b_shares: Vec<LargeField>,) {
        let entry = self.mult_tuples.entry(depth).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
        entry.0.extend(a_shares); // Add the shares of 'a' to the first vector
        entry.1.extend(b_shares); // Add the shares of 'b' to the second vector
    }

    pub fn add_mult_output_shares(&mut self, depth: usize, output_shares: Vec<LargeField>) {
        // For each multiplication tuple at this depth, we will assign the output share
        let entry = self.mult_tuples.entry(depth).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
        entry.2.extend(output_shares); // Add the shares of the output to the third vector
    }

    pub fn add_compression_level_state(&mut self, 
        depth: usize, 
        x_shares: Vec<Vec<LargeField>>, 
        y_shares: Vec<Vec<LargeField>>, 
        z_shares: Vec<LargeField>
    ){
        let entry = self.ex_compr_state.entry(depth).or_insert_with(|| ExComprState::new(depth) );
        // Add the shares of x
        entry.x_sharings.extend(x_shares);
        // Add the shares of y
        entry.y_sharings.extend(y_shares);
        // Add the shares of z
        entry.mult_sharings.extend(z_shares);
    }
}