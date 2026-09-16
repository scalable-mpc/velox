use std::collections::HashMap;

use super::ex_compr_state::ExComprState;
use lambdaworks_math::field::element::FieldElement;
use fields::ProtocolField;

pub struct VerificationState<F: ProtocolField>{
    // A vector of multiplication tuples (a,b,a*b) to be verified at each depth
    pub mult_tuples: HashMap<usize, (Vec<FieldElement<F>>, Vec<FieldElement<F>>, Vec<FieldElement<F>>)>,
    pub ex_compr_state: HashMap<usize, ExComprState<F>>,
    // Prepare a beaver triple as a random mask for verification
    pub random_mask: (Option<FieldElement<F>>,Option<FieldElement<F>>,Option<FieldElement<F>>),
    // indices, x_shares, y_shares, z_shares
    pub output_verf_reconstruction_shares: (Vec<FieldElement<F>>, Vec<FieldElement<F>>, Vec<FieldElement<F>>, Vec<FieldElement<F>>),

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
        }
    }

    /// The sharings a reveal at `depth` opens, recorded when it starts.
    pub fn add_reveal_sharings(&mut self, depth: usize, sharings: Vec<FieldElement<F>>) {
        self.revealed.entry(depth).or_insert_with(|| (Vec::new(), Vec::new())).0 = sharings;
    }

    /// The public values a reveal at `depth` produced, recorded when it ends.
    pub fn add_reveal_values(&mut self, depth: usize, values: Vec<FieldElement<F>>) {
        self.revealed.entry(depth).or_insert_with(|| (Vec::new(), Vec::new())).1 = values;
    }

    /// Every recorded reveal, in ascending depth order so all parties combine
    /// the same sequence. Takes, like `take_verified_tuples`.
    pub fn take_reveals(&mut self) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let mut depths: Vec<usize> = self.revealed.keys().copied().collect();
        depths.sort();
        let (mut sharings, mut values) = (Vec::new(), Vec::new());
        for depth in depths {
            if let Some((mut s, mut v)) = self.revealed.remove(&depth) {
                sharings.append(&mut s);
                values.append(&mut v);
            }
        }
        (sharings, values)
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

    pub fn add_compression_level_state(&mut self, 
        depth: usize, 
        x_shares: Vec<Vec<FieldElement<F>>>, 
        y_shares: Vec<Vec<FieldElement<F>>>, 
        z_shares: Vec<FieldElement<F>>
    ){
        let entry = self.ex_compr_state.entry(depth).or_insert_with(|| ExComprState::<F>::new(depth) );
        // Add the shares of x
        entry.x_sharings.extend(x_shares);
        // Add the shares of y
        entry.y_sharings.extend(y_shares);
        // Add the shares of z
        entry.mult_sharings.extend(z_shares);
    }
}