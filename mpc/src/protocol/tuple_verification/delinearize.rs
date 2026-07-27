use application::Application;
use fields::LargeField;

use crate::{Context, protocol::online_phase::APPLICATION_DEPTH_OFFSET};

impl<A: Application> Context<A>{
    // This function will be used to compress the multiplication tuples
    // It will take the shares of a, b, and the output and compress them into a single representation
    pub async fn delinearize_mult_tuples(&mut self){
        // Here we will implement the logic for compressing the multiplication tuples
        // This might involve some form of serialization or aggregation of the shares
        // Initiate the random mask generation for the last level
        log::info!("Initiating verification process: Preparing a random mask and tossing a common coin");
        let random_a_share = self.rand_sharings_state.rand_sharings_mult.pop_front().unwrap();
        let random_b_share = self.rand_sharings_state.rand_sharings_mult.pop_front().unwrap();

        //let vec_a_share = vec![vec![random_a_share]];
        //let vec_b_share = vec![vec![random_b_share]];

        self.verf_state.random_mask.0 = Some(random_a_share);
        self.verf_state.random_mask.1 = Some(random_b_share);
        // Only now is this party's tuple sequence complete. `handle_common_coin_msg`
        // can reconstruct the delinearization coin well before we get here - the
        // coin shares come from parties that have already finished their circuit -
        // and without this flag it would delinearize whatever subset of depths had
        // been recorded so far.
        self.verf_state.delinearization_ready = true;

        //self.choose_multiplication_protocol(vec_a_share, vec_b_share, self.delinearization_depth).await;
        self.toss_common_coin(self.delinearization_depth).await;
    }

    pub async fn verify_coin_toss_deserialization(&mut self){
        // Both guards are load-bearing: this is reached from `toss_common_coin`
        // and again from `handle_common_coin_msg`, in either order.
        if !self.verf_state.delinearization_ready || self.verf_state.delinearized{
            return;
        }
        if !self.verf_state.ex_compr_state.contains_key(&self.delinearization_depth){
            return;
        }
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&self.delinearization_depth).unwrap();
        if ex_compr_state.coin_output.is_none(){
            return;
        }
        let coin_value = ex_compr_state.coin_output.clone().unwrap();
        let _depth_factor = self.compression_factor;
        // Reduce multiplicative depth by a factor of k in each iteration
        // Collect all multiplication tuples so far.
        //
        // Every depth the circuit and the random bit preparation multiplied at,
        // in a fixed order so all parties delinearize the same tuple sequence.
        // Verification's own multiplications live at `delinearization_depth` and
        // above, and are not themselves verified.
        //
        // The tuples are moved out, not copied: this is their last reader, so
        // holding a second copy of every (a, b, a*b) in the circuit alongside
        // `verf_state.mult_tuples` doubled the largest long-lived allocation in
        // the engine for the whole of verification.
        self.verf_state.delinearized = true;
        let is_verified_depth = |depth: usize| {
            depth == self.preprocessing_mult_depth
                || (depth >= APPLICATION_DEPTH_OFFSET && depth < self.delinearization_depth)
        };
        let (mut x_values, y_values, mut mult_values) = self.verf_state.take_verified_tuples(is_verified_depth);
        log::info!("Initiating verification process for {} multiplication tuples: x: {}, y: {}, mult: {}",x_values.len(), x_values.len(), y_values.len(), mult_values.len());
        if x_values.len() != y_values.len() || x_values.len() != mult_values.len() || x_values.len() == 0{
            log::error!("Invalid number of shares for delinearization {} {} {}, abandoning process", x_values.len(), y_values.len(), mult_values.len());
            return;
        }
        let mut r_iter = LargeField::one();
        for (x,mult) in x_values.iter_mut().zip(mult_values.iter_mut()){
            *x *= r_iter.clone();
            *mult *= r_iter.clone();
            r_iter *= coin_value.clone();
        }
        log::info!("Multiplication tuples after coin toss: x: {}, y: {}, mult: {}",x_values.len(), y_values.len(), mult_values.len());
        // Compress shares with dimension reduction factor k
        let summed_mult_value: LargeField = mult_values.into_iter().sum();
        self.init_compression_level(x_values, y_values, summed_mult_value, self.delinearization_depth +2).await;
    }
}