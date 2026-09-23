use planner::api::engine::Application;

use crate::{Context, protocol::online_phase::APPLICATION_DEPTH_OFFSET};
use fields::ProtocolField;

use super::{StatisticalElement, pop_statistical_sharing};

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    // This function will be used to compress the multiplication tuples
    // It will take the shares of a, b, and the output and compress them into a single representation
    pub async fn delinearize_mult_tuples(&mut self){
        // Here we will implement the logic for compressing the multiplication tuples
        // This might involve some form of serialization or aggregation of the shares
        // Initiate the random mask generation for the last level
        log::info!("Initiating verification process: Preparing a random mask and tossing a common coin");
        // The mask lives in `K` with the rest of the compression: `d` random
        // sharings over `F` for each half.
        let random_a_share = pop_statistical_sharing(&mut self.rand_sharings_state.rand_sharings_mult).unwrap();
        let random_b_share = pop_statistical_sharing(&mut self.rand_sharings_state.rand_sharings_mult).unwrap();

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
        // The reveal check needs only the coin and this party's finished
        // circuit, both in hand here: start it now, alongside the compression.
        self.init_reveal_check().await;
        let is_verified_depth = |depth: usize| {
            depth == self.preprocessing_mult_depth
                || (depth >= APPLICATION_DEPTH_OFFSET && depth < self.delinearization_depth)
        };
        let (x_values, y_values, mult_values) = self.verf_state.take_verified_tuples(is_verified_depth);
        log::info!("Initiating verification process for {} multiplication tuples: x: {}, y: {}, mult: {}",x_values.len(), x_values.len(), y_values.len(), mult_values.len());
        if x_values.len() != y_values.len() || x_values.len() != mult_values.len() || x_values.len() == 0{
            log::error!("Invalid number of shares for delinearization {} {} {}, abandoning process", x_values.len(), y_values.len(), mult_values.len());
            return;
        }
        // The tuples move into the statistical extension `K` here, weighted by
        // the powers of the coin, which lies in `K`: a wrong tuple survives the
        // fold with probability about `#tuples / |K|`.
        let embed = |value: &_| F::from_statistical_coeffs(std::slice::from_ref(value));
        let mut r_iter = StatisticalElement::<F>::one();
        let mut weighted_x = Vec::with_capacity(x_values.len());
        let mut summed_mult_value = StatisticalElement::<F>::zero();
        for (x, mult) in x_values.iter().zip(mult_values.iter()){
            weighted_x.push(embed(x) * &r_iter);
            summed_mult_value += embed(mult) * &r_iter;
            r_iter *= &coin_value;
        }
        drop((x_values, mult_values));
        let y_values: Vec<StatisticalElement<F>> = y_values.iter().map(embed).collect();
        log::info!("Multiplication tuples after coin toss: {} over a degree-{} extension", weighted_x.len(), F::STATISTICAL_DEGREE);
        // Compress shares with dimension reduction factor k
        self.init_compression_level(weighted_x, y_values, summed_mult_value, self.delinearization_depth +2).await;
    }
}