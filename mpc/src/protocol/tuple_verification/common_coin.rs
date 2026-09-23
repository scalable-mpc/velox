use planner::api::engine::Application;
use fields::{LargeFieldSer, lagrange_coefficients_at_zero, ProtocolField};

use crate::{Context, msg::ProtMsg, protocol::tuple_verification::ex_compr_state::ExComprState};

use super::{StatisticalElement, deser_statistical, pop_statistical_sharing, ser_statistical};

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    /// Open a uniformly random coin in the statistical extension `K`: one
    /// random sharing over `F` per coefficient, sent together.
    pub async fn toss_common_coin(&mut self, depth: usize){
        let Some(coin_share) = pop_statistical_sharing(&mut self.rand_sharings_state.rand_sharings_coin) else {
            log::warn!("toss_common_coin: No coins left to toss at depth {}. Cannot proceed.", depth);
            return;
        };
        let prot_msg = ProtMsg::ReconstructCoin(ser_statistical::<F>(&coin_share), depth);

        self.broadcast(prot_msg).await;
        if depth == self.delinearization_depth{
            self.verify_coin_toss_deserialization().await;
        }
        else {
            self.verify_level_termination(depth).await;
        }
    }

    pub async fn handle_common_coin_msg(&mut self, lf_share: LargeFieldSer, sender: usize, depth: usize){
        if !self.verf_state.ex_compr_state.contains_key(&depth){
            self.verf_state.ex_compr_state.insert(depth, ExComprState::<F>::new(depth));
        }
        let Some(share) = deser_statistical::<F>(&lf_share) else {
            log::warn!("Undecodable coin share from sender {} at depth {}", sender, depth);
            return;
        };
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth).unwrap();
        
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());
        ex_compr_state.coin_toss_shares.0.push(evaluation_point);
        ex_compr_state.coin_toss_shares.1.push(share);
        
        log::info!("Received coin toss from sender {} at depth {}", 
            sender, depth);
        
        if ex_compr_state.coin_toss_shares.0.len() >= self.num_faults + 1 && ex_compr_state.coin_output.is_none(){
            // Reconstruct the coin via the GEMM dispatcher (single-poly, but kept
            // on the GEMM pipeline so it dispatches consistently with the rest of
            // the protocol; the 50k bailout keeps tiny inputs on CPU).
            let xs = ex_compr_state.coin_toss_shares.0[0..self.num_faults + 1].to_vec();
            let ys = ex_compr_state.coin_toss_shares.1[0..self.num_faults + 1].to_vec();
            // Only the value at zero is used: the Lagrange weights, which lie
            // in `F`, give the coin directly, with no inverse and no GEMM to
            // build around one number.
            let coin_value: StatisticalElement<F> = lagrange_coefficients_at_zero(&xs)
                .iter()
                .zip(ys.iter())
                .map(|(weight, share)| F::from_statistical_coeffs(std::slice::from_ref(weight)) * share)
                .sum();
            ex_compr_state.coin_output = Some(coin_value.clone());
            if depth == self.delinearization_depth{
                log::info!("Reconstructed common coin at delinearization depth {}: {:?}", depth, ex_compr_state.coin_output);
                self.verify_coin_toss_deserialization().await;
            }
            else{
                // Trigger subsequent phase here. 
                log::info!("Reconstructed common coin at depth {}: {:?}", depth, ex_compr_state.coin_output);
                self.verify_level_termination(depth).await;
            }
        }
    }
}