//! The linear multiplication protocol: form the masked products, open them
//! through the shared public reconstruction, and unmask.
//!
//! Per gate `k` with operands `a_k`, `b_k` (vectors, for inner-product gates)
//! and a pool mask `[r_k]`, the value handed to the reconstruction is the
//! degree-`2t` sharing `[a_k · b_k + r_k]`. Its polynomial is `f_a · f_b +
//! f_r`, whose coefficients above `t` are fixed by the factor polynomials, so
//! the reconstruction runs with the privacy term on: `t+1` degree-`2t`
//! sharings of zero per chunk of `2t+1` gates. Once the public
//! `a_k · b_k + r_k` are back and agreed on, subtracting `[r_k]` gives the
//! degree-`t` product sharing.

use planner::api::engine::Application;

use crate::{
    protocol::public_reconstruction::{ReconConfig, ReconKind},
    Context,
};

use fields::{rayon_async, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::*;

impl<F: ProtocolField, A: Application<F>> Context<F, A> {
    pub async fn init_linear_multiplication_prot(
        &mut self,
        a_vec_shares: Vec<Vec<FieldElement<F>>>,
        b_vec_shares: Vec<Vec<FieldElement<F>>>,
        depth: usize,
        mut rand_sharings: Vec<FieldElement<F>>,
        zero_sharings: Vec<FieldElement<F>>,
    ) {
        let num_gates = a_vec_shares.len();
        if num_gates != b_vec_shares.len() {
            log::error!(
                "Linear multiplication at depth {}: {} left and {} right operands",
                depth, num_gates, b_vec_shares.len()
            );
            return;
        }
        // Share inputs for later verification. Only the first sharing of each
        // gate is verified, so read it out by reference rather than cloning
        // every inner-product operand.
        if self.is_verified_depth(depth) {
            let first_a_shares: Vec<FieldElement<F>> = a_vec_shares.iter().map(|x| x[0].clone()).collect();
            let first_b_shares: Vec<FieldElement<F>> = b_vec_shares.iter().map(|x| x[0].clone()).collect();
            log::info!(
                "Adding shares to verification state with a:{} b:{} at depth {}",
                first_a_shares.len(), first_b_shares.len(), depth
            );
            self.verf_state.add_mult_inputs(depth, first_a_shares, first_b_shares);
        }

        // One mask per gate, and t+1 zero sharings per chunk of 2t+1 gates —
        // the reconstruction pads the batch up to whole chunks itself, with
        // zero shares that cost nothing, so only the real gates draw masks.
        let group = 2 * self.num_faults + 1;
        let num_zero_sharings = num_gates.div_ceil(group) * (self.num_faults + 1);
        if rand_sharings.len() < num_gates || zero_sharings.len() < num_zero_sharings {
            log::error!(
                "Not enough preprocessing for the linear multiplication at depth {}: {} gates need {} masks and {} zero sharings, got {} and {}",
                depth, num_gates, num_gates, num_zero_sharings, rand_sharings.len(), zero_sharings.len()
            );
            return;
        }
        rand_sharings.truncate(num_gates);

        let depth_state = self.mult_state.get_single_depth_state(depth, true, 0);
        depth_state.two_levels = true;
        depth_state.util_rand_sharings = rand_sharings.clone();

        // The masked products. Yielded: inner-product gates make this a real
        // job, and the operands are owned so it can be. They die with it.
        let values: Vec<FieldElement<F>> = rayon_async(move || {
            a_vec_shares
                .into_par_iter()
                .zip(b_vec_shares.into_par_iter())
                .zip(rand_sharings.into_par_iter())
                .map(|((a, b), r)| Self::dot_product(&a, &b) + r)
                .collect()
        })
        .await;

        self.init_public_reconstruction(
            depth,
            ReconKind::Multiplication,
            values,
            ReconConfig::MULTIPLICATION,
            Some(zero_sharings),
        )
        .await;
    }

    /// The reconstruction at `depth` has terminated with the public
    /// `a_k · b_k + r_k`: subtract the masks and hand the products on.
    pub async fn complete_linear_multiplication(&mut self, depth: usize, values: Vec<FieldElement<F>>) {
        let depth_state = self.mult_state.get_single_depth_state(depth, true, 0);
        if depth_state.depth_terminated {
            return;
        }
        let masks = std::mem::take(&mut depth_state.util_rand_sharings);
        if masks.len() != values.len() {
            log::error!(
                "Linear multiplication at depth {} reconstructed {} values for {} masks; abandoning the protocol",
                depth, values.len(), masks.len()
            );
            return;
        }
        log::info!("Subtracting {} masks from the reconstructed products at depth {}", masks.len(), depth);
        let products: Vec<FieldElement<F>> = values
            .into_iter()
            .zip(masks.into_iter())
            .map(|(value, mask)| value - mask)
            .collect();
        self.finish_multiplication_depth(depth, products).await;
    }
}
