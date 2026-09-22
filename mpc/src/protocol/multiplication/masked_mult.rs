//! An application's `DepthInput::MaskedMultiply`: the linear multiplication
//! with the caller's mask in place of a pool mask, and the masked products
//! handed back public instead of unmasked.
//!
//! Per gate `k` the value opened is `[x_k · y_k + mask_k]`, a degree-`2t`
//! sharing whose polynomial is `f_x · f_y + f_mask`. The application's
//! degree-`t` mask blinds the constant term, not the coefficients above `t`,
//! which the factor polynomials fix — so the reconstruction runs exactly as
//! a multiplication's does, degree `2t` with the privacy term on, drawing
//! `t+1` degree-`2t` zero sharings per chunk of `2t+1` gates from the
//! depth's reservation (`for_masked_depth`). No mask is drawn: the caller
//! brought its own.
//!
//! The public `c_k = x_k · y_k + mask_k` go to `on_reveal_complete`. For
//! verification the depth registers the tuple `(x_k, y_k, c_k − [mask_k])`,
//! so the guarantee is `Multiply`'s: a `c_k` shifted at the redundancy-free
//! L2 step fails the tuple check, and the `c_k` need no place in the reveal
//! check.

use planner::api::engine::Application;

use crate::{
    protocol::{
        online_phase::APPLICATION_DEPTH_OFFSET,
        public_reconstruction::{ReconConfig, ReconKind},
    },
    Context,
};

use fields::{rayon_async, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::*;

impl<F: ProtocolField, A: Application<F>> Context<F, A> {
    /// Start a masked multiplication batch at application depth `depth`.
    pub async fn multiply_masked_application_batch(
        &mut self,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
        mask: Vec<FieldElement<F>>,
        depth: usize,
    ) {
        if x.is_empty() {
            // Nothing to open; the application still gets its callback, so an
            // empty batch is not a hang.
            log::warn!("Application scheduled an empty masked multiplication at depth {}; completing it with no values", depth);
            self.deliver_masked_products(depth, Vec::new()).await;
            return;
        }
        if x.len() != y.len() || x.len() != mask.len() {
            log::error!(
                "Application scheduled a masked multiplication at depth {} with {} left, {} right operands and {} masks",
                depth, x.len(), y.len(), mask.len()
            );
            return;
        }
        let zero_sharings = match self.app_preprocessing.for_masked_depth(depth, x.len()) {
            Ok(zero_sharings) => zero_sharings,
            Err(err) => {
                log::error!("Cannot run the application's masked multiplication at depth {}: {}", depth, err);
                return;
            }
        };
        let engine_depth = depth + APPLICATION_DEPTH_OFFSET;

        // The tuple's inputs now, its output when the products are back.
        if self.is_verified_depth(engine_depth) {
            log::info!("Adding {} masked gates to verification state at depth {}", x.len(), engine_depth);
            self.verf_state.add_mult_inputs(engine_depth, x.clone(), y.clone());
        }

        let depth_state = self.mult_state.get_single_depth_state(engine_depth, true, 0);
        depth_state.two_levels = true;
        depth_state.util_rand_sharings = mask.clone();

        let values: Vec<FieldElement<F>> = rayon_async(move || {
            x.into_par_iter()
                .zip(y.into_par_iter())
                .zip(mask.into_par_iter())
                .map(|((x, y), mask)| x * y + mask)
                .collect()
        })
        .await;

        self.init_public_reconstruction(
            engine_depth,
            ReconKind::MaskedMultiplication,
            values,
            ReconConfig::MULTIPLICATION,
            Some(zero_sharings),
        )
        .await;
    }

    /// The reconstruction at `engine_depth` has terminated with the public
    /// `c_k`: register `c_k − [mask_k]` as the tuple's product, close the
    /// depth, and hand the `c_k` to the application.
    pub async fn complete_masked_multiplication(&mut self, engine_depth: usize, values: Vec<FieldElement<F>>) {
        let depth_state = self.mult_state.get_single_depth_state(engine_depth, true, 0);
        if depth_state.depth_terminated {
            return;
        }
        let masks = std::mem::take(&mut depth_state.util_rand_sharings);
        if masks.len() != values.len() {
            log::error!(
                "Masked multiplication at depth {} reconstructed {} values for {} masks; abandoning the protocol",
                engine_depth, values.len(), masks.len()
            );
            return;
        }
        depth_state.depth_terminated = true;
        depth_state.clear_shares();

        if self.is_verified_depth(engine_depth) {
            let products: Vec<FieldElement<F>> = values.iter().zip(masks.iter()).map(|(c, mask)| c - mask).collect();
            self.verf_state.add_mult_output_shares(engine_depth, products);
        }
        let depth = engine_depth - APPLICATION_DEPTH_OFFSET;
        log::info!(
            "Masked multiplication terminated at application depth {}, handing {} public values back to the application",
            depth, values.len()
        );
        self.deliver_masked_products(depth, values).await;
    }

    async fn deliver_masked_products(&mut self, depth: usize, values: Vec<FieldElement<F>>) {
        let depth_input = self.app.on_reveal_complete(depth, values).await;
        Box::pin(self.handle_application_depth_input(depth_input)).await;
    }
}
