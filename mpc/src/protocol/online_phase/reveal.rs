//! An application's `DepthInput::Reveal`: open degree-`t` sharings to
//! everyone and hand the values back through `on_reveal_complete`.
//!
//! The reconstruction runs at degree `t` without a privacy term, which is
//! correct for what the ops layer reveals — values blinded by a fresh random
//! sharing — and is the primitive's contract: what a party learns at L1 is
//! the sharing polynomial of a public combination of the revealed values
//! (see `public_reconstruction::math`), so an application must not reveal a
//! sharing whose polynomial is correlated with a secret it keeps.
//!
//! Every `([v], v)` pair is recorded for the verification phase, which
//! checks a coin-weighted combination of them before the output is unmasked:
//! the L2 step has no redundancy, so a corrupt party could otherwise shift a
//! revealed value consistently at every honest party.

use planner::api::engine::Application;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

use crate::{
    protocol::public_reconstruction::{ReconConfig, ReconKind},
    Context,
};

use super::online_phase::APPLICATION_DEPTH_OFFSET;

impl<F: ProtocolField, A: Application<F>> Context<F, A> {
    /// Start revealing `values` at application depth `depth`.
    pub async fn init_reveal(&mut self, depth: usize, values: Vec<FieldElement<F>>) {
        if values.is_empty() {
            // Nothing to reconstruct; the application still gets its callback,
            // so an empty batch is not a hang.
            log::warn!("Application scheduled an empty reveal at depth {}; completing it with no values", depth);
            self.deliver_reveal(depth, Vec::new()).await;
            return;
        }
        let engine_depth = depth + APPLICATION_DEPTH_OFFSET;
        // The sharings opened here, kept for the reveal check.
        self.verf_state.revealed.entry(engine_depth).or_default().0 = values.clone();
        self.init_public_reconstruction(engine_depth, ReconKind::Reveal, values, ReconConfig::REVEAL, None).await;
    }

    /// The reconstruction at `engine_depth` has terminated with the public
    /// values: record them for verification and hand them to the application.
    pub async fn complete_reveal(&mut self, engine_depth: usize, values: Vec<FieldElement<F>>) {
        self.verf_state.revealed.entry(engine_depth).or_default().1 = values.clone();
        let depth = engine_depth - APPLICATION_DEPTH_OFFSET;
        log::info!("Reveal terminated at application depth {}, handing {} values back to the application", depth, values.len());
        self.deliver_reveal(depth, values).await;
    }

    async fn deliver_reveal(&mut self, depth: usize, values: Vec<FieldElement<F>>) {
        let depth_input = self.app.on_reveal_complete(depth, values).await;
        Box::pin(self.handle_application_depth_input(depth_input)).await;
    }
}
