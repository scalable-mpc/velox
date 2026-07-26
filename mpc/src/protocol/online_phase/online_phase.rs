use application::{Application, DepthInput};
use fields::{LargeField};
use crate::{Context};

/// Application circuit depths are shifted by this offset before they reach the
/// multiplication module, keeping them clear of the engine's own depths:
/// `preprocessing_mult_depth` (random bit squaring) sits below it and
/// `delinearization_depth` (verification) above.
pub const APPLICATION_DEPTH_OFFSET: usize = 100;

impl<A: Application> Context<A>{
    // This function will be used to run the online phase of the protocol
    pub async fn init_random_shared_bits_preparation(&mut self) {
        // Take two random sharings, and multiply them using a random double sharing
        let a_shares = self.rand_sharings_state.rand_sharings_inputs.0.clone().into_iter()
            .map(|x| vec![x]).collect();
        let b_shares = self.rand_sharings_state.rand_sharings_inputs.1.clone().into_iter()
        .map(|x| vec![x]).collect();

        self.rand_sharings_state.rand_sharings_inputs.0.clear();
        self.rand_sharings_state.rand_sharings_inputs.1.clear();

        self.choose_multiplication_protocol(a_shares, b_shares, self.preprocessing_mult_depth).await;
    }

    /// Act on what the application scheduled for the next circuit depth.
    ///
    /// An empty `DepthInput` means the application has nothing to run yet — it
    /// is waiting on preprocessing, on more input sharings, or on another
    /// depth's results — so the engine simply stops and waits to be called again.
    pub async fn handle_application_depth_input(&mut self, depth_input: DepthInput){
        if depth_input.network_routing.is_some(){
            log::warn!("Application scheduled network routing, but Velox has no network routing module; ignoring the request");
        }

        let Some(mult) = depth_input.mult else {
            return;
        };

        // A `Multiplication` carrying output sharings marks the last depth: the
        // circuit is done and those sharings are its outputs.
        if let Some(output) = mult.output{
            self.handle_application_output(output.0).await;
            return;
        }

        self.multiply_application_batch(mult).await;
    }

    /// The application's circuit has terminated. Record the output sharings and
    /// move on to verifying every multiplication the circuit performed.
    async fn handle_application_output(&mut self, output_wires: Vec<LargeField>){
        log::info!("Application circuit terminated with {} output wires, adding random masks to output wires", output_wires.len());
        self.mult_state.output_layer.output_shares = Some((
            Self::get_share_evaluation_point(self.myid, self.use_fft, self.roots_of_unity.clone()),
            output_wires
        ));
        self.terminate("Online".to_string(), vec![]).await;
        log::info!("Starting verification of multiplications");
        self.delinearize_mult_tuples().await;
    }

    /// A multiplication batch at an application depth has terminated. Cede
    /// control back to the application with the resulting sharings, and run
    /// whatever it schedules next.
    pub async fn verify_application_depth_termination(&mut self, depth: usize, shares: Vec<LargeField>){
        let application_depth = depth - APPLICATION_DEPTH_OFFSET;
        log::info!("Multiplication terminated at application depth {}, handing {} sharings back to the application",
            application_depth, shares.len());
        // Velox sharings are unpacked, so the application's second-half
        // representation is always empty.
        let depth_input = self.app.on_multiplication_complete(application_depth, (shares, Vec::new())).await;
        Box::pin(self.handle_application_depth_input(depth_input)).await;
    }
}
