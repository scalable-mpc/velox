use anyhow::Result;
use application::{Application, DepthInput};
use fields::ProtocolField;
use crate::{Context};
use lambdaworks_math::field::element::FieldElement;

/// Application circuit depths are shifted by this offset before they reach the
/// multiplication module, keeping them clear of the engine's own depths:
/// `preprocessing_mult_depth` (random bit squaring) sits below it and
/// `delinearization_depth` (verification) above.
pub const APPLICATION_DEPTH_OFFSET: usize = 100;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    // This function will be used to run the online phase of the protocol
    pub async fn init_random_shared_bits_preparation(&mut self) {
        // Take two random sharings, and multiply them using a random double sharing
        // Moved out rather than cloned-then-cleared: the clear immediately below
        // was already saying these are dead here, so the clone only served to
        // hold a second copy of both halves of the batch while the `Vec<Vec<_>>`
        // wrappers were being built.
        let a_shares = std::mem::take(&mut self.rand_sharings_state.rand_sharings_inputs.0).into_iter()
            .map(|x| vec![x]).collect();
        let b_shares = std::mem::take(&mut self.rand_sharings_state.rand_sharings_inputs.1).into_iter()
            .map(|x| vec![x]).collect();

        self.choose_multiplication_protocol(a_shares, b_shares, self.preprocessing_mult_depth).await;
    }

    /// Act on what the application scheduled for the next circuit depth.
    ///
    /// [`DepthInput::Waiting`] means the application has nothing to run yet — it
    /// is waiting on preprocessing, on more input sharings, or on another
    /// depth's results — so the engine stops and waits to be called again. An
    /// `Err` means the application has given up, which is a different thing and
    /// used to be indistinguishable from waiting: both arrived as an empty
    /// `DepthInput` and the protocol hung with the reason buried in one party's
    /// log.
    pub async fn handle_application_depth_input(&mut self, depth_input: Result<DepthInput<F>>){
        let depth_input = match depth_input{
            Ok(depth_input) => depth_input,
            Err(err) => {
                log::error!("Application aborted the circuit: {:#}", err);
                return;
            }
        };

        match depth_input{
            DepthInput::Waiting => {}
            DepthInput::Multiply { depth, x, y } => {
                // The depth is the application's, not a counter kept here: it
                // selects the reserved preprocessing slice, so it has to be the
                // batch's position in the circuit rather than its position in
                // whatever order this party happened to schedule things.
                self.multiply_application_batch(x, y, depth).await;
            }
            // The reveal primitive and the masked multiplication are declared
            // ahead of their protocols so applications can be written against
            // them; until those land, scheduling one is an application error
            // like any other, surfaced here rather than as a hang.
            DepthInput::Reveal { depth, values } => {
                log::error!(
                    "Application scheduled a reveal of {} values at depth {}, which this engine does not implement yet",
                    values.len(), depth
                );
            }
            DepthInput::MaskedMultiply { depth, x, .. } => {
                log::error!(
                    "Application scheduled a masked multiplication of {} gates at depth {}, which this engine does not implement yet",
                    x.len(), depth
                );
            }
            DepthInput::Done(output_wires) => {
                self.handle_application_output(output_wires).await;
            }
        }
    }

    /// The application's circuit has terminated. Record the output sharings and
    /// move on to verifying every multiplication the circuit performed.
    async fn handle_application_output(&mut self, output_wires: Vec<FieldElement<F>>){
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
    pub async fn verify_application_depth_termination(&mut self, depth: usize, shares: Vec<FieldElement<F>>){
        let application_depth = depth - APPLICATION_DEPTH_OFFSET;
        log::info!("Multiplication terminated at application depth {}, handing {} sharings back to the application",
            application_depth, shares.len());
        let depth_input = self.app.on_depth_complete(application_depth, shares).await;
        Box::pin(self.handle_application_depth_input(depth_input)).await;
    }
}
