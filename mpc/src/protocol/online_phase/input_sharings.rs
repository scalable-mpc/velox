use application::Application;
use std::collections::HashMap;

use fields::{LargeFieldSer, ProtocolField, FieldSer};
use types::Replica;

use crate::Context;
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    /// Ask the application what this party contributes to the circuit's input
    /// wires and deal it through ACSS-Ab.
    ///
    /// One secret per sharing: Velox sharings are unpacked. The application used
    /// to return a `Vec<Vec<_>>` and have everything past the first secret of
    /// each inner vector dropped with a warning — packing residue, like the
    /// first-half/second-half pairs that used to run through the rest of this
    /// interface.
    pub async fn initialize_input_sharing(&mut self){
        let input_secrets = self.app.inputs().await;
        if input_secrets.is_empty(){
            log::info!("No input sharings to propose on party {}; skipping input ACSS", self.myid);
            return;
        }

        let inputs_ser: Vec<LargeFieldSer> = input_secrets.iter().map(|secret| secret.ser_be()).collect();

        log::info!("Initiating input sharing in preprocessing phase for {} inputs", inputs_ser.len());
        let status = self.acss_ab_send.send((self.input_acss_id_offset, inputs_ser)).await;
        if status.is_err(){
            log::error!("Failed to send input value to ACSS protocol because of error: {:?}", status.err().unwrap());
        }
    }

    pub async fn handle_input_acss_termination(&mut self, instance_id: usize, sender: Replica, shares: Option<Vec<LargeFieldSer>>){
        log::info!("Received input ACSS termination message from sender {} for instance ID {}", sender, instance_id);
        if shares.is_none(){
            log::error!("Abort ACSS protocol of dealer {} and terminate MPC", sender);
            return;
        }

        let shares = shares.unwrap();
        let input_sharing_inst = instance_id - self.input_acss_id_offset;

        if !self.mix_circuit_state.input_acss_shares.contains_key(&sender){
            let input_sharing_state = HashMap::default();
            self.mix_circuit_state.input_acss_shares.insert(sender, input_sharing_state);
        }

        let input_sharing_state = self.mix_circuit_state.input_acss_shares.get_mut(&sender).unwrap();
        let shares_deser: Vec<FieldElement<F>> = shares.into_iter()
            .map(|el| F::from_bytes_be(&el).unwrap())
            .collect();

        input_sharing_state.insert(input_sharing_inst, shares_deser.clone());
        
        let depth_input = self.app.input_sharing_termination(sender, shares_deser).await;
        self.handle_application_depth_input(depth_input).await;
    }
}
