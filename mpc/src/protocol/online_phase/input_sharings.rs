use application::Application;
use std::collections::HashMap;

use fields::ByteConversion;
use fields::{LargeFieldSer, LargeField};
use types::Replica;

use crate::Context;

impl<A: Application> Context<A>{
    /// Ask the application what this party contributes to the circuit's input
    /// wires and deal it through ACSS-Ab.
    ///
    /// The application returns one inner `Vec` per sharing. Velox sharings are
    /// unpacked — one secret each — so anything beyond the first secret of a
    /// sharing has nowhere to go and is dropped with a warning.
    pub async fn initialize_input_sharing(&mut self){
        let input_sharings = self.app.inputs().await;
        if input_sharings.is_empty(){
            log::info!("No input sharings to propose on party {}; skipping input ACSS", self.myid);
            return;
        }

        let mut inputs_ser: Vec<LargeFieldSer> = Vec::with_capacity(input_sharings.len());
        for secrets in input_sharings.into_iter(){
            if secrets.len() != 1{
                log::warn!(
                    "Application proposed a sharing packing {} secrets, but Velox sharings hold one; keeping the first",
                    secrets.len()
                );
            }
            match secrets.into_iter().next(){
                Some(secret) => inputs_ser.push(secret.to_bytes_be()),
                None => log::warn!("Application proposed an empty input sharing; skipping it"),
            }
        }

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
        let shares_deser: Vec<LargeField> = shares.into_iter()
            .map(|el| LargeField::from_bytes_be(&el).unwrap())
            .collect();

        input_sharing_state.insert(input_sharing_inst, shares_deser.clone());
        
        let depth_input = self.app.input_sharing_termination(sender, shares_deser).await;
        self.handle_application_depth_input(depth_input).await;
    }
}
