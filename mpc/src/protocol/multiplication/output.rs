use planner::api::engine::Application;
use std::collections::HashSet;

use crypto::hash::do_hash;
use fields::{LargeFieldSer, check_if_all_points_lie_on_degree_x_polynomial, ProtocolField, FieldSer};
use types::Replica;

use crate::{Context, msg::ProtMsg};
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    // Last layer of the protocol
    pub async fn reconstruct_output(&mut self){
        if self.mult_state.output_layer.output_shares.is_none(){
            log::error!("Output layer shares are missing, abandoning protocol");
            return;
        }
        
        let mut output_wire_shares = self.mult_state.output_layer.output_shares.clone().unwrap().1;
        // Add random masks
        let mut random_mask_shares = Vec::with_capacity(output_wire_shares.len());
        for output_wire_share in output_wire_shares.iter_mut(){
            if self.output_mask_state.rand_sharings.is_empty(){
                log::error!("Not enough random sharings for mask reconstruction, abandoning the protocol");
                return;
            }
            let random_mask_share = self.output_mask_state.rand_sharings.pop_front().unwrap();
            *output_wire_share += random_mask_share.clone();
            random_mask_shares.push(random_mask_share);
        }
        let evaluation_point = Self::get_share_evaluation_point(self.myid, self.use_fft, self.roots_of_unity.clone());
        self.mult_state.output_layer.output_wire_shares.insert(self.myid, (evaluation_point, output_wire_shares.clone()));
        // Reconstruct the output
        let output_masks_ser = output_wire_shares.iter()
            .map(|x| x.ser_be())
            .collect::<Vec<LargeFieldSer>>();
        

        self.broadcast(ProtMsg::ReconstructMaskedOutput(output_masks_ser)).await;
        // Save random masks for public recontruction after the output
    }

    pub async fn handle_reconstruct_masked_output(&mut self, ser_shares: Vec<LargeFieldSer>, sender:Replica){
        log::info!("Handling reconstruct masked output shares from sender {}", sender);
        // Deserialize shares
        let shares_lf: Vec<FieldElement<F>> = ser_shares.into_iter().map(|x| F::from_bytes_be(&x).unwrap()).collect();
        let evaluation_point = Self::get_share_evaluation_point(sender,self.use_fft, self.roots_of_unity.clone());
        
        self.mult_state.output_layer.output_wire_shares.insert(sender, (evaluation_point, shares_lf));

        if self.mult_state.output_layer.output_wire_shares.len() >= self.num_nodes - self.num_faults && self.mult_state.output_layer.reconstructed_masked_outputs.is_none(){
            // Reconstruct output
            let mut evaluation_points = Vec::with_capacity(self.num_nodes);
            let mut evaluations = Vec::new();
            for i in 0..self.num_nodes{
                if self.mult_state.output_layer.output_wire_shares.contains_key(&i){
                    // Evaluations and evaluation points
                    let (evaluation_point, evaluation) = self.mult_state.output_layer.output_wire_shares.get(&i).unwrap().clone();
                    evaluation_points.push(evaluation_point);
                    if evaluations.len()  == 0{
                        for _ in 0..evaluation.len(){
                            evaluations.push(Vec::new());
                        }
                    }
                    for (index,eval) in evaluation.into_iter().enumerate(){
                        evaluations[index].push(eval);
                    }
                }
            }
            // Reconstruct the outputs. A circuit with no output wires — one
            // whose product is the sharings the application kept, as in a
            // key-generation setup — still goes through the CTRBC/ACS steps
            // below: that is where the parties agree that enough of them
            // passed verification. There is just nothing to interpolate, and
            // the degree check cannot be asked about zero polynomials.
            let outputs_recon = if evaluations.is_empty(){
                log::info!("Circuit declared no output wires; nothing to reconstruct");
                Vec::new()
            }
            else{
                let verification_result = check_if_all_points_lie_on_degree_x_polynomial(evaluation_points, evaluations, self.num_faults+1);
                let Some(polys) = verification_result.1.filter(|_| verification_result.0) else{
                    log::error!("Output reconstruction failed, shares not on a degree-t polynomial");
                    return;
                };
                log::info!("Masked output wires successfully reconstructed, shares are on a degree-t polynomial");
                polys.iter().map(|poly|poly.evaluate(&FieldElement::<F>::zero())).collect::<Vec<FieldElement<F>>>()
            };
            self.mult_state.output_layer.reconstructed_masked_outputs = Some(outputs_recon.clone());
            // Broadcast using a CTRBC channel
            let mut broadcast_output = Vec::new();
            broadcast_output.push(1u8);
            for output in outputs_recon.iter(){
                broadcast_output.extend(output.ser_be());
            }
            let _status = self.ctrbc_event_send.send(broadcast_output).await;
        }
    }

    pub async fn handle_output_delivery_ctrbc(&mut self, _instance_id: usize, sender: Replica, output_value: Vec<u8>){
        log::info!("Received CTRBC output from party {}",sender);
        if output_value.len() == 0{
            log::error!("Received empty CTRBC output from party {}",sender);
            return;
        }
        log::info!("Party {} successfully reconstructed output wires", sender);
        self.mult_state.output_layer.broadcasted_masked_outputs.insert(sender, output_value);
        if self.mult_state.output_layer.broadcasted_masked_outputs.len() == self.num_nodes-self.num_faults{
            // TODO: Add coins for coin hybrid model
            let coins: Vec<consensus::LargeFieldSer> = (0..crate::context::NUM_CONSENSUS_COINS).into_iter().map(|x| consensus::LargeField::from(x as u64).to_bytes_be()).collect();
            let parties_output: Vec<usize> = self.mult_state.output_layer.broadcasted_masked_outputs.keys().into_iter().map(|x| x.clone()).collect();
            let parties_ser_output = bincode::serialize(&parties_output).unwrap();
            let _status = self.acs_2_event_send.send((1, parties_ser_output, coins)).await;        
        }
        self.verify_masked_output_reconstruction_termination().await;
    }

    pub async fn handle_prot_end_ba_output(&mut self, acs_vec: Vec<u8>){
        let acs_vec: Vec<usize> = bincode::deserialize(&acs_vec).unwrap();
        self.mult_state.output_layer.acs_output.extend(acs_vec.clone());
        self.verify_masked_output_reconstruction_termination().await;
    }

    pub async fn verify_masked_output_reconstruction_termination(&mut self){
        if self.mult_state.output_layer.acs_output.is_empty(){
            return;
        }
        else{
            let mut output_hash_set = HashSet::new();
            let mut successful_parties = 0;
            let mut aborted_parties = 0;
            let acs_vec = self.mult_state.output_layer.acs_output.clone();
            
            for party in acs_vec{
                if self.mult_state.output_layer.broadcasted_masked_outputs.contains_key(&party){
                    let output = self.mult_state.output_layer.broadcasted_masked_outputs.get(&party).unwrap().clone();
                    let output_hash = do_hash(&output);
                    output_hash_set.insert(output_hash);

                    if output[0] == 1u8{
                        successful_parties += 1;
                    }
                    else{
                        aborted_parties += 1;
                    }
                }
            }

            if successful_parties >= self.num_faults +1{
                log::info!("MPC terminated successfully, t+1 parties have reconstructed the output");
                log::info!("Reconstructing random mask using AVSS and public reconstruction");
                self.reconstruct_random_masks().await;
            }
            if aborted_parties >= self.num_faults +1{
                log::error!("MPC terminated with Abort, t+1 parties have aborted the protocol");
                return;
            }
        }
    }
}