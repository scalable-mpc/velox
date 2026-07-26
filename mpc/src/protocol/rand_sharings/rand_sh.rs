use application::Application;
use std::{collections::{HashMap, HashSet}, ops::{Add, Mul}};

use fields::ByteConversion;
use fields::{LargeField, LargeFieldSer, rand_field_element};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use types::{ProtSyncMsg, Replica, SyncMsg, SyncState};
use crate::{context::Context};

/// ACSS batch carrying the `r` values that get squared into random bits.
pub const RAND_BIT_ACSS_BATCH: usize = 0;
/// ACSS batch carrying the masks consumed by multiplication gates, coins and
/// verification.
pub const MULT_ACSS_BATCH: usize = 1;
/// Number of ACSS batches each party deals during preprocessing.
pub const NUM_ACSS_BATCHES: usize = 2;

/// Sh2t batch carrying the 2t-sharings of zero.
pub const ZERO_SH2T_BATCH: usize = 0;
/// Number of Sh2t batches each party deals during preprocessing.
pub const NUM_SH2T_BATCHES: usize = 1;

impl<A: Application> Context<A>{
    pub async fn init_rand_sh(&mut self){
        // How much preprocessing the circuit needs is the application's call.
        let counts = self.app.preprocessing_count();
        if counts.net_route.is_some(){
            log::warn!("Application asked for network routing preprocessing, but Velox has no network routing module; ignoring it");
        }

        let num_mult_gates = counts.simd_mult;
        let num_rand_bits = counts.rand_bits;

        let t = self.num_faults;
        // Combining the ACS-agreed dealers' contributions through the Vandermonde
        // matrix turns each raw value a party deals into `t+1` random sharings.
        let sharings_per_value = t + 1;
        let batch_size_for = |sharings: usize| sharings.div_ceil(sharings_per_value);

        // `init_linear_multiplication_prot` pads every batch it is handed up to a
        // multiple of 2t+1, and the padding consumes masks like real gates do.
        // Each circuit depth is one batch, the rand-bit squaring is another, and
        // the handover to the application rounds up once more.
        let mult_padding = (counts.depth + 2) * (2*t + 1);

        // Verification multiplies its own tuples, and those multiplications draw
        // from the same pool. Compression divides the tuple count by
        // `compression_factor` per level, each level being one padded batch.
        let compression_factor = self.compression_factor.max(2);
        let mut compression_levels = 1;
        let mut remaining_tuples = num_mult_gates + num_rand_bits;
        while remaining_tuples > 1 {
            remaining_tuples /= compression_factor;
            compression_levels += 1;
        }
        // Two extra masks: the random beaver mask `delinearize_mult_tuples` pops.
        let verification_gates = compression_levels * (2*t + 1) + 2;

        // Random sharings, by consumer:
        //  - one mask per multiplication gate,
        //  - two per random bit — the `r` that gets squared (dealt in batch 0)
        //    and the mask for that squaring multiplication (batch 1),
        //  - the coin sharings and the verification multiplications.
        self.rand_bit_batch_size = batch_size_for(num_rand_bits + (2*t + 1));
        self.mult_batch_size = batch_size_for(
            num_mult_gates
            + num_rand_bits
            + mult_padding
            + verification_gates
            + self.total_sharings_for_coins
        );
        // Zero sharings: half a sharing per multiplication gate and half per
        // random bit, since the linear protocol draws `t+1` zeros per group of
        // 2t+1 gates.
        self.zero_batch_size = batch_size_for(
            (num_mult_gates + num_rand_bits + mult_padding + verification_gates).div_ceil(2)
        );
        // AVSS masks blind the output wires before public reconstruction.
        self.output_mask_size = batch_size_for(counts.output) + 1;

        log::info!(
            "Preprocessing sized from the application: {} multiplication gates, {} random bits, \
             {} output wires over {} depths -> ACSS batches of {} (rand bit) and {} (mult) values, \
             {} zero values, {} output masks per party",
            num_mult_gates, num_rand_bits, counts.output, counts.depth,
            self.rand_bit_batch_size, self.mult_batch_size, self.zero_batch_size, self.output_mask_size
        );

        // Start ACSS with abort and 2t-sharing simultaneously for each batch.
        let acss_batch_sizes = [
            (RAND_BIT_ACSS_BATCH, self.rand_bit_batch_size),
            (MULT_ACSS_BATCH, self.mult_batch_size),
        ];
        for (index, batch_size) in acss_batch_sizes.into_iter(){
            log::info!("Initiating secret sharing in preprocessing phase for batch {} with {} values", index, batch_size);
            let batch: Vec<LargeFieldSer> = (0..batch_size).into_par_iter()
                .map(|_| rand_field_element().to_bytes_be())
                .collect();
            let status = self.acss_ab_send.send((index,batch)).await;
            if status.is_err(){
                log::error!("Failed to send random values to ACSS protocol for batch {} because of error: {:?}", index, status.err().unwrap());
            }
        }

        // Initiate input sharing module as well. The application decides what
        // this party contributes to the circuit's input wires.
        self.initialize_input_sharing().await;

        log::info!("Initiating 2t sharing in preprocessing phase with {} values", self.zero_batch_size);
        let zeros: Vec<LargeFieldSer> = (0..self.zero_batch_size).into_par_iter()
            .map(|_| LargeField::zero().to_bytes_be())
            .collect();
        let status = self.sh2t_send.send((ZERO_SH2T_BATCH,zeros)).await;
        if status.is_err(){
            log::error!("Failed to send random values to Sh2t protocol for batch {} because of error: {:?}", ZERO_SH2T_BATCH, status.err().unwrap());
        }

        // Random masks for output wires
        let mut random_masks = Vec::new();
        for _ in 0..self.output_mask_size{
            random_masks.push(rand_field_element().to_bytes_be());
        }
        let avss_status = self.avss_send.send((true, Some(random_masks), None)).await;
        if avss_status.is_err(){
            log::error!("Failed to send random values to AVSS protocol {:?}", avss_status.err().unwrap());
        }
    }

    pub async fn handle_acss_term_msg(&mut self, instance: usize, sender: usize, shares: Option<Vec<LargeFieldSer>>){
        log::info!("Received ACSS shares from sender {} for batch {}", sender, instance);
        if shares.is_none(){
            log::error!("Abort ACSS protocol of dealer {} and terminate MPC", sender);
            return;
        }
        
        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            log::info!("Finished processing random sharings, ignoring ACSS and SH2t for all subsequent batches and senders: sender {}", sender);
            return;
        }

        let shares_deser: Vec<LargeField> = shares.unwrap().into_par_iter().map(|x| 
            LargeField::from_bytes_be(&x).unwrap()
        ).collect();

        if !self.rand_sharings_state.shares.contains_key(&sender){
            self.rand_sharings_state.shares.insert(sender, HashMap::default());
        }

        let shares_batches_map = self.rand_sharings_state.shares.get_mut(&sender).unwrap();
        shares_batches_map.insert(instance, shares_deser);

        self.verify_sender_termination(sender).await;
    }

    pub async fn handle_sh2t_term_msg(&mut self, instance: usize, sender: usize, shares: Option<Vec<LargeFieldSer>>){
        log::info!("Received Sh2t shares from sender {} for batch {}", sender, instance);
        if shares.is_none(){
            log::error!("Abort 2t-sharing protocol of dealer {} and terminate MPC", sender);
            return;
        }

        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            log::info!("Finished processing random sharings, ignoring ACSS and SH2t for all subsequent batches and senders: sender {}", sender);
            return;
        }
        let shares_deser: Vec<LargeField> = shares.unwrap().into_par_iter().map(|x| 
            LargeField::from_bytes_be(&x).unwrap()
        ).collect();

        if !self.rand_sharings_state.sh2t_shares.contains_key(&sender){
            self.rand_sharings_state.sh2t_shares.insert(sender, HashMap::default());
        }

        let shares_batches_map = self.rand_sharings_state.sh2t_shares.get_mut(&sender).unwrap();
        shares_batches_map.insert(instance, shares_deser);

        self.verify_sender_termination(sender).await;
    }

    pub async fn verify_sender_termination(&mut self, sender: usize){
        if !self.rand_sharings_state.shares.contains_key(&sender) || !self.rand_sharings_state.sh2t_shares.contains_key(&sender) || !self.output_mask_state.avss_shares.contains_key(&sender){
            log::debug!("ACSS, Sh2t, and AVSS not completed for sender {} for all batches", sender);
            return;
        }
        if !self.mix_circuit_state.input_acss_shares.contains_key(&sender){
            return;
        }
        if self.rand_sharings_state.acss_completed_parties.contains(&sender){
            log::debug!("ACSS, Sh2t, and AVSS already completed for sender {} for all batches", sender);
            return;
        }
        let shares_batches_map = self.rand_sharings_state.shares.get_mut(&sender).unwrap();
        let share_2t_batches_map = self.rand_sharings_state.sh2t_shares.get_mut(&sender).unwrap();
        let input_sharings = self.mix_circuit_state.input_acss_shares.get_mut(&sender).unwrap();
        if shares_batches_map.len() == NUM_ACSS_BATCHES &&
            share_2t_batches_map.len() == NUM_SH2T_BATCHES &&
            input_sharings.len() == 1 &&
            self.output_mask_state.avss_shares.contains_key(&sender){
            // ACSS is complete. Wait for sh2t sharings now
            log::info!("ACSS, ACSS Input, Sh2t, and AVSS completed for sender {} for all batches", sender);
            log::info!("Batches info: {:?} {:?}", shares_batches_map.keys(),share_2t_batches_map.keys());
            self.rand_sharings_state.acss_completed_parties.insert(sender);

            if self.rand_sharings_state.acss_completed_parties.len() == self.num_nodes-self.num_faults{
                let coins: Vec<consensus::LargeFieldSer> = (0..crate::context::NUM_CONSENSUS_COINS).into_iter().map(|x| consensus::LargeField::from(x as u64).to_bytes_be()).collect();
                let parties_set: Vec<usize> = self.rand_sharings_state.acss_completed_parties.clone().into_iter().collect();
                let ser_set = bincode::serialize(&parties_set).unwrap();
                let _status = self.acs_event_send.send((1, ser_set, coins)).await;
            }
            self.verify_termination().await;
        }
    }

    pub async fn handle_acs_output(&mut self, partyset: Vec<u8>){
        let deser_set: Vec<usize> = bincode::deserialize(&partyset).unwrap();
        self.rand_sharings_state.acs_output.extend(deser_set);
        // Check if all parties have completed ACSS and 2t-sharing
        self.verify_termination().await;
    }

    pub async fn verify_termination(&mut self){
        log::info!("Checking termination for random sharings");
        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            // Sharings already generated, return back
            return;
        } 
        if self.rand_sharings_state.acs_output.len() > 0{
            let mut flag = true;
            for party in self.rand_sharings_state.acs_output.clone().into_iter(){
                flag =  flag && self.rand_sharings_state.acss_completed_parties.contains(&party);
            }
            if flag{
                // All parties in the ACS state have completed ACSS and 2t-sharing
                // Generate random sharings
                // Vandermonde matrix
                
                let x_values: Vec<LargeField> = (2..self.num_faults+3).into_iter().map(|x| LargeField::from(x as u64)).collect();
                let vandermonde_matrix = Self::vandermonde_matrix(x_values, 2*self.num_faults+1);
                
                // Build party-accumulated share vectors
                let acs_indexed_rand_bit_group = self.gen_random_sharings(RAND_BIT_ACSS_BATCH, self.rand_bit_batch_size);
                let acs_indexed_mult_group = self.gen_random_sharings(MULT_ACSS_BATCH, self.mult_batch_size);

                let acs_indexed_2t_share_groups = self.gen_2t_sharings(ZERO_SH2T_BATCH, self.zero_batch_size);

                // Multiply each vector with the indexed vector in the Vandermonde matrix
                let rand_sharings_bits: Vec<LargeField> = acs_indexed_rand_bit_group.into_par_iter().map(|x| {
                    let res = Self::matrix_vector_multiply(&vandermonde_matrix, &x);
                    res
                }).flatten().collect();

                let mut rand_sharings_mult: Vec<LargeField> = acs_indexed_mult_group.into_par_iter().map(|x| {
                    let res = Self::matrix_vector_multiply(&vandermonde_matrix, &x);
                    res
                }).flatten().collect();

                let rand_sharings_2t_mult: Vec<LargeField> = acs_indexed_2t_share_groups.into_par_iter().map(|x| {
                    let res = Self::matrix_vector_multiply(&vandermonde_matrix, &x);
                    res
                }).flatten().collect();

                log::info!("Completed preprocessing and generated {} random sharings for random bits, {} random sharings for multiplication, and {} random 2t sharings",
                        rand_sharings_bits.len(), rand_sharings_mult.len(), rand_sharings_2t_mult.len());

                // Every random bit consumes one `r` from this batch, squared through
                // the multiplication protocol below.
                self.rand_sharings_state.rand_sharings_inputs = (rand_sharings_bits.clone(), rand_sharings_bits.clone());
                self.mix_circuit_state.rand_bit_inp_shares.extend(rand_sharings_bits);

                // Allocate 2n sharings to common coins
                let rand_sharings_coin =  rand_sharings_mult.split_off(rand_sharings_mult.len()- self.total_sharings_for_coins);
                
                // Add sharings and coins to state
                self.rand_sharings_state.rand_sharings_mult.extend(rand_sharings_mult);
                self.rand_sharings_state.rand_sharings_coin.extend(rand_sharings_coin);
                self.rand_sharings_state.rand_2t_sharings_mult.extend(rand_sharings_2t_mult);    
                
                // Clear acss sharings now
                self.rand_sharings_state.shares.clear();
                self.rand_sharings_state.sh2t_shares.clear();

                self.generate_random_mask_shares(self.rand_sharings_state.acs_output.clone(),vandermonde_matrix).await;
                self.init_random_shared_bits_preparation().await;
            }
        }
    }

    /// Constructs the Vandermonde matrix for a given set of x-values. Note that the x-values are parties and are converted to the ith root of unity for the evaluation
    pub fn vandermonde_matrix(x_values: Vec<LargeField>, y_vals_target: usize) -> Vec<Vec<LargeField>> {
        let n = x_values.len();
        let mut matrix = vec![vec![LargeField::zero(); y_vals_target]; n];

        for (row, x) in x_values.iter().enumerate() {
            let mut value = LargeField::one();
            for col in 0..y_vals_target {
                matrix[row][col] = value.clone();
                value = value * x;
            }
        }
        matrix
    }

    pub fn matrix_vector_multiply(
        matrix: &Vec<Vec<LargeField>>,
        vector: &Vec<LargeField>,
    ) -> Vec<LargeField> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(vector)
                    .fold(LargeField::zero(), |sum, (a, b)| sum.add(a.mul(b)))
            })
            .collect()
    }

    pub fn gen_random_sharings(&self, batch: usize, batch_size: usize)-> Vec<Vec<LargeField>>{
        Self::group_batch_by_position(
            &self.rand_sharings_state.shares,
            &self.rand_sharings_state.acs_output,
            self.num_nodes,
            batch,
            batch_size,
        )
    }

    pub fn gen_2t_sharings(&self, batch: usize, batch_size: usize) -> Vec<Vec<LargeField>>{
        Self::group_batch_by_position(
            &self.rand_sharings_state.sh2t_shares,
            &self.rand_sharings_state.acs_output,
            self.num_nodes,
            batch,
            batch_size,
        )
    }

    /// Group one batch's contributions by position: group `i` collects the
    /// `i`-th value dealt by every party in the ACS output set, ready to be
    /// combined through the Vandermonde matrix into `t+1` sharings.
    fn group_batch_by_position(
        shares_per_party: &HashMap<usize, HashMap<usize, Vec<LargeField>>>,
        acs_output: &HashSet<Replica>,
        num_nodes: usize,
        batch: usize,
        batch_size: usize,
    ) -> Vec<Vec<LargeField>>{
        let mut indexed_share_groups: Vec<Vec<LargeField>> = vec![Vec::new(); batch_size];
        for party in 0..num_nodes{
            if !acs_output.contains(&party){
                continue;
            }
            let Some(batches) = shares_per_party.get(&party) else {
                log::error!("No sharings recorded for party {} in the ACS output set", party);
                continue;
            };
            let Some(shares_batch) = batches.get(&batch) else {
                log::error!("Batch {} not found in sharings of party {}", batch, party);
                continue;
            };
            if shares_batch.len() < batch_size{
                log::error!("Party {} dealt {} values in batch {}, expected {}",
                    party, shares_batch.len(), batch, batch_size);
            }
            for (index, share) in shares_batch.iter().take(batch_size).enumerate(){
                indexed_share_groups[index].push(share.clone());
            }
        }
        indexed_share_groups
    }
    //Invoke this function once you terminate the protocol
    pub async fn terminate(&mut self, status: String, value: Vec<u8>) {
        let rbc_sync_msg = ProtSyncMsg{
            id: 1,
            status,
            value
        };

        let ser_msg = bincode::serialize(&rbc_sync_msg).unwrap();
        let cancel_handler = self
            .sync_send
            .send(
                0,
                SyncMsg {
                    sender: self.myid,
                    state: SyncState::COMPLETED,
                    value: ser_msg,
                },
            )
            .await;
        self.add_cancel_handler(cancel_handler);
    }
}