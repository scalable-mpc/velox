use crate::Context;
use crypto::{hash::{do_hash, Hash}};
use fields::{ser_be_flat, generate_evaluation_points_fft, generate_evaluation_points, generate_evaluation_points_opt, sample_polynomials_from_prf, rand_field_element, ProtocolField, FieldSer};
use types::Replica;
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField> Context<F>{
    pub async fn init_sh2t(&mut self, secrets: Vec<FieldElement<F>>, instance_id: usize){
        if !self.sh2t_state_map.contains_key(&instance_id){
            let sh2t_state = crate::Sh2tState::new();
            self.sh2t_state_map.insert(instance_id, sh2t_state);
        }

        let tot_sharings = secrets.len();

        let mut _indices;
        let mut evaluations;
        let nonce_evaluations;
        let mut coefficients;
        
        if !self.use_fft{
            // Generate evaluations right here
            let evaluations_prf = sample_polynomials_from_prf(
                secrets, 
                self.sec_key_map.clone(), 
                2*self.num_faults, 
                false, 
                1u8
            );

            evaluations = Vec::new();
            coefficients = Vec::new();

            let (evaluations_batch, coefficients_batch) = generate_evaluation_points_opt(
                evaluations_prf,
                2*self.num_faults,
                self.num_nodes,
            ).await;

            evaluations.extend(evaluations_batch);
            coefficients.extend(coefficients_batch);
            _indices = Vec::new();
            for party in 0..self.num_nodes{
                _indices.push(FieldElement::<F>::from((party+1) as u64));
            }

            // Generate nonce evaluations
            let evaluations_nonce_prf = sample_polynomials_from_prf(
                vec![rand_field_element::<F>()], 
                self.sec_key_map.clone(), 
                2*self.num_faults, 
                true, 
                1u8
            );
            let (nonce_evaluations_ret,_nonce_coefficients) = generate_evaluation_points(
                evaluations_nonce_prf,
                2*self.num_faults,
                self.num_nodes
            ).await;
            nonce_evaluations = nonce_evaluations_ret[0].clone();
        }
        else{
            // Parallelize the generation of evaluation points
            let (evaluations_batch, coefficients_batch) = generate_evaluation_points_fft(
                secrets,
                2*self.num_faults-1,
                self.num_nodes
            ).await;
            evaluations = Vec::new();
            coefficients = Vec::new();
            _indices = self.roots_of_unity.clone();

            evaluations.extend(evaluations_batch);
            coefficients.extend(coefficients_batch);
            
            // Generate nonce evaluations
            let (nonce_evaluations_ret,_nonce_coefficients) = generate_evaluation_points_fft(
                vec![rand_field_element::<F>()],
                2*self.num_faults-1,
                self.num_nodes,
            ).await;
            nonce_evaluations = nonce_evaluations_ret[0].clone();
        }
        // Transform the shares to element wise shares
        // The shares of each party as one flat buffer, and the n commitments:
        // the hash of the party's flat shares followed by its nonce share.
        let mut party_wise_shares: Vec<Vec<u8>> = Vec::new();
        let mut commitments: Vec<Hash> = Vec::new();
        for i in 0..self.num_nodes{
            let mut flat_shares = ser_be_flat(evaluations.iter().map(|party_evaluations| &party_evaluations[i]));
            let shares_len = flat_shares.len();
            flat_shares.extend(nonce_evaluations[i].ser_be());
            commitments.push(do_hash(&flat_shares));
            flat_shares.truncate(shares_len);
            party_wise_shares.push(flat_shares);
        }

        let broadcast_vec = (commitments, tot_sharings);
        let inst_broadcast_vec = (instance_id, broadcast_vec);
        let ser_vec = bincode::serialize(&inst_broadcast_vec).unwrap();

        let mut shares: Vec<(Replica,Option<Vec<u8>>)> = Vec::new();
        for (rep, shares_party) in party_wise_shares.into_iter().enumerate(){
            // prepare shares
            if (self.use_fft) || (!self.use_fft && rep >= 2*self.num_faults){
                let nonce_share = nonce_evaluations[rep].clone().ser_be();
                
                let shares_full = (shares_party, nonce_share);
                let inst_shares_full = (instance_id, shares_full);
                let shares_ser = bincode::serialize(&inst_shares_full).unwrap();

                //let enc_shares = encrypt(sec_key.as_slice(), shares_ser);
                shares.push((rep, Some(shares_ser)));
            }
        }
        // Reliably broadcast this vector
        let _rbc_status = self.inp_ctrbc.send(ser_vec).await;
        
        // Invoke AVID on vectors of shares
        // Use AVID to send the shares to parties
        let _avid_status = self.inp_avid_channel.send(shares).await;
    }

    pub async fn verify_shares(&mut self, sender: Replica, instance_id: usize){
        if !self.sh2t_state_map.contains_key(&instance_id) {
            let sh2t_state = crate::Sh2tState::new();
            self.sh2t_state_map.insert(instance_id, sh2t_state);
        }
        let sh2t_state = self.sh2t_state_map.get(&instance_id).unwrap();

        if sh2t_state.status.contains(&sender){
            // Dealer already terminated and its state was cleared; ignore.
            return;
        }
        if sh2t_state.verification_status.contains_key(&sender){
            // Already verified status, abandon sharing
            return;
        }

        if !sh2t_state.commitments.contains_key(&sender) || !sh2t_state.shares.contains_key(&sender){
            // AVID and CTRBC did not yet terminate
            return;
        }

        let (shares, nonce_share) = sh2t_state.shares.get(&sender).unwrap();
        let share_commitments = sh2t_state.commitments.get(&sender).unwrap();
        
        // First, verify share commitments
        let mut appended_share = Vec::with_capacity(shares.len() + nonce_share.len());
        appended_share.extend_from_slice(shares);
        appended_share.extend_from_slice(nonce_share);
        let comm_hash = do_hash(appended_share.as_slice());
        drop(appended_share);
        let commitment_matches = comm_hash == share_commitments[self.myid];
        let sh2t_state = self.sh2t_state_map.get_mut(&instance_id).unwrap();
        if !commitment_matches{
            // Invalid share commitments
            log::error!("Invalid share commitments from {}", sender);
            sh2t_state.verification_status.insert(sender, false);
            return;
        }
        
        log::info!("Share from {} verified", sender);
        // If successful, add to verified list
        sh2t_state.verification_status.insert(sender,true);
        // Start reliable agreement
        let _status = self.inp_ra_channel.send((sender,1,instance_id)).await;
        self.check_termination(sender, instance_id).await;
    }
}