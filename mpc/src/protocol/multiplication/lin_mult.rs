use application::Application;
use std::{collections::HashMap, ops::Add};

use crate::Context;

use bincode::{Result};
use crypto::hash::do_hash;
use lambdaworks_math::{polynomial::Polynomial};
use fields::ByteConversion;
use fields::{LargeField, LargeFieldSer, vandermonde_matrix, inverse_vandermonde, matrix_vector_multiply, matrix_matrix_multiply, powers_matrix};
use rayon::prelude::{ ParallelIterator, IntoParallelRefIterator};
use types::{Replica, WrapperMsg};

use crate::{msg::ProtMsg};

impl<A: Application> Context<A>{
    pub async fn init_linear_multiplication_prot(&mut self,
        mut a_vec_shares: Vec<Vec<LargeField>>,
        mut b_vec_shares: Vec<Vec<LargeField>>,
        depth: usize,
        mut rand_sharings: Vec<LargeField>,
        mut zero_sharings: Vec<LargeField>
    ) {
        // Pad shares until they become a multiple of 2t+1
        // Share inputs for later verification
        if self.is_verified_depth(depth) {
            // Only the first sharing of each gate is verified, so read it out by
            // reference: cloning `a_vec_shares` wholesale duplicated every
            // inner-product operand as well.
            let first_a_shares: Vec<LargeField> = a_vec_shares.iter().map(|x| x[0].clone()).collect();
            let first_b_shares: Vec<LargeField> = b_vec_shares.iter().map(|x| x[0].clone()).collect();
            log::info!("Adding shares to verification state with a:{} b:{} at depth {}", first_a_shares.len(), first_b_shares.len(), depth);
            self.verf_state.add_mult_inputs(depth, first_a_shares, first_b_shares);
        }
        
        let multiple_of_val = 2*self.num_faults+1;
        let mut padding_length = multiple_of_val - (a_vec_shares.len()%multiple_of_val);
        if (a_vec_shares.len()%multiple_of_val) == 0{
            padding_length =0;
        }
        // Pad the shares until it becomes a multiple of 2t+1
        for _ in 0..padding_length{
            a_vec_shares.push(vec![LargeField::zero()]);
            b_vec_shares.push(vec![LargeField::zero()]);
        }
        if a_vec_shares.len()%multiple_of_val != 0{
            
        }
        let tot_groups = a_vec_shares.len() / (2 * self.num_faults + 1);
        // Use linear multiplication protocol here
        let tot_shares = a_vec_shares.len();
        
        let depth_state = self.mult_state.get_single_depth_state(depth, true, tot_groups);
        // A `HashZMsg` that reached us first created this entry with
        // `two_levels = false` (the flag on the wire is always `false`), which
        // would make termination read `l1_shares_reconstructed` instead of the
        // L2 output. The party running the linear protocol at this depth is the
        // authority on which protocol it is.
        depth_state.two_levels = true;
        depth_state.padding_shares = padding_length;

        // Take the masks this batch consumes: one per padded gate, and t+1 zero
        // sharings per group of 2t+1 gates.
        let num_zero_sharings = tot_groups*(self.num_faults+1);
        if rand_sharings.len() < tot_shares {
            log::error!("Not enough random shares for linear multiplication protocol at depth {}: need {}, got {}",
                depth, tot_shares, rand_sharings.len());
            return;
        }
        if zero_sharings.len() < num_zero_sharings {
            log::error!("Not enough random shares for zero multiplication protocol at depth {}: need {}, got {}",
                depth, num_zero_sharings, zero_sharings.len());
            return;
        }
        rand_sharings.truncate(tot_shares);
        zero_sharings.truncate(num_zero_sharings);

        let r_sharings = rand_sharings;
        let o_sharings = zero_sharings;
        depth_state.util_rand_sharings.extend(r_sharings.iter().cloned());

        // Chunk boundaries are read straight out of the flat input vectors.
        // Materialising `a/b/r/o_..._grouped` with `chunks(..).map(|x| x.to_vec())`
        // duplicated every operand of the batch — for inner-product gates that is
        // the entire input a second time — purely so it could be indexed as
        // `[chunk][k]`. The arithmetic below is unchanged: chunk `i` element `k`
        // is flat index `i*(2t+1) + k`, and the zero sharings chunk by `t+1`.
        // Padding above guarantees `a_vec_shares.len() == tot_groups*(2t+1)`.
        let group_size = 2*self.num_faults + 1;
        let zero_group_size = self.num_faults + 1;
        let total_chunks = tot_groups;

        // Check that there are the correct number of groups

        let vandermonde_points: Vec<LargeField> = (2..self.num_nodes+2).into_iter().map(|x| LargeField::from(x as u64)).collect();
        let vdm_matrix = Self::vandermonde_matrix(vandermonde_points, self.num_faults); // TODO: can initialize the vdm_matrix somewhere outside to not compute it each time this gets called

        // Build every chunk's z_vector and o_vec first, then do ONE GEMM across all
        // chunks to evaluate them at the n party points. Per-chunk GEMM lost ~6× to
        // the scalar loop because each call paid Rayon setup overhead for a tiny
        // 16×11 product; bench `BatchedPartyEval` characterizes the right shape.
        let z_vector_len = 2 * self.num_faults + 1;
        let party_powers = powers_matrix(&self.roots_of_unity, z_vector_len);

        let mut z_vectors: Vec<Vec<LargeField>> = Vec::with_capacity(total_chunks);
        let mut o_vecs: Vec<Vec<LargeField>> = Vec::with_capacity(total_chunks);
        for i in 0..total_chunks {
            o_vecs.push(Self::matrix_vector_multiply(
                &vdm_matrix,
                &o_sharings[i*zero_group_size..(i+1)*zero_group_size],
            ));
            let mut z_vector = Vec::with_capacity(z_vector_len);
            for k in 0..=(2 * self.num_faults) {
                let a: &Vec<LargeField> = &a_vec_shares[i*group_size + k];
                let b: &Vec<LargeField> = &b_vec_shares[i*group_size + k];
                z_vector.push(Self::dot_product(a, b).add(r_sharings[i*group_size + k].clone()));
            }
            z_vectors.push(z_vector);
        }
        // The batch's operands and preprocessing are dead once every chunk's
        // `z_vector` has been formed: from here on only `z_vectors` and `o_vecs`
        // are read, and this depth's masks already live in `util_rand_sharings`.
        drop(a_vec_shares);
        drop(b_vec_shares);
        drop(r_sharings);
        drop(o_sharings);

        // One GEMM: party_powers (n × (2t+1)) · z_vectors (chunks vectors of length (2t+1))
        // → evals (n × chunks). evals[p][chunk] is the share for party `p` in chunk `chunk`,
        // pre-`o_vec` add. Replaces the previous per-chunk `Polynomial::new(&z).evaluate(&el)`
        // loop; `Polynomial::new`'s trailing-zero trim is a no-op for correctness here since
        // zero coefficients contribute zero to the GEMM dot product.
        let evals = matrix_matrix_multiply(&party_powers, &z_vectors, true);

        let mut shares_party: HashMap<usize, Vec<LargeField>> = HashMap::default();
        for party in 0..self.num_nodes {
            shares_party.insert(party, Vec::with_capacity(tot_shares));
        }
        for i in 0..total_chunks {
            for p in 0..self.num_nodes {
                let share = evals[p][i].clone() + o_vecs[i][p].clone();
                shares_party.get_mut(&p).unwrap().push(share);
            }
        }

        // Send shares for all groups to all parties
        for (party,shares) in shares_party.into_iter(){
            let ser_shares: Vec<LargeFieldSer> = shares.into_iter().map(|share| {
                share.to_bytes_be()
            }).collect();
            // Encrypt shares before putting them in a message
            let ser_shares_bytes = bincode::serialize(&ser_shares).unwrap();
            let sec_key = self.sec_key_map.get(&party).clone().unwrap();

            // let encrypted_msg = encrypt(sec_key, ser_shares_bytes);
            let prot_msg = ProtMsg::SharesL1(ser_shares_bytes, depth);

            let wrapper_msg = WrapperMsg::new(prot_msg, self.myid, &sec_key);
            let cancel_handler = self.net_send.send(party, wrapper_msg).await;

            self.add_cancel_handler(cancel_handler);
        }
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_l1_message(&mut self, ser_shares: Vec<u8>, depth: usize, sender: usize) {
        // Try deserializing the message now
        log::info!("Received L1 multiplication shares from party {} for depth {}", sender, depth);
        let shares_option: Result<Vec<LargeFieldSer>> = bincode::deserialize(&ser_shares);
        if shares_option.is_err() {
            log::error!("Error deserializing shares: {:?}", shares_option.err());
            return;
        }

        let shares_ser = shares_option.unwrap();
        
        // Received message as L1 share so multiplication at this depth must be linear
        
        let shares: Vec<LargeField> = shares_ser.into_iter().map(|share| {
            return LargeField::from_bytes_be(&share).unwrap();
        }).collect();

        let depth_state = self.mult_state.get_single_depth_state(depth, true, shares.len());
        // If this depth already terminated, its per-party share buffers were
        // freed by `clear_shares()` (l1_shares.1 is now empty). Likewise, once
        // the L1 interpolation has run, `clear_l1_shares()` freed them. Either
        // way a late L1 share is dead - dropping it here avoids indexing the
        // emptied l1_shares.1 out of bounds below and keeps
        // recv_share_count_l1 / l1_shares.0 consistent.
        if depth_state.depth_terminated || depth_state.l1_reconstruction_done {
            return;
        }
        // At L1, the evaluation point is the point at which the polynomials have been evaluated.
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft.clone(), self.roots_of_unity.clone());
        depth_state.l1_shares.0.push(evaluation_point);
        for (index, share) in shares.into_iter().enumerate(){
            depth_state.l1_shares.1[index].push(share);
        }
        
        depth_state.recv_share_count_l1 +=1;
        //depth_state.recv_share_count_l1 = depth_state.recv_share_count_l1.clone().add(1).into();
        let mut ser_shares = None;
        if depth_state.recv_share_count_l1 == self.num_nodes - self.num_faults {
            log::info!("Attempting L1 reconstruction at depth {}", depth);
            // Claim the interpolation before doing it, so a share that lands
            // afterwards is dropped by the guard above instead of piling onto
            // buffers nobody reads again.
            depth_state.l1_reconstruction_done = true;
            // Start reconstruction here
            let indices = depth_state.l1_shares.0.clone();
            let vdm_matrix = vandermonde_matrix(indices);

            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);
            let secrets: Vec<LargeField> = depth_state.l1_shares.1.par_iter().map(|group_shares|{
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &group_shares);
                let poly = Polynomial::new(&coefficients);
                let secret = poly.evaluate(&LargeField::zero()); // Evaluate at zero to get the secret
                return secret;
            }).collect();

            // The raw per-party L1 shares are dead: this interpolation is their
            // only reader, it runs exactly once, and its result is what the rest
            // of the depth uses. Freeing them here rather than at termination
            // takes the whole L2 round-trip - during which they are the largest
            // live buffer of the depth - off the peak.
            depth_state.clear_l1_shares();

            let shares_bytes: Vec<LargeFieldSer> = secrets.iter().map(|el| el.to_bytes_be()).collect();
            depth_state.l1_shares_reconstructed.extend(secrets);
            ser_shares = Some(bincode::serialize(&shares_bytes).unwrap());
        }

        if ser_shares.is_some(){
            log::info!("L1 reconstruction successful, sending L2 shares to all parties");
            self.broadcast(ProtMsg::SharesL2(ser_shares.unwrap(), depth)).await;
        }
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_l2_message(&mut self, group_shares: Vec<u8>, depth: usize, sender: Replica){
        // Multiplication at this depth is of course using two levels of mult
        log::info!("Received L2 multiplication shares from party {} for depth {}", sender, depth);
        let group_shares: Vec<LargeFieldSer> = bincode::deserialize(&group_shares).unwrap();
        
        let depth_state = self.mult_state.get_single_depth_state(depth, true, group_shares.len());

        // If this depth already terminated, `clear_shares()` emptied l2_shares,
        // and once the L2 interpolation has run `clear_l2_shares()` did the
        // same, so the per-group inner vectors are gone. A late L2 share is
        // dead; drop it to avoid corrupting l2_shares.0 / recv_share_count_l2
        // against an empty l2_shares.1.
        if depth_state.depth_terminated || depth_state.l2_reconstruction_done {
            return;
        }
        // At this depth, we are using roots of unity to conduct evaluation
        let evaluation_point = self.roots_of_unity.get(sender).clone().unwrap();
        depth_state.l2_shares.0.push(evaluation_point.clone());
        for (state,group_share) in depth_state.l2_shares.1.iter_mut().zip(group_shares.into_iter()){
            let group_lf_share = LargeField::from_bytes_be(&group_share).unwrap();
            state.push(group_lf_share); // Store the share itself
        }

        depth_state.recv_share_count_l2 +=1;
        // Interpolate polynomial
        // Idempotence satisfied here
        if depth_state.recv_share_count_l2 == self.num_nodes - self.num_faults{
            log::info!("Attempting L2 reconstruction at depth {}", depth);
            // Claimed before the work, so later L2 shares hit the guard above.
            depth_state.l2_reconstruction_done = true;
            // We have enough shares to reconstruct the polynomial
            let indices = depth_state.l2_shares.0.clone();
            let vdm_matrix = vandermonde_matrix(indices);

            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);

            let reconstructed_secrets: Vec<LargeField> = depth_state.l2_shares.1.par_iter().map(|group_shares|{
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &group_shares);
                coefficients
            }).flatten().collect();

            // Same argument as L1: this interpolation is the only reader of the
            // raw L2 shares and runs once. What the depth needs from here on is
            // `l2_shares_reconstructed`.
            depth_state.clear_l2_shares();

            let mut appended_msg = Vec::new();
            for secret in reconstructed_secrets.iter(){
                appended_msg.extend(secret.to_bytes_be());
            }
            depth_state.l2_shares_reconstructed.extend(reconstructed_secrets);
            let hash = do_hash(&appended_msg);
            log::info!("Completed processing triples at depth {} with linear sharings, broadcasting hash {:?}", depth, hash);
            self.init_hash_broadcast(hash, depth).await;
            self.verify_depth_mult_termination(depth).await;
        }
    }
}