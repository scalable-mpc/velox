use application::Application;
use std::ops::Mul;

use crypto::hash::{do_hash, Hash};
use lambdaworks_math::polynomial::Polynomial;
use fields::ByteConversion;
use fields::mersenne_61::Sqrt;
use fields::{LargeFieldSer, LargeField, vandermonde_matrix, inverse_vandermonde, matrix_vector_multiply, matrix_matrix_multiply, powers_matrix};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use types::{Replica, WrapperMsg};

use crate::{Context, msg::ProtMsg};

impl<A: Application> Context<A>{
    /// Publicly reconstruct the squared random sharings.
    ///
    /// Broadcasting every share costs O(n²) elements per value. Instead, mirror
    /// the linear multiplication protocol: split the sharings into chunks of
    /// 2t+1, read each chunk as the coefficients of a degree-2t polynomial `Z`,
    /// and send every party `p` this party's share of `Z(α_p)`. Each party then
    /// reconstructs its own point privately (L1) and broadcasts it (L2), so
    /// 2t+1 broadcast points interpolate `Z` and hand everyone all 2t+1 values
    /// of the chunk at once.
    pub async fn init_rand_bit_reconstruction(&mut self){
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }
        if !self.mix_circuit_state.rand_bit_recon_shares.contains_key(&self.myid){
            return;
        }
        let chunk_size = 2*self.num_faults + 1;
        let mut my_shares = self.mix_circuit_state.rand_bit_recon_shares.get(&self.myid).unwrap().clone();

        // Pad with shares of zero so the batch splits into whole chunks. A
        // degree-t sharing of zero is the all-zero share vector, so every party
        // pads identically.
        let padding = (chunk_size - (my_shares.len() % chunk_size)) % chunk_size;
        my_shares.extend(vec![LargeField::zero(); padding]);
        let num_chunks = my_shares.len()/chunk_size;

        log::info!("Initializing random bit reconstruction of {} sharings over {} chunks of {} ({} padded)",
            my_shares.len() - padding, num_chunks, chunk_size, padding);
        self.mix_circuit_state.rand_bit_recon_state.init_chunks(num_chunks);
        // Set independently of the chunk count: a peer's message may have sized
        // the chunks already, but only this party's own batch knows the padding.
        self.mix_circuit_state.rand_bit_recon_state.padding = padding;

        // Evaluate every chunk polynomial at all n party points in one GEMM:
        // evals[p][chunk] is this party's share of Z_chunk(α_p).
        let chunks: Vec<Vec<LargeField>> = my_shares.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect();
        let party_powers = powers_matrix(&self.roots_of_unity, chunk_size);
        let evals = matrix_matrix_multiply(&party_powers, &chunks, true);

        for party in 0..self.num_nodes{
            let shares_ser: Vec<LargeFieldSer> = evals[party].iter().map(|share| share.to_bytes_be()).collect();
            let ser_shares_bytes = bincode::serialize(&shares_ser).unwrap();
            let sec_key = self.sec_key_map.get(&party).clone().unwrap();

            let prot_msg = ProtMsg::RandBitReconL1(ser_shares_bytes);
            let wrapper_msg = WrapperMsg::new(prot_msg, self.myid, &sec_key);
            let cancel_handler = self.net_send.send(party, wrapper_msg).await;
            self.add_cancel_handler(cancel_handler);
        }
    }

    /// L1: shares of this party's point on each chunk polynomial. With n-t of
    /// them the point itself is reconstructed and broadcast.
    pub async fn handle_rand_bit_recon_l1(&mut self, ser_shares: Vec<u8>, sender: Replica){
        log::debug!("Received L1 random bit reconstruction shares from party {}", sender);
        let shares_ser: Vec<LargeFieldSer> = match bincode::deserialize(&ser_shares){
            Ok(shares) => shares,
            Err(e) => {
                log::error!("Error deserializing L1 random bit reconstruction shares: {:?}", e);
                return;
            }
        };
        let shares: Vec<LargeField> = shares_ser.into_iter()
            .map(|share| LargeField::from_bytes_be(&share).unwrap())
            .collect();

        let num_nodes = self.num_nodes;
        let num_faults = self.num_faults;
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());

        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || !recon_state.l1_reconstructed.is_empty(){
            return;
        }
        // The sender chunked the same batch we did, so it fixes the chunk count
        // for parties whose own squaring output has not arrived yet.
        recon_state.init_chunks(shares.len());
        recon_state.l1_shares.0.push(evaluation_point);
        for (index, share) in shares.into_iter().enumerate(){
            recon_state.l1_shares.1[index].push(share);
        }
        recon_state.recv_share_count_l1 += 1;

        if recon_state.recv_share_count_l1 != num_nodes - num_faults{
            return;
        }

        log::info!("Attempting L1 reconstruction of random bit sharings");
        let inv_vdm_matrix = inverse_vandermonde(vandermonde_matrix(recon_state.l1_shares.0.clone()));
        let my_points: Vec<LargeField> = recon_state.l1_shares.1.par_iter()
            .map(|chunk_shares|{
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, chunk_shares);
                Polynomial::new(&coefficients).evaluate(&LargeField::zero())
            })
            .collect();
        recon_state.l1_reconstructed.extend(my_points.clone());

        let points_ser: Vec<LargeFieldSer> = my_points.iter().map(|point| point.to_bytes_be()).collect();
        let ser_points = bincode::serialize(&points_ser).unwrap();
        self.broadcast(ProtMsg::RandBitReconL2(ser_points)).await;
    }

    /// L2: the points other parties reconstructed. n-t of them interpolate each
    /// chunk polynomial, whose coefficients are the reconstructed values.
    pub async fn handle_rand_bit_recon_l2(&mut self, ser_points: Vec<u8>, sender: Replica){
        log::debug!("Received L2 random bit reconstruction shares from party {}", sender);
        let points_ser: Vec<LargeFieldSer> = match bincode::deserialize(&ser_points){
            Ok(points) => points,
            Err(e) => {
                log::error!("Error deserializing L2 random bit reconstruction shares: {:?}", e);
                return;
            }
        };
        let points: Vec<LargeField> = points_ser.into_iter()
            .map(|point| LargeField::from_bytes_be(&point).unwrap())
            .collect();

        let num_nodes = self.num_nodes;
        let num_faults = self.num_faults;
        // L1 evaluated the chunk polynomials at these same points.
        let evaluation_point = self.roots_of_unity.get(sender).unwrap().clone();

        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || !recon_state.l2_reconstructed.is_empty(){
            return;
        }
        recon_state.init_chunks(points.len());
        recon_state.l2_shares.0.push(evaluation_point);
        for (index, point) in points.into_iter().enumerate(){
            recon_state.l2_shares.1[index].push(point);
        }
        recon_state.recv_share_count_l2 += 1;

        if recon_state.recv_share_count_l2 != num_nodes - num_faults{
            return;
        }

        log::info!("Attempting L2 reconstruction of random bit sharings");
        let inv_vdm_matrix = inverse_vandermonde(vandermonde_matrix(recon_state.l2_shares.0.clone()));
        // Interpolating 2t+1 points of a degree-2t polynomial gives back its
        // coefficients, which are the chunk's values.
        let mut reconstructed_values: Vec<LargeField> = recon_state.l2_shares.1.par_iter()
            .map(|chunk_points| matrix_vector_multiply(&inv_vdm_matrix, chunk_points))
            .flatten()
            .collect();
        for _ in 0..recon_state.padding{
            reconstructed_values.pop();
        }
        recon_state.l2_reconstructed.extend(reconstructed_values.clone());

        // Pin down what everyone reconstructed before deriving random bits from it.
        let mut appended_msg = Vec::new();
        for value in reconstructed_values.iter(){
            appended_msg.extend(value.to_bytes_be());
        }
        let hash = do_hash(&appended_msg);
        log::info!("Reconstructed {} squared random sharings, broadcasting hash {:?}", reconstructed_values.len(), hash);
        self.broadcast(ProtMsg::RandBitReconHash(hash)).await;
        self.verify_rand_bit_recon_termination().await;
    }

    /// Hash agreement over the reconstructed values.
    pub async fn handle_rand_bit_recon_hash(&mut self, hash: Hash, sender: Replica){
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated{
            return;
        }
        recon_state.recv_hash_set.insert(hash);
        recon_state.recv_hash_msgs.push(sender);
        self.verify_rand_bit_recon_termination().await;
    }

    /// Once n-t parties agree on the reconstructed values, turn them into the
    /// inverse square roots the random bits are built from.
    pub async fn verify_rand_bit_recon_termination(&mut self){
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || recon_state.l2_reconstructed.is_empty(){
            return;
        }
        if recon_state.recv_hash_msgs.len() < self.num_nodes - self.num_faults{
            return;
        }
        if recon_state.recv_hash_set.len() != 1{
            log::error!("Parties disagree on the publicly reconstructed squared sharings ({} distinct hashes), abandoning the protocol",
                recon_state.recv_hash_set.len());
            return;
        }
        recon_state.terminated = true;
        let reconstructed_values = recon_state.l2_reconstructed.clone();

        // Take each value's square root in Fp4_61 via the local `Sqrt` trait
        // (Scott's complex method recursing Fp4 → Fp2 → Fp), and invert. Matches
        // async_mpc's pub_rec.rs:78 pattern — discard the sign-choice branch
        // (`let (sqrt, _) = ...`); the randomness comes from upstream `r`, not
        // from which root is picked.
        let reconstructed_square_inverses: Vec<LargeField> = reconstructed_values.into_par_iter()
            .map(|secret| {
                let (sqrt_root, _) = Sqrt::sqrt(&secret).expect("Square root does not exist");
                sqrt_root.inv()
            })
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
            .collect();

        self.mix_circuit_state.rand_bit_inverse_recon_values.extend(reconstructed_square_inverses);
        log::info!("Reconstructed random bit shares of size: {}", self.mix_circuit_state.rand_bit_inverse_recon_values.len());

        self.verify_rand_bit_reconstruction().await;
    }

    pub async fn handle_reconstruct_rand_bits_verify(&mut self, shares: Vec<LargeFieldSer>, share_sender: Replica){
        log::info!("Handling reconstruction of random bit verify shares from sender {}", share_sender);
        let shares: Vec<LargeField> = shares.into_iter()
            .map(|x| LargeField::from_bytes_be(&x).unwrap())
            .collect();

        let shares_len = shares.len();
        self.mix_circuit_state.rand_bit_reconstruction.insert(share_sender, shares);

        if self.mix_circuit_state.rand_bit_reconstruction.len() == self.num_faults+1{
            log::info!("Received threshold number of shares for random bit reconstruction, proceeding to reconstruct.");
            let mut indices = Vec::new();
            let mut shares_index_wise = vec![vec![];shares_len];
            
            for rep in 0..self.num_nodes{
                if self.mix_circuit_state.rand_bit_reconstruction.contains_key(&rep){
                    indices.push(Self::get_share_evaluation_point(rep, self.use_fft, self.roots_of_unity.clone()));
                    let rep_shares = self.mix_circuit_state.rand_bit_reconstruction.get(&rep).unwrap();
                    for (index, share) in rep_shares.iter().enumerate(){
                        shares_index_wise[index].push(share.clone());
                    }
                }
            }

            // generate inverse vandermonde matrix
            let vdm_matrix = vandermonde_matrix(indices);
            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);
            
            let one = LargeField::one();
            let mut reconstructed_square_inverses: Vec<LargeField> = shares_index_wise.into_par_iter()
                .map(|x| {
                    let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &x);
                    let secret = Polynomial::new(&coefficients).evaluate(&LargeField::from(0 as u64));
                    secret
                }).collect();
            reconstructed_square_inverses.truncate(100);
            for secret in reconstructed_square_inverses{
                log::info!("Reconstructed random bit: {:?}", secret);
                log::info!("One: {:?}", one);
                log::info!("Minus one: {:?}", one.inv().unwrap());
            }
        }
    }

    pub async fn verify_rand_bit_reconstruction(&mut self){
        if self.mix_circuit_state.rand_bit_inverse_recon_values.is_empty(){
            return;
        }
        if self.mix_circuit_state.rand_bit_inp_shares.is_empty(){
            return;
        }
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }
        
        let reconstructed_shares = self.mix_circuit_state.rand_bit_inverse_recon_values.clone();
        let rand_bit_input_shares = self.mix_circuit_state.rand_bit_inp_shares.clone();

    
        let final_rand_bit_sharings: Vec<LargeField> = rand_bit_input_shares.into_par_iter().zip(reconstructed_shares.into_par_iter()).map(|(r,re)|{
            let mult_share = r.mul(re);
            return mult_share
        }).collect();

        self.mix_circuit_state.rand_bit_sharings.extend(final_rand_bit_sharings.clone());

        self.terminate("Preprocessing".to_string(), vec![]).await;
        // Preprocessing is complete: hand the application its share of it and
        // start the circuit.
        self.hand_preprocessing_to_application().await;
    }

    /// Give the application the preprocessed sharings its circuit consumes, and
    /// run whatever it schedules in response.
    ///
    /// The engine keeps the rest of the pool for itself: verification multiplies
    /// its own compressed tuples, and the common coin draws from it too.
    pub async fn hand_preprocessing_to_application(&mut self){
        let counts = self.app.preprocessing_count();

        // One mask per multiplication gate, plus the padding each depth's batch
        // needs when the linear protocol rounds it up to a multiple of 2t+1.
        let mult_padding = (counts.depth + 1) * (2*self.num_faults + 1);
        let rand_sharings_needed = counts.simd_mult + mult_padding;
        // Half a zero sharing per gate.
        let zero_sharings_needed = rand_sharings_needed.div_ceil(2);

        let available_rand = self.rand_sharings_state.rand_sharings_mult.len();
        let available_zero = self.rand_sharings_state.rand_2t_sharings_mult.len();
        if available_rand < rand_sharings_needed || available_zero < zero_sharings_needed{
            log::error!("Preprocessing fell short of the application's needs: {} random and {} zero sharings required, {} and {} generated",
                rand_sharings_needed, zero_sharings_needed, available_rand, available_zero);
        }

        let rand_sharings: Vec<LargeField> = self.rand_sharings_state.rand_sharings_mult
            .drain(0..rand_sharings_needed.min(available_rand))
            .collect();
        let zero_sharings: Vec<LargeField> = self.rand_sharings_state.rand_2t_sharings_mult
            .drain(0..zero_sharings_needed.min(available_zero))
            .collect();
        let rand_bits: Vec<LargeField> = self.mix_circuit_state.rand_bit_sharings.drain(..).collect();

        log::info!("Handing {} random sharings, {} zero sharings and {} random bits to the application; {} random and {} zero sharings held back for verification",
            rand_sharings.len(), zero_sharings.len(), rand_bits.len(),
            self.rand_sharings_state.rand_sharings_mult.len(),
            self.rand_sharings_state.rand_2t_sharings_mult.len());

        // Velox has no network routing, so the application never gets routing
        // preprocessing.
        let depth_input = self.app.on_preprocessing_complete(
            rand_sharings,
            zero_sharings,
            rand_bits,
            None,
        ).await;
        self.handle_application_depth_input(depth_input).await;

        // The input sharings may have terminated while preprocessing was still
        // running; hand them over now if so.
        self.forward_input_sharings().await;
    }
}