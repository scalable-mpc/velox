use application::{Application, RandomWireShares};
use std::ops::Mul;

use crypto::hash::{do_hash, Hash};
use fields::{LargeFieldSer, rayon_async, interpolate_at_zero, inverse_vandermonde_from_points, lagrange_coefficients_at_zero, matrix_vector_multiply, matrix_matrix_multiply, powers_matrix, ProtocolField, FieldSer};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use types::{Replica, WrapperMsg};

use crate::{Context, msg::ProtMsg};
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
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
        // Taken, not cloned: this is the only reader of our own squaring output,
        // and removing it makes the `contains_key` guard above short-circuit any
        // re-entry - which is what the guard was there for anyway.
        let mut my_shares = self.mix_circuit_state.rand_bit_recon_shares.remove(&self.myid).unwrap();

        // Pad with shares of zero so the batch splits into whole chunks. A
        // degree-t sharing of zero is the all-zero share vector, so every party
        // pads identically.
        let padding = (chunk_size - (my_shares.len() % chunk_size)) % chunk_size;
        my_shares.extend(vec![FieldElement::<F>::zero(); padding]);
        let num_chunks = my_shares.len()/chunk_size;

        log::info!("Initializing random bit reconstruction of {} sharings over {} chunks of {} ({} padded)",
            my_shares.len() - padding, num_chunks, chunk_size, padding);
        self.mix_circuit_state.rand_bit_recon_state.init_chunks(num_chunks);
        // Set independently of the chunk count: a peer's message may have sized
        // the chunks already, but only this party's own batch knows the padding.
        self.mix_circuit_state.rand_bit_recon_state.padding = Some(padding);

        // Evaluate every chunk polynomial at all n party points in one GEMM:
        // evals[p][chunk] is this party's share of Z_chunk(α_p).
        let chunks: Vec<Vec<FieldElement<F>>> = my_shares.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect();
        let party_powers = powers_matrix(&self.roots_of_unity, chunk_size);
        let evals = matrix_matrix_multiply(&party_powers, &chunks, true);

        for party in 0..self.num_nodes{
            let shares_ser: Vec<LargeFieldSer> = evals[party].iter().map(|share| share.ser_be()).collect();
            let ser_shares_bytes = bincode::serialize(&shares_ser).unwrap();
            let sec_key = self.sec_key_map.get(&party).clone().unwrap();

            let prot_msg = ProtMsg::RandBitReconL1(ser_shares_bytes);
            let wrapper_msg = WrapperMsg::new(prot_msg, self.myid, &sec_key);
            let cancel_handler = self.net_send.send(party, wrapper_msg).await;
            self.add_cancel_handler(cancel_handler);
        }
        self.verify_rand_bit_recon_l1().await;
        self.verify_rand_bit_recon_l2().await;
        self.verify_rand_bit_recon_termination().await;
    }

    /// L1: record a sender's shares of this party's point on each chunk
    /// polynomial. Only bookkeeping — the interpolation happens in
    /// `verify_rand_bit_recon_l1` once the threshold is in.
    pub async fn handle_rand_bit_recon_l1(&mut self, ser_shares: Vec<u8>, sender: Replica){
        log::debug!("Received L1 random bit reconstruction shares from party {}", sender);
        let shares_ser: Vec<LargeFieldSer> = match bincode::deserialize(&ser_shares){
            Ok(shares) => shares,
            Err(e) => {
                log::error!("Error deserializing L1 random bit reconstruction shares: {:?}", e);
                return;
            }
        };
        let shares: Vec<FieldElement<F>> = shares_ser.into_iter()
            .map(|share| F::from_bytes_be(&share).unwrap())
            .collect();

        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());

        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || recon_state.l1_reconstruction_started{
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

        self.verify_rand_bit_recon_l1().await;
    }

    /// Interpolate this party's point on every chunk polynomial from the L1
    /// shares, and broadcast the result.
    ///
    /// The claim flag is set before any of the work, so whichever message
    /// crosses the threshold owns the reconstruction and the ones behind it
    /// return immediately.
    pub async fn verify_rand_bit_recon_l1(&mut self){
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || recon_state.l1_reconstruction_started{
            return;
        }
        if recon_state.recv_share_count_l1 < self.num_nodes - self.num_faults{
            return;
        }
        recon_state.l1_reconstruction_started = true;

        log::info!("Attempting L1 reconstruction of random bit sharings");
        // The per-party L1 shares are dead: this interpolation is their only
        // reader, the claim flag above makes it run once, and every later L1
        // message is dropped by the same flag in `handle_rand_bit_recon_l1`.
        // Taking them here also gives the rayon job owned inputs so it can be
        // yielded instead of parking the `Context` task.
        let (indices, l1_shares) = std::mem::take(&mut recon_state.l1_shares);

        // Constant term only - see the note in `quad_mult.rs`.
        let my_points: Vec<FieldElement<F>> = rayon_async(move || {
            let lambdas = lagrange_coefficients_at_zero(&indices);
            l1_shares.par_iter()
                .map(|chunk_shares| interpolate_at_zero(&lambdas, chunk_shares))
                .collect()
        }).await;

        // Re-acquire: `recon_state` borrowed `self` across the await. The claim
        // flag set above means any L1 message handled during the job returned
        // early, so nothing else touched this state.
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        recon_state.l1_reconstructed.extend(my_points.iter().cloned());

        let points_ser: Vec<LargeFieldSer> = my_points.iter().map(|point| point.ser_be()).collect();
        let ser_points = bincode::serialize(&points_ser).unwrap();
        self.broadcast(ProtMsg::RandBitReconL2(ser_points)).await;
        self.verify_rand_bit_recon_l2().await;
    }

    /// L2: record the point a sender reconstructed on each chunk polynomial.
    /// Only bookkeeping — `verify_rand_bit_recon_l2` does the interpolation.
    pub async fn handle_rand_bit_recon_l2(&mut self, ser_points: Vec<u8>, sender: Replica){
        log::debug!("Received L2 random bit reconstruction shares from party {}", sender);
        let points_ser: Vec<LargeFieldSer> = match bincode::deserialize(&ser_points){
            Ok(points) => points,
            Err(e) => {
                log::error!("Error deserializing L2 random bit reconstruction shares: {:?}", e);
                return;
            }
        };
        let points: Vec<FieldElement<F>> = points_ser.into_iter()
            .map(|point| F::from_bytes_be(&point).unwrap())
            .collect();

        // L1 evaluated the chunk polynomials at these same points.
        let evaluation_point = self.roots_of_unity.get(sender).unwrap().clone();

        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || recon_state.l2_reconstruction_started{
            return;
        }
        recon_state.init_chunks(points.len());
        recon_state.l2_shares.0.push(evaluation_point);
        for (index, point) in points.into_iter().enumerate(){
            recon_state.l2_shares.1[index].push(point);
        }
        recon_state.recv_share_count_l2 += 1;

        self.verify_rand_bit_recon_l2().await;
    }

    /// Interpolate the publicly reconstructed values from the L2 points, and
    /// broadcast a hash of them. Claimed by the first message to cross the
    /// threshold, like the L1 step.
    pub async fn verify_rand_bit_recon_l2(&mut self){
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;
        if recon_state.terminated || recon_state.l2_reconstruction_started{
            return;
        }
        if recon_state.recv_share_count_l2 < self.num_nodes - self.num_faults{
            return;
        }
        if recon_state.padding.is_none(){
            return;
        }
        recon_state.l2_reconstruction_started = true;

        log::info!("Attempting L2 reconstruction of random bit sharings");
        // Same argument as L1: the raw L2 points have no reader past this
        // interpolation, and late L2 messages are dropped by the claim flag.
        let (indices, l2_shares) = std::mem::take(&mut recon_state.l2_shares);
        let padding = recon_state.padding.unwrap();

        // Interpolating 2t+1 points of a degree-2t polynomial gives back its
        // coefficients, which are the chunk's values.
        // L2 wants every coefficient (they are the chunk's values), so a full
        // inverse - via the O(n^2) closed form.
        let mut reconstructed_values: Vec<FieldElement<F>> = rayon_async(move || {
            let inv_vdm_matrix = inverse_vandermonde_from_points(&indices);
            l2_shares.par_iter()
                .map(|chunk_points| matrix_vector_multiply(&inv_vdm_matrix, chunk_points))
                .flatten()
                .collect()
        }).await;
        for _ in 0..padding{
            reconstructed_values.pop();
        }
        // Re-acquire after the await; the claim flag above kept this state ours.
        let recon_state = &mut self.mix_circuit_state.rand_bit_recon_state;

        // Pin down what everyone reconstructed before deriving random bits from it.
        let mut appended_msg = Vec::new();
        for value in reconstructed_values.iter(){
            appended_msg.extend(value.ser_be());
        }
        let hash = do_hash(&appended_msg);
        log::info!("Reconstructed {} squared random sharings, broadcasting hash {:?}", reconstructed_values.len(), hash);
        recon_state.l2_reconstructed.extend(reconstructed_values);
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
        // Moved out: the `terminated` flag set above turns every later entry
        // into this function - and into both share handlers - into an early
        // return, so nothing reads `l2_reconstructed` again.
        let reconstructed_values = std::mem::take(&mut recon_state.l2_reconstructed);

        // Take each value's square root in Fp4_61 via the local `Sqrt` trait
        // (Scott's complex method recursing Fp4 → Fp2 → Fp), and invert. Matches
        // async_mpc's pub_rec.rs:78 pattern — discard the sign-choice branch
        // (`let (sqrt, _) = ...`); the randomness comes from upstream `r`, not
        // from which root is picked.
        // Yielded: `F::sqrt` is an exponentiation over the extension field and
        // `.inv()` is a second one, so this is the densest per-element job in the
        // engine - order 10^3 field multiplications each, against ~1 for the
        // interpolation sites. Running it inline parked the `Context` task for
        // the whole batch.
        let reconstructed_square_inverses: Vec<FieldElement<F>> = rayon_async(move || {
            reconstructed_values.into_par_iter()
                .map(|secret| {
                    let (sqrt_root, _) = F::sqrt(&secret).expect("Square root does not exist");
                    sqrt_root.inv()
                })
                .filter(|x| x.is_ok())
                .map(|x| x.unwrap())
                .collect()
        }).await;

        self.mix_circuit_state.rand_bit_inverse_recon_values.extend(reconstructed_square_inverses);
        log::info!("Reconstructed random bit shares of size: {}", self.mix_circuit_state.rand_bit_inverse_recon_values.len());

        self.verify_rand_bit_reconstruction().await;
    }

    pub async fn handle_reconstruct_rand_bits_verify(&mut self, shares: Vec<LargeFieldSer>, share_sender: Replica){
        log::info!("Handling reconstruction of random bit verify shares from sender {}", share_sender);
        let shares: Vec<FieldElement<F>> = shares.into_iter()
            .map(|x| F::from_bytes_be(&x).unwrap())
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

            // Only the first 100 are logged, so only the first 100 are
            // interpolated. This used to reconstruct the entire batch and then
            // `truncate(100)`, doing `shares_len/100` times the necessary work on
            // the `Context` task for a debug log.
            let one = FieldElement::<F>::one();
            shares_index_wise.truncate(100);
            let lambdas = lagrange_coefficients_at_zero(&indices);
            let reconstructed_square_inverses: Vec<FieldElement<F>> = shares_index_wise.into_par_iter()
                .map(|x| interpolate_at_zero(&lambdas, &x))
                .collect();
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
        
        // Both inputs are consumed here and have no other reader; taking them
        // also makes the two guards above short-circuit any re-entry, which is
        // exactly what the third guard (`rand_bit_sharings` non-empty) did.
        let reconstructed_shares = std::mem::take(&mut self.mix_circuit_state.rand_bit_inverse_recon_values);
        let rand_bit_input_shares = std::mem::take(&mut self.mix_circuit_state.rand_bit_inp_shares);

        let final_rand_bit_sharings: Vec<FieldElement<F>> = rayon_async(move || {
            rand_bit_input_shares.into_par_iter()
                .zip(reconstructed_shares.into_par_iter())
                .map(|(r,re)| r.mul(re))
                .collect()
        }).await;

        self.mix_circuit_state.rand_bit_sharings.extend(final_rand_bit_sharings);

        self.terminate("Preprocessing".to_string(), vec![]).await;
        // Preprocessing is complete: hand the application its share of it and
        // start the circuit.
        self.hand_preprocessing_to_application().await;
    }

    /// Hand the application the random wires its circuit consumes — the bits
    /// squared above and the sharings carved off in `verify_termination` — and
    /// run whatever it schedules in response.
    ///
    /// The multiplication masks stay in the engine's pool: it draws them per
    /// batch in `choose_multiplication_protocol`, for the application's depths
    /// exactly as for its own. Applications used to be handed a slice of the
    /// pool up front, sized here and re-portioned there, which put the
    /// protocol's `2t+1` batching rule in application code.
    pub async fn hand_preprocessing_to_application(&mut self){
        let rand_bits: Vec<FieldElement<F>> = self.mix_circuit_state.rand_bit_sharings.drain(..).collect();
        let wire_sharings = std::mem::take(&mut self.rand_sharings_state.app_wire_sharings);

        log::info!("Handing {} random bits and {} random sharings to the application; {} random and {} zero sharings held in the engine's pool for its depths, verification and coins",
            rand_bits.len(),
            wire_sharings.len(),
            self.rand_sharings_state.rand_sharings_mult.len(),
            self.rand_sharings_state.rand_2t_sharings_mult.len());

        let depth_input = self.app.on_preprocessing_complete(RandomWireShares::new(rand_bits, wire_sharings)).await;
        self.handle_application_depth_input(depth_input).await;
    }
}
