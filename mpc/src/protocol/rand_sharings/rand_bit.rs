use application::{Application, RandomWireShares};
use std::ops::Mul;

use fields::{LargeFieldSer, rayon_async, interpolate_at_zero, lagrange_coefficients_at_zero, ProtocolField};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use types::{Replica};

use crate::{Context, protocol::{online_phase::RAND_BIT_RECON_DEPTH, public_reconstruction::{ReconConfig, ReconKind}}};
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    /// Publicly reconstruct the squared random sharings.
    ///
    /// The squares are outputs of the preprocessing-depth multiplication, so
    /// their sharing polynomials are uniformly random and the reconstruction
    /// runs at degree `t` without a privacy term. It is keyed at its own
    /// depth, `RAND_BIT_RECON_DEPTH`: the multiplication that produced the
    /// squares already owns `preprocessing_mult_depth`, and a depth carries
    /// one reconstruction.
    pub async fn init_rand_bit_reconstruction(&mut self){
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }
        if !self.mix_circuit_state.rand_bit_recon_shares.contains_key(&self.myid){
            return;
        }
        // Taken, not cloned: this is the only reader of our own squaring output,
        // and removing it makes the `contains_key` guard above short-circuit any
        // re-entry - which is what the guard was there for anyway.
        let my_shares = self.mix_circuit_state.rand_bit_recon_shares.remove(&self.myid).unwrap();
        log::info!("Initializing random bit reconstruction of {} sharings", my_shares.len());
        self.init_public_reconstruction(RAND_BIT_RECON_DEPTH, ReconKind::RandBit, my_shares, ReconConfig::RAND_BIT, None).await;
    }

    /// The squares are public and agreed on: turn them into the inverse
    /// square roots the random bits are built from.
    pub async fn complete_rand_bit_reconstruction(&mut self, reconstructed_values: Vec<FieldElement<F>>){
        // Take each value's square root via `ProtocolField::sqrt` and invert.
        // Discard the sign-choice branch (`let (sqrt, _) = ...`); the
        // randomness comes from upstream `r`, not from which root is picked.
        // Yielded: `F::sqrt` is an exponentiation and `.inv()` a second one, so
        // this is the densest per-element job in the engine - order 10^3 field
        // multiplications each. Running it inline parked the `Context` task for
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
