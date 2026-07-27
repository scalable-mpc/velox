use application::{Application, Multiplication};
use std::ops::Mul;

use crypto::{hash::{Hash}};
use lambdaworks_math::{polynomial::Polynomial};
use fields::{LargeField};
use types::{Replica};

use crate::{Context, msg::ProtMsg, protocol::online_phase::APPLICATION_DEPTH_OFFSET};

use super::mult_state::SingleDepthState;

impl<A: Application> Context<A>{
    /// Multiplication driven by the engine itself — random bit squaring and
    /// tuple verification — which draws its preprocessing from the engine's pool.
    pub async fn choose_multiplication_protocol(&mut self,a_shares: Vec<Vec<LargeField>>, b_shares: Vec<Vec<LargeField>>, depth: usize){
        let (rand_needed, zero_needed) = self.multiplication_preprocessing_requirement(a_shares.len());
        let Some((rand_sharings, zero_sharings)) = self.take_multiplication_preprocessing(rand_needed, zero_needed) else {
            log::error!("Not enough preprocessed sharings for the multiplication at depth {}: need {} random and {} zero sharings, {} and {} left",
                depth, rand_needed, zero_needed,
                self.rand_sharings_state.rand_sharings_mult.len(),
                self.rand_sharings_state.rand_2t_sharings_mult.len());
            return;
        };
        self.multiply_with_preprocessing(a_shares, b_shares, depth, rand_sharings, zero_sharings).await;
    }

    /// Multiplication scheduled by the application. The batch carries the
    /// preprocessing it consumes — the engine's pool is reserved for random bit
    /// generation and verification — and its depth is shifted into the
    /// application range before it reaches the multiplication protocol.
    pub async fn multiply_application_batch(&mut self, mult: Multiplication<LargeField>){
        let input = &mult.input;
        if !input.x.1.is_empty() || !input.y.1.is_empty() || !input.x_ip.1.is_empty() || !input.y_ip.1.is_empty(){
            log::warn!("Application supplied second-half sharings at depth {}, but Velox sharings are unpacked; ignoring them", input.depth);
        }

        // Element-wise gates multiply one sharing by one sharing; inner-product
        // gates multiply two vectors of sharings into a single output.
        let mut a_shares: Vec<Vec<LargeField>> = input.x.0.iter().map(|x| vec![x.clone()]).collect();
        let mut b_shares: Vec<Vec<LargeField>> = input.y.0.iter().map(|y| vec![y.clone()]).collect();
        a_shares.extend(input.x_ip.0.iter().cloned());
        b_shares.extend(input.y_ip.0.iter().cloned());

        if a_shares.is_empty(){
            log::warn!("Application scheduled an empty multiplication batch at depth {}; nothing to do", input.depth);
            return;
        }

        let depth = input.depth + APPLICATION_DEPTH_OFFSET;
        let (rand_needed, zero_needed) = self.multiplication_preprocessing_requirement(a_shares.len());
        if mult.random_sharings.len() < rand_needed || mult.random_zero_sharings.len() < zero_needed{
            log::error!("Application supplied too little preprocessing for its {} gates at depth {}: need {} random and {} zero sharings, got {} and {}",
                a_shares.len(), input.depth, rand_needed, zero_needed,
                mult.random_sharings.len(), mult.random_zero_sharings.len());
            return;
        }

        self.multiply_with_preprocessing(
            a_shares,
            b_shares,
            depth,
            mult.random_sharings.clone(),
            mult.random_zero_sharings.clone()
        ).await;
    }

    /// Run a multiplication batch against preprocessing the caller supplies.
    pub async fn multiply_with_preprocessing(&mut self,
        a_shares: Vec<Vec<LargeField>>,
        b_shares: Vec<Vec<LargeField>>,
        depth: usize,
        rand_sharings: Vec<LargeField>,
        zero_sharings: Vec<LargeField>
    ){
        // Padding necessary to make sure each group has the same number of elements
        let num_multiplications = a_shares.len();
        if num_multiplications > self.multiplication_switch_threshold{
            Box::pin(self.init_linear_multiplication_prot(a_shares, b_shares, depth, rand_sharings, zero_sharings)).await;
        }
        else{
            // Use quadratic multiplication protocol here
            Box::pin(self.init_quadratic_multiplication_prot(a_shares, b_shares, depth, rand_sharings, zero_sharings)).await;
        }
    }

    /// Preprocessing a batch of `num_gates` multiplications consumes, as
    /// `(random sharings, 2t-sharings of zero)`.
    pub fn multiplication_preprocessing_requirement(&self, num_gates: usize) -> (usize, usize){
        if num_gates > self.multiplication_switch_threshold{
            // The linear protocol pads the batch up to a multiple of 2t+1, draws
            // one mask per padded gate, and t+1 zero sharings per group — half a
            // zero sharing per gate.
            let group = 2*self.num_faults + 1;
            let padded_gates = num_gates.div_ceil(group)*group;
            (padded_gates, (padded_gates/group)*(self.num_faults + 1))
        }
        else{
            // The quadratic protocol draws one of each per gate.
            (num_gates, num_gates)
        }
    }

    /// Depths whose multiplication tuples the verification phase checks: the
    /// random bit squaring and every application circuit depth. Verification's
    /// own multiplications are excluded — they are what does the checking.
    pub fn is_verified_depth(&self, depth: usize) -> bool{
        depth == self.preprocessing_mult_depth
            || (depth >= APPLICATION_DEPTH_OFFSET && depth < self.delinearization_depth)
    }

    /// Draw multiplication preprocessing from the engine's own pool.
    fn take_multiplication_preprocessing(&mut self, num_rand: usize, num_zero: usize) -> Option<(Vec<LargeField>, Vec<LargeField>)>{
        let pool = &mut self.rand_sharings_state;
        if pool.rand_sharings_mult.len() < num_rand || pool.rand_2t_sharings_mult.len() < num_zero{
            return None;
        }
        let rand_sharings = pool.rand_sharings_mult.drain(0..num_rand).collect();
        let zero_sharings = pool.rand_2t_sharings_mult.drain(0..num_zero).collect();
        Some((rand_sharings, zero_sharings))
    }

    pub async fn init_hash_broadcast(&mut self, hash: Hash, depth: usize){
        self.broadcast(ProtMsg::HashZMsg(hash,depth,false)).await;
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_hash_broadcast(&mut self, hash: Hash, depth: usize, lin_or_quad: bool, sender: Replica){
        if !self.mult_state.depth_share_map.contains_key(&depth){
            let single_depth_state = SingleDepthState::new(lin_or_quad);
            self.mult_state.depth_share_map.insert(depth, single_depth_state);
        }
        
        let ex_mult_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
        ex_mult_state.recv_hash_set.insert(hash.clone());
        ex_mult_state.recv_hash_msgs.push(sender);
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn verify_depth_mult_termination(&mut self, depth: usize){
        // Now, subtract random sharings from the reconstructed secrets
        if !self.mult_state.depth_share_map.contains_key(&depth){
            return;
        }
        // Computed before `mult_state` borrows `self.mult_state`.
        let verified_depth = self.is_verified_depth(depth);
        let mult_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
        if mult_state.depth_terminated{
            return;
        }
        if mult_state.recv_hash_msgs.len() >= self.num_nodes-self.num_faults && mult_state.recv_hash_set.len() == 1{
            log::info!("Received 2t+1 Hashes for multiplication at depth {} with Hash {:?}, computing sharings of output gate",depth, mult_state.recv_hash_set);            
        }
        else{
            return;
        }
        // Which reconstruction this depth terminates on depends on the protocol
        // it ran. Length-check before taking ownership: this function is re-run
        // on every hash message, and the reconstruction may legitimately still
        // be empty (hash agreement can be reached before this party's own L1/L2
        // interpolation), in which case the depth has to stay retryable.
        let reconstructed_len = if mult_state.two_levels {
            mult_state.l2_shares_reconstructed.len()
        } else {
            // Quadratic multiplication layer
            mult_state.l1_shares_reconstructed.len()
        };

        // Get the random sharings
        // Subtract random sharings
        log::info!("Subtracting random sharings with length {} from reconstructed secrets {} at depth {}",mult_state.util_rand_sharings.len(), reconstructed_len, depth);

        // Weird bugs occurring in this phase.
        if mult_state.util_rand_sharings.len() == reconstructed_len && reconstructed_len > 0{
            log::info!("Moving on to depth {}", depth + 1);
            // Both operands are consumed here and nothing else reads them again
            // (`clear_shares()` below would free them regardless), so move them
            // out instead of cloning two full per-depth vectors.
            let reconstructed_blinded_secrets = if mult_state.two_levels {
                std::mem::take(&mut mult_state.l2_shares_reconstructed)
            } else {
                std::mem::take(&mut mult_state.l1_shares_reconstructed)
            };
            // Par iter from rayon not needed here because we are not doing heavy computation
            let mut shares_next_depth: Vec<LargeField>
                    = std::mem::take(&mut mult_state.util_rand_sharings).into_iter()
                        .zip(reconstructed_blinded_secrets.into_iter())
                            .map(|(sharing, recon_secret)|recon_secret-sharing)
                                .collect();

            // Trim the last k shares for padding
            for _i in 0..mult_state.padding_shares{
                shares_next_depth.pop();
            }
            log::info!("Shares for next depth: {}", shares_next_depth.len());
            // Only verified depths' tuples are ever read back (delinearization
            // filters on `is_verified_depth`), so recording the outputs of
            // verification's own compression multiplications just retained a
            // vector per compression level for the rest of the run.
            if verified_depth{
                self.verf_state.add_mult_output_shares(depth, shares_next_depth.clone()); // Store the shares for the next depth
            }
            // self.choose_multiplication_protocol(a_shares, b_shares, depth)
            // How to handle next depth wires?
            mult_state.depth_terminated = true;
            // The per-party L1/L2 share buffers for this depth are now dead: the
            // reconstructed secrets have been moved into `shares_next_depth`
            // above and handed to the next depth / verification. Free them. The
            // termination bookkeeping is retained so late shares stay deduped.
            // NOTE: this only frees party-sent shares; the multiplication tuples
            // in `verf_state` are kept, as verification consumes them across all
            // depths.
            // Most of this is already gone by now - the raw shares were freed at
            // their interpolation and the two vectors above were moved out - so
            // this is the backstop for the paths that did not reach those points
            // (e.g. hash agreement on a depth whose L1 shares never all arrived).
            mult_state.clear_shares();
            if depth == self.preprocessing_mult_depth{
                // Random bit sharings, add them to mix_circuit state
                log::info!("Multiplication complete for rand_bit preparation with shares_len: {:?}", shares_next_depth.len());
                self.mix_circuit_state.rand_bit_recon_shares.insert(self.myid, shares_next_depth);
                self.init_rand_bit_reconstruction().await;
            }
            else if depth >= self.delinearization_depth{
                // Verification multiplies its own compressed tuples.
                self.verify_ex_mult_termination_verification(depth, shares_next_depth).await;
            }
            else{
                // An application circuit depth: hand the results back and let the
                // application decide what runs next.
                self.verify_application_depth_termination(depth, shares_next_depth).await;
            }
        }
        else{
            log::error!("Secrets less than number of random sharings used, this should not happen. Abandoning the protocol at depth {}",depth);
            return;
        }
    }

    pub(crate) fn dot_product(
        a: &Vec<LargeField>,
        b: &Vec<LargeField>,
    ) -> LargeField {
        // Assert that the vectors have the same length
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
        // Compute the dot product
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.clone().mul(y.clone()))
            .sum()
    }

    #[allow(dead_code)] // Preserved as fallback API; the hot caller in lin_mult.rs
    // now uses the batched GEMM path via `matrix_matrix_multiply(&party_powers, …)`.
    pub(crate) fn evaluate_polynomial_from_coefficients_at_position(
        coefficients: Vec<LargeField>,
        evaluation_point: LargeField,
    ) -> LargeField {
        Polynomial::new(&coefficients).evaluate(&evaluation_point)
    }

    pub fn get_share_evaluation_point(party: usize, use_fft:bool, roots_of_unity: Vec<LargeField>)-> LargeField{
        if use_fft{
            roots_of_unity.get(party).clone().unwrap().clone()
        }
        else{
            LargeField::from((party+1) as u64)
        }
    }
}