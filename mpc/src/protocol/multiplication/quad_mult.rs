use application::Application;
use crypto::hash::do_hash;
use fields::{LargeFieldSer, rayon_async, interpolate_at_zero, lagrange_coefficients_at_zero, ProtocolField, FieldSer};
use rayon::prelude::{IntoParallelIterator, IndexedParallelIterator, ParallelIterator, IntoParallelRefIterator};
use types::Replica;

use crate::{Context, msg::ProtMsg};
use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    pub async fn init_quadratic_multiplication_prot(&mut self,
        a_shares: Vec<Vec<FieldElement<F>>>,
        b_shares: Vec<Vec<FieldElement<F>>>,
        depth: usize,
        mut rand_sharings: Vec<FieldElement<F>>,
        mut zero_sharings: Vec<FieldElement<F>>
    ){
        log::info!("Starting quadratic multiplication protocol");
        if a_shares.len() != b_shares.len() {
            log::error!("Quadratic multiplication protocol failed: a and b shares length mismatch");
            return;
        }
        let n = a_shares.len();
        // One mask and one zero sharing per gate.
        if rand_sharings.len() < n || zero_sharings.len() < n{
            log::error!("Not enough random shares for multiplication protocol at depth {}: {} gates need {} random and {} zero sharings, got {} and {}",
                depth, n, n, n, rand_sharings.len(), zero_sharings.len());
            return;
        }
        rand_sharings.truncate(n);
        zero_sharings.truncate(n);

        let verified_depth = self.is_verified_depth(depth);
        let depth_state = self.mult_state.get_single_depth_state(depth, false, n);

        // Log these entries in the verification state for later verification.
        // Read by reference: only the first sharing of each gate is verified, so
        // cloning the whole batch duplicated every inner-product operand too.
        if verified_depth {
            let first_a_shares = a_shares.iter().map(|x| x[0].clone()).collect();
            let first_b_shares = b_shares.iter().map(|x| x[0].clone()).collect();
            self.verf_state.add_mult_inputs(depth, first_a_shares, first_b_shares);
        }

        // Share rand_utils. Cloned element-wise rather than as a whole `Vec`, so
        // the batch's masks are not materialised a third time as a temporary.
        depth_state.util_rand_sharings.extend(rand_sharings.iter().cloned());
        
        // Perform multiplication. This is the matmul core - one inner product per
        // gate - and the largest per-depth job, so it is yielded to rayon rather
        // than run inline: the `select!` loop keeps draining while it runs.
        let mult_shares = rayon_async(move || {
            (a_shares.into_par_iter()
                .zip(b_shares.into_par_iter()))
            .zip(rand_sharings.into_par_iter()
                .zip(zero_sharings.into_par_iter()))
            .map(|((a,b),(r,o))| (Self::dot_product(&a,&b)+r+o).ser_be())
            .collect::<Vec<LargeFieldSer>>() // Perform dot product and add random shares
        }).await;

        let ser_shares = bincode::serialize(&mult_shares).unwrap();
        self.broadcast(ProtMsg::QuadShares(ser_shares, depth)).await;
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_quadratic_mult_shares(&mut self, depth: usize, shares: Vec<u8>, sender: Replica){
        log::info!("Handling quadratic multiplication shares for depth {} from sender {}", depth, sender);
        // Deserialize shares
        let shares_deser = bincode::deserialize::<Vec<LargeFieldSer>>(&shares).unwrap();
        let shares_lf: Vec<FieldElement<F>> = shares_deser.into_iter().map(|x| F::from_bytes_be(&x).unwrap()).collect();

        let evaluation_point = Self::get_share_evaluation_point(sender,self.use_fft, self.roots_of_unity.clone());

        // Add shares to the depth state
        let depth_state = self.mult_state.get_single_depth_state(depth, false, shares_lf.len());
        // If this depth already terminated, `clear_shares()` emptied l1_shares,
        // and once this depth's interpolation has run `clear_l1_shares()` did
        // the same, so its per-group inner vectors are gone. A late quadratic
        // share is dead; drop it to avoid corrupting l1_shares.0 /
        // recv_share_count_l1 against an empty l1_shares.1.
        if depth_state.depth_terminated || depth_state.l1_reconstruction_done {
            return;
        }
        depth_state.l1_shares.0.push(evaluation_point.clone()); // Add the evaluation point to the indices
        for (share,shares) 
                in shares_lf.into_iter().zip(depth_state.l1_shares.1.iter_mut()){
            shares.push(share);
        }

        depth_state.recv_share_count_l1 = depth_state.recv_share_count_l1 + 1; // Increment the count of received shares
        
        if depth_state.recv_share_count_l1 == self.num_nodes - self.num_faults{
            // Reconstruct secrets
            log::info!("Received n-t shares for quadratic protocol reconstruction at depth {}, reconstructing secrets", depth);
            // Claimed before the work, so a later quadratic share is dropped by
            // the guard above instead of piling onto buffers nobody reads again.
            depth_state.l1_reconstruction_done = true;

            let indices = depth_state.l1_shares.0.clone();

            // The raw per-party shares are dead once interpolated: this is their
            // only reader and `l1_reconstruction_done` above makes it run once.
            // Taking them here (rather than clearing after) also gives the rayon
            // job an owned input, which is what lets it be yielded.
            let l1_shares = std::mem::take(&mut depth_state.l1_shares.1);
            depth_state.clear_l1_shares();

            // Only the secret is wanted, so only the secret is computed: the
            // Lagrange weights at zero are one O(n^2) vector, after which each
            // group is a length-n dot product. Building the full n x n inverse
            // and reading one entry out of the result cost O(n^3) plus an n-fold
            // matvec per group.
            let reconstructed_secrets: Vec<FieldElement<F>> = rayon_async(move || {
                let lambdas = lagrange_coefficients_at_zero(&indices);
                l1_shares.par_iter()
                    .map(|evaluations| interpolate_at_zero(&lambdas, evaluations))
                    .collect()
            }).await;

            // `depth_state` was borrowed across the await above; re-acquire it.
            // The entry itself cannot disappear (depths are reclaimed in place by
            // `clear_shares()`, never removed from `depth_share_map`), and the
            // depth cannot terminate during the await either: termination needs
            // `reconstructed_len > 0`, which is exactly what this job is about to
            // produce, so `verify_depth_mult_termination` returns early and
            // leaves the depth retryable until we write the result back.
            let depth_state = self.mult_state.get_single_depth_state(depth, false, 0);
            // Broadcast hash of this reconstructed value.
            let mut appended_msg = Vec::new();
            for secret in reconstructed_secrets.iter(){
                appended_msg.extend(secret.ser_be());
            }
            depth_state.l1_shares_reconstructed.extend(reconstructed_secrets);
            let hash = do_hash(&appended_msg);
            log::info!("Completed processing triples at depth {} with quadratic sharings, broadcasting hash {:?}", depth, hash);
            self.init_hash_broadcast(hash, depth).await;
            self.verify_depth_mult_termination(depth).await;
        }
    }
}