//! The public reconstruction as the engine runs it: one batch per depth, the
//! two message levels, hash agreement, and the hand-off to whoever started it.
//!
//! The state lives on the depth's `SingleDepthState`. A peer that is ahead of
//! us may create it with its first message; our own `init` fills in what only
//! we know — the kind, the config, the padding — and every step that needs
//! those waits for it. Waiting is safe: every honest party starts every
//! depth, and until we have started one our own L1 messages are not out
//! either, so nobody is waiting on us to be faster than that.

use planner::api::engine::Application;
use bincode::Result as BincodeResult;
use crypto::hash::Hash;
use fields::{rayon_async, FieldSer, LargeFieldSer, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::*;
use types::{Replica, WrapperMsg};

use crate::{msg::ProtMsg, Context};

use super::{
    config::{ReconConfig, ReconKind},
    math::{hash_of, l1_interpolate, l1_messages, l2_interpolate, pad_and_chunk, zero_terms},
};

impl<F: ProtocolField, A: Application<F>> Context<F, A> {
    /// The point of every party, in party order — where the chunk polynomials
    /// are evaluated (L1) and read back (L2). The same procedure as the
    /// sharing polynomials' points, so one definition serves both levels.
    fn party_points(&self) -> Vec<FieldElement<F>> {
        (0..self.num_nodes)
            .map(|party| Self::get_share_evaluation_point(party, self.use_fft, self.roots_of_unity.clone()))
            .collect()
    }

    /// Start reconstructing `values` at `depth`. With `config.privacy`, the
    /// caller supplies `t+1` degree-`2t` zero sharings per chunk of `2t+1`
    /// values.
    ///
    /// Refuses a depth that already has a reconstruction of a different kind,
    /// or one this party already started: one batch per depth.
    pub async fn init_public_reconstruction(
        &mut self,
        depth: usize,
        kind: ReconKind,
        values: Vec<FieldElement<F>>,
        config: ReconConfig,
        zero_sharings: Option<Vec<FieldElement<F>>>,
    ) {
        let chunk_size = 2 * self.num_faults + 1;
        let zero_per_chunk = self.num_faults + 1;
        let party_points = self.party_points();

        {
            let recon = self.mult_state.recon(depth);
            if recon.own_init_done() {
                log::error!("A public reconstruction at depth {} was started twice; ignoring the second", depth);
                return;
            }
            if let Some(existing) = recon.kind {
                if existing != kind {
                    log::error!(
                        "Depth {} already carries a {:?} reconstruction, refusing to start a {:?} one: one batch per depth",
                        depth, existing, kind
                    );
                    return;
                }
            }
        }

        let (chunks, padding) = pad_and_chunk(values, chunk_size);
        let num_chunks = chunks.len();
        log::info!(
            "Starting {:?} public reconstruction at depth {}: {} chunks of {} ({} padded), degree {:?}, privacy {}",
            kind, depth, num_chunks, chunk_size, padding, config.degree, config.privacy
        );

        let privacy_input = if config.privacy {
            match zero_sharings {
                Some(zero) => Some(zero),
                None => {
                    log::error!("Public reconstruction at depth {} asked for privacy without zero sharings", depth);
                    return;
                }
            }
        } else {
            None
        };
        let messages: Result<Vec<Vec<FieldElement<F>>>, String> = rayon_async(move || {
            let mut messages = l1_messages(&chunks, &party_points);
            if let Some(zero) = privacy_input {
                let terms = zero_terms(&zero, num_chunks, zero_per_chunk, &party_points)?;
                messages
                    .par_iter_mut()
                    .zip(terms.par_iter())
                    .for_each(|(party_msgs, party_terms)| {
                        for (m, z) in party_msgs.iter_mut().zip(party_terms.iter()) {
                            *m = &*m + z;
                        }
                    });
            }
            Ok(messages)
        })
        .await;
        let messages = match messages {
            Ok(messages) => messages,
            Err(err) => {
                log::error!("Public reconstruction at depth {}: {}", depth, err);
                return;
            }
        };

        // Re-acquire after the await, then record what only the owner knows.
        // The owner is the authority on the chunk count: a peer's message may
        // have sized the buffers already, and if it sized them differently
        // that peer is not running the same batch.
        {
            let recon = self.mult_state.recon(depth);
            if recon.num_chunks != 0 && recon.num_chunks != num_chunks {
                log::error!(
                    "Public reconstruction at depth {}: peers sent {} chunks but this party's batch has {}; abandoning",
                    depth, recon.num_chunks, num_chunks
                );
                return;
            }
            recon.kind = Some(kind);
            recon.config = Some(config);
            recon.padding = Some(padding);
            recon.init_chunks(num_chunks);
        }

        for (party, shares) in messages.into_iter().enumerate() {
            let ser: Vec<LargeFieldSer> = shares.iter().map(|s| s.ser_be()).collect();
            let bytes = bincode::serialize(&ser).unwrap();
            let sec_key = self.sec_key_map.get(&party).clone().unwrap();
            let wrapper = WrapperMsg::new(ProtMsg::ReconL1(bytes, depth), self.myid, &sec_key);
            let cancel_handler = self.net_send.send(party, wrapper).await;
            self.add_cancel_handler(cancel_handler);
        }

        // Peers' messages may already be in; every step is gated on own init.
        self.verify_recon_l1(depth).await;
        self.verify_recon_l2(depth).await;
        self.verify_recon_termination(depth).await;
    }

    /// L1: a sender's shares of this party's point on each chunk polynomial.
    pub async fn handle_recon_l1(&mut self, bytes: Vec<u8>, depth: usize, sender: Replica) {
        let ser: BincodeResult<Vec<LargeFieldSer>> = bincode::deserialize(&bytes);
        let shares: Vec<FieldElement<F>> = match ser {
            Ok(ser) => ser.into_iter().map(|s| F::from_bytes_be(&s).unwrap()).collect(),
            Err(e) => {
                log::error!("Error deserializing L1 reconstruction shares from {} at depth {}: {:?}", sender, depth, e);
                return;
            }
        };
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());

        let recon = self.mult_state.recon(depth);
        if !recon.accepts_l1() {
            return;
        }
        // The sender chunked the same batch we will, so it sizes the buffers
        // for parties whose own init has not run yet.
        recon.init_chunks(shares.len());
        if shares.len() != recon.num_chunks {
            log::error!(
                "L1 reconstruction shares from {} at depth {} cover {} chunks, expected {}; dropped",
                sender, depth, shares.len(), recon.num_chunks
            );
            return;
        }
        recon.l1_shares.0.push(evaluation_point);
        for (index, share) in shares.into_iter().enumerate() {
            recon.l1_shares.1[index].push(share);
        }
        recon.recv_l1 += 1;

        self.verify_recon_l1(depth).await;
    }

    /// Interpolate this party's point on every chunk polynomial once `n − t`
    /// L1 shares are in, and broadcast it.
    async fn verify_recon_l1(&mut self, depth: usize) {
        let threshold = self.num_nodes - self.num_faults;
        let num_faults = self.num_faults;
        let recon = self.mult_state.recon(depth);
        if !recon.accepts_l1() || !recon.own_init_done() || recon.recv_l1 < threshold {
            return;
        }
        let degree = recon.config.unwrap().degree;
        // Claimed before the job: `take_l1` sets `l1_started`.
        let (indices, shares) = recon.take_l1();

        log::info!("Attempting L1 reconstruction at depth {}", depth);
        let points = rayon_async(move || l1_interpolate(indices, shares, degree, num_faults)).await;
        let points = match points {
            Ok(points) => points,
            Err(err) => {
                log::error!("Abandoning the reconstruction at depth {}: {}", depth, err);
                return;
            }
        };

        let ser: Vec<LargeFieldSer> = points.iter().map(|p| p.ser_be()).collect();
        let bytes = bincode::serialize(&ser).unwrap();
        self.broadcast(ProtMsg::ReconL2(bytes, depth)).await;
        self.verify_recon_l2(depth).await;
    }

    /// L2: the points a sender reconstructed on each chunk polynomial.
    pub async fn handle_recon_l2(&mut self, bytes: Vec<u8>, depth: usize, sender: Replica) {
        let ser: BincodeResult<Vec<LargeFieldSer>> = bincode::deserialize(&bytes);
        let points: Vec<FieldElement<F>> = match ser {
            Ok(ser) => ser.into_iter().map(|s| F::from_bytes_be(&s).unwrap()).collect(),
            Err(e) => {
                log::error!("Error deserializing L2 reconstruction points from {} at depth {}: {:?}", sender, depth, e);
                return;
            }
        };
        // L1 evaluated the chunk polynomials at the party points, so this is
        // the sender's point on every chunk polynomial.
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());

        let recon = self.mult_state.recon(depth);
        if !recon.accepts_l2() {
            return;
        }
        recon.init_chunks(points.len());
        if points.len() != recon.num_chunks {
            log::error!(
                "L2 reconstruction points from {} at depth {} cover {} chunks, expected {}; dropped",
                sender, depth, points.len(), recon.num_chunks
            );
            return;
        }
        recon.l2_shares.0.push(evaluation_point);
        for (index, point) in points.into_iter().enumerate() {
            recon.l2_shares.1[index].push(point);
        }
        recon.recv_l2 += 1;

        self.verify_recon_l2(depth).await;
    }

    /// Recover every chunk's values once `n − t` L2 points are in, and
    /// broadcast their hash.
    async fn verify_recon_l2(&mut self, depth: usize) {
        let threshold = self.num_nodes - self.num_faults;
        let recon = self.mult_state.recon(depth);
        if !recon.accepts_l2() || !recon.own_init_done() || recon.recv_l2 < threshold {
            return;
        }
        let (indices, points) = recon.take_l2();

        log::info!("Attempting L2 reconstruction at depth {}", depth);
        let values = rayon_async(move || l2_interpolate(indices, points)).await;
        let hash = hash_of(&values);

        let recon = self.mult_state.recon(depth);
        recon.values = values;
        recon.values_ready = true;
        log::info!("Reconstructed {} values at depth {}, broadcasting hash {:?}", recon.values.len(), depth, hash);

        self.broadcast(ProtMsg::ReconHash(hash, depth)).await;
        self.verify_recon_termination(depth).await;
    }

    /// Hash agreement over the reconstructed values.
    pub async fn handle_recon_hash(&mut self, hash: Hash, depth: usize, sender: Replica) {
        let recon = self.mult_state.recon(depth);
        if recon.terminated {
            return;
        }
        recon.hash_votes.insert(hash);
        recon.hash_voters.push(sender);
        self.verify_recon_termination(depth).await;
    }

    /// Terminate once `n − t` parties agree on the values and this party has
    /// them: trim the padding, move them out, and hand them to the consumer.
    async fn verify_recon_termination(&mut self, depth: usize) {
        let threshold = self.num_nodes - self.num_faults;
        let recon = self.mult_state.recon(depth);
        if recon.terminated || !recon.own_init_done() || !recon.values_ready {
            return;
        }
        if recon.hash_voters.len() < threshold {
            return;
        }
        if recon.hash_votes.len() != 1 {
            log::error!(
                "Parties disagree on the values reconstructed at depth {} ({} distinct hashes), abandoning the protocol",
                depth, recon.hash_votes.len()
            );
            return;
        }
        let kind = recon.kind.unwrap();
        let padding = recon.padding.unwrap();
        let mut values = recon.take_values();
        values.truncate(values.len() - padding);
        log::info!("Public reconstruction at depth {} terminated with {} values ({:?})", depth, values.len(), kind);

        // Boxed: every completion can start the next reconstruction, which
        // re-enters this module.
        match kind {
            ReconKind::Multiplication => Box::pin(self.complete_linear_multiplication(depth, values)).await,
            ReconKind::RandBit => Box::pin(self.complete_rand_bit_reconstruction(values)).await,
            ReconKind::Reveal => Box::pin(self.complete_reveal(depth, values)).await,
        }
    }
}
