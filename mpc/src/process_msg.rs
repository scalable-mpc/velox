use application::Application;
use std::sync::Arc;

use crate::{context::Context, msg::ProtMsg};
use crypto::{hash::verf_mac};
use types::{WrapperMsg};
use fields::ProtocolField;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    // This function verifies the Message Authentication Code (MAC) of a sent message
    // A node cannot impersonate as another node because of MACs
    pub fn check_proposal(&self, wrapper_msg: Arc<WrapperMsg<ProtMsg>>) -> bool {
        // validate MAC
        let byte_val =
            bincode::serialize(&wrapper_msg.protmsg).expect("Failed to serialize object");
        let sec_key = match self.sec_key_map.get(&wrapper_msg.clone().sender) {
            Some(val) => val,
            None => {
                panic!("Secret key not available, this shouldn't happen")
            }
        };
        if !verf_mac(&byte_val, &sec_key.as_slice(), &wrapper_msg.mac) {
            log::warn!("MAC Verification failed.");
            return false;
        }
        true
    }

    pub(crate) async fn process_msg(&mut self, wrapper_msg: WrapperMsg<ProtMsg>) {
        log::trace!("Received protocol msg: {:?}", wrapper_msg);
        let msg = Arc::new(wrapper_msg.clone());

        // Verify the message's authenticity before proceeding
        if self.check_proposal(msg) {
            match wrapper_msg.clone().protmsg {
                ProtMsg::ReconL1(bytes, depth) => {
                    log::debug!("Received reconstruction L1 shares for depth {} from node : {}", depth, wrapper_msg.sender);
                    self.handle_recon_l1(bytes, depth, wrapper_msg.sender).await;
                }
                ProtMsg::ReconL2(bytes, depth) => {
                    log::debug!("Received reconstruction L2 points for depth {} from node : {}", depth, wrapper_msg.sender);
                    self.handle_recon_l2(bytes, depth, wrapper_msg.sender).await;
                }
                ProtMsg::ReconHash(hash, depth) => {
                    log::debug!("Received reconstruction hash for depth {} from node : {}", depth, wrapper_msg.sender);
                    self.handle_recon_hash(hash, depth, wrapper_msg.sender).await;
                }
                ProtMsg::QuadShares(main_msg, depth) => {
                    log::debug!("Received quadratic shares for depth {} from node : {}", depth, wrapper_msg.sender);
                    self.handle_quadratic_mult_shares(depth,main_msg, wrapper_msg.sender).await;
                },
                ProtMsg::ReconstructRandBits(shares)=>{
                    log::debug!("Received ReconstructRandBits message");
                    self.handle_reconstruct_rand_bits_verify(shares, wrapper_msg.sender).await;
                },
                ProtMsg::HashZMsg(hash_val, depth, lin_or_quad) => {
                    // RBC initialized
                    log::debug!("Received HashZMsg for depth {} from node : {}", depth, wrapper_msg.sender);
                    self.handle_hash_broadcast(hash_val, depth, lin_or_quad, wrapper_msg.sender).await;
                },
                ProtMsg::ReconstructCoin(ser_share, depth) => {
                    log::debug!("Received ReconstructCoin message");
                    self.handle_common_coin_msg(ser_share, wrapper_msg.sender, depth).await;
                },
                ProtMsg::ReconstructVerfOutputSharing(ser_x_share, ser_y_share, ser_z_share)=>{
                    log::debug!("Received ReconstructVerfOutputSharing message");
                    self.handle_reconstruct_verf_output_sharing(ser_x_share, ser_y_share, ser_z_share, wrapper_msg.sender).await;
                },
                ProtMsg::ReconstructMaskedOutput(ser_shares) =>{
                    log::debug!("Received ReconstructMaskedOutput message");
                    self.handle_reconstruct_masked_output(ser_shares, wrapper_msg.sender).await;
                },
                ProtMsg::ReconstructOutputMasks(_origin, _ser_shares, _ser_nonce, _ser_blinding)=>{
                    log::debug!("Received ReconstructOutputMasks message");
                    self.handle_random_mask_shares(wrapper_msg.sender, _origin, _ser_shares,_ser_nonce, _ser_blinding).await;
                },
                ProtMsg::ReconstructMultSharings(_shares, _index) => {
                    log::debug!("Received ReconstructMultSharings message");
                    //self.handle_reconstruct_mult_sharings(shares, index, wrapper_msg.sender).await;
                },
            }
        } else {
            log::warn!(
                "MAC Verification failed for message {:?}",
                wrapper_msg.protmsg
            );
        }
    }
}
