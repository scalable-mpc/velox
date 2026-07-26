use crypto::hash::Hash;
use fields::LargeFieldSer;
use serde::{Serialize, Deserialize};
use types::Replica;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ProtMsg{
    // Encrypted shares, and the depth of the circuit
    SharesL1(Vec<u8>, usize),
    SharesL2(Vec<u8>, usize),

    QuadShares(Vec<u8>, usize),

    // Public reconstruction of the squared random sharings during random bit
    // preparation, structured like the linear multiplication protocol:
    // L1 carries this party's evaluation of every chunk polynomial at the
    // recipient's point (privately addressed), L2 broadcasts the points each
    // party reconstructed from L1, and the hash pins down the values everyone
    // recovered from L2.
    RandBitReconL1(Vec<u8>),
    RandBitReconL2(Vec<u8>),
    RandBitReconHash(Hash),

    // Hash Message to ensure at least t+1 parties are consistent with the hash value
    // Bool is for indicating linear or quadratic layer
    HashZMsg(Hash, usize, bool),
    ReconstructCoin(LargeFieldSer, usize),

    ReconstructVerfOutputSharing(LargeFieldSer, LargeFieldSer, LargeFieldSer),
    ReconstructMaskedOutput(Vec<LargeFieldSer>),

    ReconstructOutputMasks(Replica, Vec<LargeFieldSer>, LargeFieldSer, LargeFieldSer),

    // Temporary for testing
    ReconstructMultSharings(Vec<LargeFieldSer>, usize),
    ReconstructRandBits(Vec<LargeFieldSer>), 
}