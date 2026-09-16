use crypto::hash::Hash;
use fields::LargeFieldSer;
use serde::{Serialize, Deserialize};
use types::Replica;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ProtMsg{
    /// Public reconstruction at a depth, structured as two levels: L1 carries
    /// this party's share of every chunk polynomial at the recipient's point
    /// (privately addressed), L2 broadcasts the points each party recovered
    /// from L1, and the hash pins down the values everyone recovered from L2.
    /// Carries the linear multiplication, the random-bit batch and reveals;
    /// see `protocol::public_reconstruction`.
    ReconL1(Vec<u8>, usize),
    ReconL2(Vec<u8>, usize),
    ReconHash(Hash, usize),

    QuadShares(Vec<u8>, usize),

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