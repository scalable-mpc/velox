//! Erasure coding for AVID, backed by [`commonware_coding`].
//!
//! Replaces `consensus::get_shards` / `consensus::reconstruct_data`
//! (`reed-solomon-erasure`, scalar GF(2^8)) with commonware's vendored
//! `reed-solomon-simd` — Leopard-RS over GF(2^16), O(n log n) — together with
//! its Merkle commitment. `encode` returns a single 32-byte commitment plus `n`
//! shards, each of which already carries an inclusion proof against it, so AVID
//! no longer builds a per-recipient Merkle tree of its own. The master tree
//! over the per-recipient commitments stays: that is what binds one batch
//! together, and it is not something the coder provides.
//!
//! The `consensus` crate this used to call is pulled in from a pinned git URL,
//! so the coding lives here rather than being changed there.
//!
//! # Wire format
//!
//! [`Shard`] and [`Commitment`] travel inside this repository's
//! `serde`/`bincode` messages, but commonware types implement
//! `commonware_codec` rather than `serde`. [`Shard`] bridges the two: it
//! serializes as the byte string commonware's codec produces and is read back
//! with [`MAX_SHARD_SIZE`] as the length bound. [`Commitment`] is a SHA-256
//! digest carried as a plain `[u8; 32]`, so it is directly usable wherever a
//! Merkle root was before.

use std::fmt;
use std::num::NonZeroU16;

use commonware_codec::{Decode, Encode};
use commonware_coding::{CodecConfig, Config, ReedSolomon, Scheme};
use commonware_cryptography::{sha256::Digest as Sha256Digest, Sha256};
use commonware_parallel::Sequential;
use crypto::hash::Hash;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The coding scheme.
type Coder = ReedSolomon<Sha256>;

/// Parallelism strategy handed to commonware for a *single* message.
///
/// Deliberately sequential. The dealer codes one message per recipient and
/// [`encode_batch`] already spreads those `n` independent encodings across
/// rayon, which saturates the pool at the level where the work is embarrassingly
/// parallel and needs no coordination inside the coder. Asking commonware to
/// stripe each individual encode on top of that only adds join overhead.
const STRATEGY: Sequential = Sequential;

/// Upper bound accepted when reading a [`Shard`] off the wire, in bytes.
///
/// This bounds the allocation a malformed message can request; it is not a
/// protocol parameter. The previous `shard: Vec<u8>` field had no bound at all.
pub const MAX_SHARD_SIZE: usize = 256 * 1024 * 1024;

/// A commitment to a full set of shards: the root of commonware's binary Merkle
/// tree over the shard hashes.
///
/// Same 32 bytes AVID previously used as a per-recipient Merkle root, so it
/// drops into the master tree unchanged.
pub type Commitment = Hash;

/// A shard verified against a [`Commitment`].
///
/// Produced by [`check`] and consumed by [`decode`]. Not serializable by
/// design: it is evidence that *this* process verified the shard, so it must
/// not be accepted from the network.
pub type CheckedShard = <Coder as Scheme>::CheckedShard;

/// One shard of an erasure-coded message, with its Merkle inclusion proof
/// against the [`Commitment`].
///
/// Replaces the old `(Vec<u8>, crypto::aes_hash::Proof)` pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard(<Coder as Scheme>::Shard);

impl Shard {
    /// Size of this shard on the wire, in bytes.
    ///
    /// A shard does not expose its index, deliberately: a shard's position is
    /// asserted by the verifier in [`check`], never read out of the shard.
    pub fn wire_size(&self) -> usize {
        self.0.encode().len()
    }
}

impl Serialize for Shard {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = self.0.encode();
        serializer.serialize_bytes(bytes.as_ref())
    }
}

impl<'de> Deserialize<'de> for Shard {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = <Vec<u8>>::deserialize(deserializer)?;
        let cfg = CodecConfig {
            maximum_shard_size: MAX_SHARD_SIZE,
        };
        <Coder as Scheme>::Shard::decode_cfg(bytes.as_slice(), &cfg)
            .map(Shard)
            .map_err(serde::de::Error::custom)
    }
}

/// Errors returned by the coding layer.
#[derive(Debug)]
pub enum Error {
    /// `data_shards` / `parity_shards` cannot be used with this coder. Both
    /// must be non-zero and their sum at most 65536.
    InvalidParameters {
        data_shards: usize,
        parity_shards: usize,
    },
    /// The shard did not verify against the commitment at the given index.
    InvalidShard { index: usize },
    /// Reconstruction produced data that does not match the commitment, or the
    /// coder itself failed.
    Coding(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParameters {
                data_shards,
                parity_shards,
            } => write!(
                f,
                "unsupported erasure code parameters: {} data + {} parity shards",
                data_shards, parity_shards
            ),
            Error::InvalidShard { index } => write!(f, "shard {} failed verification", index),
            Error::Coding(msg) => write!(f, "coding error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

/// Build a commonware [`Config`] from the `(data, parity)` split AVID uses.
fn config(data_shards: usize, parity_shards: usize) -> Result<Config, Error> {
    let invalid = || Error::InvalidParameters {
        data_shards,
        parity_shards,
    };
    if data_shards + parity_shards > u16::MAX as usize {
        return Err(invalid());
    }
    let minimum_shards =
        NonZeroU16::new(u16::try_from(data_shards).map_err(|_| invalid())?).ok_or_else(invalid)?;
    let extra_shards =
        NonZeroU16::new(u16::try_from(parity_shards).map_err(|_| invalid())?).ok_or_else(invalid)?;
    Ok(Config {
        minimum_shards,
        extra_shards,
    })
}

/// Erasure-code `data` into `data_shards + parity_shards` shards and commit to
/// them.
///
/// Shards come back in index order: shard `i` is the one node `i` must hold,
/// and it will only verify at that index. Any `data_shards` of them reconstruct
/// the message *exactly* — commonware prefixes the payload with its length, so
/// unlike `get_shards` there is no trailing zero padding for the caller to
/// account for.
pub fn encode(
    data: &[u8],
    data_shards: usize,
    parity_shards: usize,
) -> Result<(Commitment, Vec<Shard>), Error> {
    let config = config(data_shards, parity_shards)?;
    let (commitment, shards) =
        Coder::encode(&config, data, &STRATEGY).map_err(|e| Error::Coding(format!("{:?}", e)))?;
    Ok((commitment.0, shards.into_iter().map(Shard).collect()))
}

/// Erasure-code a whole batch of messages in parallel, one per recipient.
///
/// This is the dealer's dominant cost in `start_init`: `n` messages, each coded
/// into `n` shards and committed, for `O(n^2)` shard work per dissemination.
/// The messages are independent, so they go straight onto rayon.
///
/// Results stay in input order, so `result[i]` belongs to `messages[i]`.
///
/// Call [`encode_batch_async`] from async code — this function blocks until the
/// whole batch is done.
pub fn encode_batch(
    messages: &[Vec<u8>],
    data_shards: usize,
    parity_shards: usize,
) -> Result<Vec<(Commitment, Vec<Shard>)>, Error> {
    // Validate once up front so a bad split is not reported `n` times over.
    config(data_shards, parity_shards)?;
    messages
        .par_iter()
        .map(|message| encode(message, data_shards, parity_shards))
        .collect()
}

/// [`encode_batch`], run on rayon's pool without blocking the caller.
///
/// `par_iter` called directly from an `async fn` does not yield: it injects the
/// job and then parks the calling thread on a latch, so the tokio worker
/// running that task is unavailable until the whole batch finishes and the
/// task's inbox keeps growing behind it (see `TODO.md`). Handing the batch to
/// `rayon::spawn` and awaiting a oneshot suspends the future instead, leaving
/// the worker free to drive other tasks.
pub async fn encode_batch_async(
    messages: Vec<Vec<u8>>,
    data_shards: usize,
    parity_shards: usize,
) -> Result<Vec<(Commitment, Vec<Shard>)>, Error> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn(move || {
        // Send failure only means the receiver was dropped, i.e. the awaiting
        // task went away; there is nothing to do about it here.
        let _ = tx.send(encode_batch(&messages, data_shards, parity_shards));
    });
    rx.await.unwrap_or_else(|_| {
        Err(Error::Coding(
            "the rayon encoding task was dropped before it produced a result".to_string(),
        ))
    })
}

/// Verify `shard` against `commitment` at `index`.
///
/// `index` is the position the shard was encoded at, which is the identity of
/// the node that holds it. A shard verifies at exactly one index, so this also
/// authenticates the holder's claim to that position.
pub fn check(
    commitment: &Commitment,
    index: usize,
    shard: &Shard,
    data_shards: usize,
    parity_shards: usize,
) -> Result<CheckedShard, Error> {
    let config = config(data_shards, parity_shards)?;
    let index = u16::try_from(index).map_err(|_| Error::InvalidShard { index })?;
    Coder::check(&config, &Sha256Digest(*commitment), index, &shard.0).map_err(|_| {
        Error::InvalidShard {
            index: index as usize,
        }
    })
}

/// Reconstruct the message from at least `data_shards` verified shards.
///
/// commonware re-derives the commitment while decoding and rejects the result
/// if it does not match, so a successful return means the message is bound to
/// `commitment`. The explicit "rebuild the Merkle tree and compare roots" step
/// AVID performed afterwards is therefore redundant. Shards checked against a
/// *different* commitment are rejected outright, so a node that forwards
/// somebody else's fragment makes this fail rather than corrupt the result.
pub fn decode<'a, I>(
    commitment: &Commitment,
    shards: I,
    data_shards: usize,
    parity_shards: usize,
) -> Result<Vec<u8>, Error>
where
    I: IntoIterator<Item = &'a CheckedShard>,
{
    let config = config(data_shards, parity_shards)?;
    Coder::decode(
        &config,
        &Sha256Digest(*commitment),
        shards.into_iter(),
        &STRATEGY,
    )
    .map_err(|e| Error::Coding(format!("{:?}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split AVID uses: `n = 3f + 1`, `k = n - 2f` data shards, `2f` parity.
    fn split(n: usize) -> (usize, usize) {
        let f = (n - 1) / 3;
        (n - 2 * f, 2 * f)
    }

    #[test]
    fn round_trips_from_k_survivors() {
        for n in [4usize, 7, 16, 64] {
            let (k, m) = split(n);
            let message: Vec<u8> = (0..7000u32).map(|i| (i % 251) as u8).collect();
            let (commitment, shards) = encode(&message, k, m).unwrap();
            assert_eq!(shards.len(), n);

            // Decode from the last k, so at least one parity shard is involved
            // and interpolation actually happens.
            let checked: Vec<CheckedShard> = shards
                .iter()
                .enumerate()
                .skip(n - k)
                .map(|(i, s)| check(&commitment, i, s, k, m).unwrap())
                .collect();
            assert_eq!(checked.len(), k);

            let decoded = decode(&commitment, checked.iter(), k, m).unwrap();
            assert_eq!(decoded, message, "n={}", n);
        }
    }

    /// AVID reconstructs once `k` READY shards have arrived, which in general
    /// is more than the minimum the coder needs.
    #[test]
    fn decodes_with_more_than_k_shards() {
        let n = 16;
        let (k, m) = split(n);
        let f = (n - 1) / 3;
        let message: Vec<u8> = (0..4096u32).map(|i| (i % 253) as u8).collect();
        let (commitment, shards) = encode(&message, k, m).unwrap();

        let checked: Vec<CheckedShard> = shards
            .iter()
            .enumerate()
            .skip(f)
            .map(|(i, s)| check(&commitment, i, s, k, m).unwrap())
            .collect();
        assert!(checked.len() > k);
        assert_eq!(decode(&commitment, checked.iter(), k, m).unwrap(), message);
    }

    #[test]
    fn a_shard_only_verifies_at_its_own_index() {
        let (k, m) = split(16);
        let (commitment, shards) = encode(b"authenticated position", k, m).unwrap();
        assert!(check(&commitment, 3, &shards[3], k, m).is_ok());
        assert!(check(&commitment, 4, &shards[3], k, m).is_err());
    }

    #[test]
    fn a_shard_from_another_message_is_rejected() {
        let (k, m) = split(16);
        let (commitment, _) = encode(b"the message we committed to", k, m).unwrap();
        let (_, other) = encode(b"a different message entirely", k, m).unwrap();
        assert!(check(&commitment, 0, &other[0], k, m).is_err());
    }

    #[test]
    fn shards_survive_the_serde_wire() {
        let (k, m) = split(16);
        let (commitment, shards) = encode(b"over the wire", k, m).unwrap();
        for (index, shard) in shards.iter().enumerate() {
            let bytes = bincode::serialize(shard).unwrap();
            let restored: Shard = bincode::deserialize(&bytes).unwrap();
            assert_eq!(&restored, shard);
            check(&commitment, index, &restored, k, m).unwrap();
        }
    }

    /// The batch is the dealer's whole `start_init` workload: one message per
    /// recipient. It must agree with encoding them one at a time, and stay in
    /// input order.
    #[test]
    fn batch_encoding_matches_sequential_encoding() {
        let n = 16;
        let (k, m) = split(n);
        let messages: Vec<Vec<u8>> = (0..n)
            .map(|i| (0..1500u32).map(|j| ((i as u32 * 31 + j) % 251) as u8).collect())
            .collect();

        let batched = encode_batch(&messages, k, m).unwrap();
        assert_eq!(batched.len(), n);
        for (index, message) in messages.iter().enumerate() {
            let expected = encode(message, k, m).unwrap();
            assert_eq!(batched[index], expected, "message {}", index);
        }
    }

    #[test]
    fn batch_encoding_rejects_a_bad_split() {
        assert!(encode_batch(&[vec![1u8; 32]], 4, 0).is_err());
        assert!(encode_batch(&[vec![1u8; 32]], 0, 4).is_err());
    }

    /// The async bridge must hand back exactly what the blocking call would.
    #[tokio::test]
    async fn the_async_bridge_matches_the_blocking_batch() {
        let n = 7;
        let (k, m) = split(n);
        let messages: Vec<Vec<u8>> = (0..n)
            .map(|i| (0..2500u32).map(|j| ((i as u32 * 13 + j) % 251) as u8).collect())
            .collect();

        let expected = encode_batch(&messages, k, m).unwrap();
        assert_eq!(
            encode_batch_async(messages, k, m).await.unwrap(),
            expected
        );
    }

    #[test]
    fn empty_and_tiny_messages_round_trip() {
        let (k, m) = split(16);
        for message in [vec![], vec![7u8], vec![9u8; 3]] {
            let (commitment, shards) = encode(&message, k, m).unwrap();
            let checked: Vec<CheckedShard> = shards
                .iter()
                .enumerate()
                .skip(m)
                .map(|(i, s)| check(&commitment, i, s, k, m).unwrap())
                .collect();
            assert_eq!(decode(&commitment, checked.iter(), k, m).unwrap(), message);
        }
    }
}
