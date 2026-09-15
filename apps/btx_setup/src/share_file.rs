//! The file party `j` writes at the end of the setup: its Shamir shares
//! `⟨τ¹⟩_j … ⟨τ^{2B}⟩_j`, plus enough context for a reader to know what they
//! are shares *of*. (Notation as in the crate root: `⟨x⟩_j` is party `j`'s
//! degree-`t` Shamir share of the field element `x`.)
//!
//! Indices `1..=B` are the party's BTX/BTE secret key `sk_j`. The rest exist
//! only to be lifted into G₂ — `g₂^{⟨τ^i⟩_j}`, and `e(g₁^{⟨τ^{B+1}⟩_j}, g₂)`
//! for the punctured power — to build the public keys `h_i`, `v_j^i` and
//! `ek`; they are written to the same file because that lifting is the
//! party's own job, done from its own shares.
//!
//! Elements are the field's big-endian bytes as lower-case hex, so the file is
//! readable and diffable, and so that a reader over the same field can
//! `from_bytes_be` them back without knowing anything about the writer.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use velox::{FieldElement, ProtocolField};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ShareFile {
    /// Name of the field the shares live in, as the binary's `--field` spells
    /// it. The exponent-side tools refuse anything but the BLS12-381 scalar
    /// field; other fields only exist here for local benchmarking.
    pub field: String,
    pub num_nodes: usize,
    /// The engine's `t`, so a reader knows how many files reconstruct.
    pub threshold: usize,
    /// `B`: the scheme's (maximum) batch size. The file holds `2B` shares.
    pub batch_size: usize,
    /// This party's id, `0..num_nodes`. Its Shamir evaluation point is
    /// `party + 1` — the engine's convention (`use_fft` is off), which a reader
    /// combining files has to use too.
    pub party: usize,
    /// `shares[i - 1]` is `⟨τ^i⟩_party`, `i = 1..=2B`.
    pub shares: Vec<String>,
}

impl ShareFile {
    pub fn new<F: ProtocolField>(
        field: &str,
        num_nodes: usize,
        threshold: usize,
        batch_size: usize,
        party: usize,
        shares: &[FieldElement<F>],
    ) -> Result<Self> {
        if shares.len() != 2 * batch_size {
            bail!(
                "share file for batch size {} needs {} shares, got {}",
                batch_size,
                2 * batch_size,
                shares.len()
            );
        }
        Ok(Self {
            field: field.to_string(),
            num_nodes,
            threshold,
            batch_size,
            party,
            shares: shares.iter().map(|s| hex_encode(&F::to_bytes_be(s))).collect(),
        })
    }

    /// The shares as field elements, in the file's order.
    pub fn shares<F: ProtocolField>(&self) -> Result<Vec<FieldElement<F>>> {
        self.shares
            .iter()
            .enumerate()
            .map(|(index, hex)| {
                let bytes = hex_decode(hex)
                    .with_context(|| format!("share {} of party {}", index + 1, self.party))?;
                F::from_bytes_be(&bytes)
                    .map_err(|e| anyhow::anyhow!("share {} of party {}: {:?}", index + 1, self.party, e))
            })
            .collect()
    }

    /// The conventional file name for a party's shares under `dir`.
    pub fn path(dir: &Path, party: usize) -> std::path::PathBuf {
        dir.join(format!("btx_setup_{}.json", party))
    }

    pub fn write(&self, dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = Self::path(dir, self.party);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        bail!("odd-length hex string");
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).with_context(|| format!("bad hex at offset {}", i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = velox::fields::BLS12381ScalarField;

    #[test]
    fn round_trips_through_json_and_hex() {
        let shares: Vec<FieldElement<F>> = (0..8).map(|_| F::rand()).collect();
        let file = ShareFile::new::<F>("bls381", 4, 1, 4, 2, &shares).unwrap();
        let json = serde_json::to_string(&file).unwrap();
        let back: ShareFile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, file);
        assert_eq!(back.shares::<F>().unwrap(), shares);
    }

    #[test]
    fn rejects_the_wrong_share_count() {
        let shares: Vec<FieldElement<F>> = (0..7).map(|_| F::rand()).collect();
        assert!(ShareFile::new::<F>("bls381", 4, 1, 4, 0, &shares).is_err());
    }

    #[test]
    fn writes_and_reads_a_file() {
        let dir = std::env::temp_dir().join(format!("btx_share_file_{}", std::process::id()));
        let shares: Vec<FieldElement<F>> = (0..2).map(|_| F::rand()).collect();
        let file = ShareFile::new::<F>("bls381", 4, 1, 1, 3, &shares).unwrap();
        let path = file.write(&dir).unwrap();
        assert_eq!(path, dir.join("btx_setup_3.json"));
        assert_eq!(ShareFile::read(&path).unwrap(), file);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
