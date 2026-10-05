//! Threshold-decrypt LWE ciphertexts on the Velox engine, over Mersenne-127.
//!
//! `--ciphertexts` is the public file every party reads; every ciphertext in
//! it is decrypted. `--key` is read by party 0 only, which deals the key bits.
//! `testdata/fhe/gen_lwe.py` writes both. `--method` picks how the noise is
//! removed: `carry` (the Planner's `Mod2m`) or `table` (the paper's lookup
//! tables).

use fhe_decrypt::{parse_ciphertexts, FheDecrypt, Method, LWE_DIMENSION};
use velox::{bail, EngineOptions, Planner, Result};

fn main() -> Result<()> {
    let matches = velox::engine_args("fhe_decrypt")
        .arg(velox::arg("ciphertexts", "x", "public ciphertext file (default testdata/fhe/ciphertexts.txt)", false))
        .arg(velox::arg("key", "k", "key file, read by party 0 only (default testdata/fhe/key.txt)", false))
        .arg(velox::arg("method", "m", "how the noise is removed: carry (default) or table", false))
        .get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;
        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => {
                let method = match matches.value_of("method").unwrap_or("carry") {
                    "carry" => Method::Carry,
                    "table" => Method::Table,
                    other => bail!("unknown --method {:?}; expected carry or table", other),
                };
                let path = matches.value_of("ciphertexts").unwrap_or("testdata/fhe/ciphertexts.txt");
                let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {}", path, e))?;
                let ciphertexts = parse_ciphertexts(&text)?;

                let key = if config.id == 0 {
                    let path = matches.value_of("key").unwrap_or("testdata/fhe/key.txt");
                    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {}", path, e))?;
                    let bits = text
                        .split_whitespace()
                        .map(|v| match v {
                            "0" => Ok(0u8),
                            "1" => Ok(1u8),
                            other => bail!("{}: key bit {:?} is not 0 or 1", path, other),
                        })
                        .collect::<Result<Vec<u8>>>()?;
                    if bits.len() != LWE_DIMENSION {
                        bail!("{}: {} key bits, the key has {}", path, bits.len(), LWE_DIMENSION);
                    }
                    Some(bits)
                } else {
                    None
                };

                let app = FheDecrypt::new(ciphertexts, key, method);
                velox::spawn(config, Planner::new(app)?, &EngineOptions::from_matches(&matches)?)
            }
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}
