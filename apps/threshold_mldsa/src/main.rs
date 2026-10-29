//! Threshold ML-DSA-44 on the Velox engine, over the ML-DSA field `F_q`.
//!
//! Each run makes a new key with the parties, logs the public key,
//! prepares `--attempts` signing attempts (default 16), and signs
//! `--message`. Each party checks the signature with RustCrypto `ml-dsa`.

use threshold_mldsa::ThresholdMlDsa;
use velox::{bail, EngineOptions, Planner, Result};

/// Decision D2 of issue 19: 16 attempts pass the rejection test with a high
/// probability.
const DEFAULT_ATTEMPTS: usize = 16;

fn main() -> Result<()> {
    let matches = velox::engine_args("threshold_mldsa")
        .arg(velox::arg("attempts", "a", "signing attempts to prepare (default 16)", false))
        .arg(velox::arg("message", "m", "the message to sign (default \"threshold ML-DSA\")", false))
        .get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;
        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => {
                let attempts = match matches.value_of("attempts") {
                    Some(a) => a.parse::<usize>().map_err(|_| anyhow::anyhow!("--attempts must be a positive integer"))?,
                    None => DEFAULT_ATTEMPTS,
                };
                if attempts == 0 {
                    bail!("--attempts must be at least 1");
                }
                let message = matches.value_of("message").unwrap_or("threshold ML-DSA").as_bytes().to_vec();
                velox::spawn(config, Planner::new(ThresholdMlDsa::new(attempts, message))?, &EngineOptions::from_matches(&matches)?)
            }
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}
