//! Threshold ML-DSA-44 on the Velox engine, over the ML-DSA field `F_q`.
//!
//! Each run makes a new key with the parties, logs the public key, and signs
//! `--signatures` messages (default 1) with `--attempts` signing attempts
//! for each (default 16). The messages are `--message` when there is one,
//! and `"<message> 0"`, `"<message> 1"`, ... when there are more. Each party
//! checks each signature with RustCrypto `ml-dsa`.

use threshold_mldsa::ThresholdMlDsa;
use velox::{bail, ArgMatches, EngineOptions, Planner, Result};

/// Decision D2 of issue 19: 16 attempts pass the rejection test with a high
/// probability.
const DEFAULT_ATTEMPTS: usize = 16;

fn main() -> Result<()> {
    let matches = velox::engine_args("threshold_mldsa")
        .arg(velox::arg("attempts", "a", "signing attempts to prepare for each message (default 16)", false))
        .arg(velox::arg("signatures", "s", "messages to sign in the run (default 1)", false))
        .arg(velox::arg("message", "m", "the message to sign (default \"threshold ML-DSA\")", false))
        .get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;
        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => {
                let attempts = positive_count(&matches, "attempts", DEFAULT_ATTEMPTS)?;
                let signatures = positive_count(&matches, "signatures", 1)?;
                let message = matches.value_of("message").unwrap_or("threshold ML-DSA");
                let messages = match signatures {
                    1 => vec![message.as_bytes().to_vec()],
                    s => (0..s).map(|i| format!("{} {}", message, i).into_bytes()).collect(),
                };
                velox::spawn(config, Planner::new(ThresholdMlDsa::new(attempts, messages))?, &EngineOptions::from_matches(&matches)?)
            }
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}

/// The value of the flag `name`, at least 1, or `default` when it is not
/// given.
fn positive_count(matches: &ArgMatches, name: &str, default: usize) -> Result<usize> {
    let count = match matches.value_of(name) {
        Some(v) => v.parse::<usize>().map_err(|_| anyhow::anyhow!("--{} must be a positive integer", name))?,
        None => default,
    };
    if count == 0 {
        bail!("--{} must be at least 1", name);
    }
    Ok(count)
}
