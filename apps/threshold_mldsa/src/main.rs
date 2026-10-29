//! Threshold ML-DSA-44 on the Velox engine, over the ML-DSA field `F_q`.
//!
//! Each run makes a new key with the parties and logs the public key.

use threshold_mldsa::ThresholdMlDsa;
use velox::{bail, EngineOptions, Planner, Result};

fn main() -> Result<()> {
    let matches = velox::engine_args("threshold_mldsa").get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;
        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => velox::spawn(config, Planner::new(ThresholdMlDsa::new())?, &EngineOptions::from_matches(&matches)?),
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}
