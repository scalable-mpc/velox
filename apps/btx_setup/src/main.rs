//! Generate a party's BTX/BTE setup shares on the Velox engine.
//!
//! The application owns its binary: velox supplies argument parsing, config
//! loading, the tokio runtime, the syncer and signal handling, and this file
//! supplies what is specific to this application — the batch size and where
//! the share file goes.

use std::path::PathBuf;

use btx_setup::BtxSetup;
use velox::{bail, ArgMatches, EngineOptions, ExitSender, Node, ProtocolField, Result};

fn main() -> Result<()> {
    let matches = velox::engine_args("btx-setup")
        .arg(velox::arg(
            "batch_size",
            "B",
            "the scheme's batch size B; shares of tau^1..tau^{2B} are generated",
            true,
        ))
        .arg(velox::arg(
            "out",
            "o",
            "directory the share file btx_setup_<id>.json is written to",
            true,
        ))
        .get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;

        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => start_over_field(config, &matches),
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}

/// Start the setup over whichever field `--field` names.
///
/// The scheme lives in BLS12-381's scalar field, which is the default; the
/// exponent-side tools refuse a share file over any other field. The others
/// are accepted so the circuit can be benchmarked at the engine's cheaper
/// fields, and the file records which one was used.
fn start_over_field(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    let field = matches.value_of("field").unwrap_or("bls381");
    match field {
        "bls381" | "bls12-381" => start::<velox::fields::BLS12381ScalarField>(config, matches, field),
        "m61" | "mersenne61" => start::<velox::fields::DefaultField>(config, matches, field),
        "m61base" => start::<velox::fields::Mersenne61Field>(config, matches, field),
        "stark252" => start::<velox::fields::Stark252Field>(config, matches, field),
        "bn254" => start::<velox::fields::BN254Field>(config, matches, field),
        other => bail!(
            "unknown field {:?}; expected one of bls381, m61, m61base, stark252, bn254",
            other
        ),
    }
}

fn start<F: ProtocolField>(config: Node, matches: &ArgMatches, field: &str) -> Result<ExitSender> {
    let batch_size = matches
        .value_of("batch_size")
        .ok_or_else(|| anyhow::anyhow!("--batch_size is required"))?
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("--batch_size must be a positive integer"))?;
    let out = PathBuf::from(
        matches
            .value_of("out")
            .ok_or_else(|| anyhow::anyhow!("--out is required"))?,
    );

    let app = BtxSetup::<F>::new(config.num_nodes, config.num_faults, config.id, batch_size)?
        .with_output(out, field);

    velox::spawn(config, app, &EngineOptions::from_matches(matches)?)
}
