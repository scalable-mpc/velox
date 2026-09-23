//! Run private ERC20 payments on the Velox engine, through the Planner.
//!
//! `--ledger` is the public ledger every party reads (see `erc20::Ledger`).
//! This party's secret inputs — its accounts' balances, then its accounts'
//! transfer amounts — are read from `testdata/inputs/erc20_input_<id>.txt`,
//! one signed decimal per line.

use erc20::{Erc20, Ledger};
use velox::{bail, ArgMatches, EngineOptions, ExitSender, MersennePrimeField, Node, Planner, ProtocolField, Result};

fn main() -> Result<()> {
    let matches = velox::engine_args("erc20")
        .arg(velox::arg("ledger", "l", "path to the public ledger file", true))
        .get_matches();

    velox::run_node(|| {
        velox::init_logging()?;
        let config = velox::load_config(&matches)?;
        match matches.value_of("protocol").unwrap_or("mpc") {
            "sync" => velox::spawn_syncer(&config, &matches),
            "mpc" => match matches.value_of("field").unwrap_or("m61base") {
                "m61base" => start::<velox::fields::Mersenne61Field>(config, &matches),
                "m31base" => start::<velox::fields::Mersenne31Field>(config, &matches),
                other => bail!("erc20 compares balances, which needs a Mersenne-prime field: --field m61base or m31base, not {:?}", other),
            },
            other => bail!("unknown --protocol {:?}; expected mpc or sync", other),
        }
    })
}

fn start<F: ProtocolField + MersennePrimeField>(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    let path = matches.value_of("ledger").ok_or_else(|| anyhow::anyhow!("--ledger is required"))?;
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading ledger {}: {}", path, e))?;
    let app = Erc20::<F>::new(config.num_nodes, config.id, Ledger::parse(&text)?);

    let count = app.inputs_per_party();
    let primary = format!("testdata/inputs/erc20_input_{}.txt", config.id);
    let fallback = format!("erc20_input_{}.txt", config.id);
    let inputs = velox::input::read_numeric_input_from_files::<F>(&primary, &fallback, count).unwrap_or_else(|e| {
        log::error!("Erc20: cannot read {} inputs: {}; dealing zeros", count, e);
        Vec::new()
    });

    velox::spawn(config, Planner::new(app.with_inputs(inputs))?, &EngineOptions::from_matches(matches)?)
}
