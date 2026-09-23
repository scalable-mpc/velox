//! Run the reveal probe on the Velox engine.

use reveal_probe::RevealProbe;
use velox::{bail, ArgMatches, EngineOptions, ExitSender, Node, ProtocolField, Result};

fn main() -> Result<()> {
    let matches = velox::engine_args("reveal-probe")
        .arg(velox::arg("values", "v", "inputs each party deals and reveals (default 4)", false))
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

fn start_over_field(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    match matches.value_of("field").unwrap_or("m61") {
        "m61" | "mersenne61" => start::<velox::fields::DefaultField>(config, matches),
        "m61base" => start::<velox::fields::Mersenne61Field>(config, matches),
        "m31" | "mersenne31" => start::<velox::fields::Mersenne31Degree8ExtensionField>(config, matches),
        "m31base" => start::<velox::fields::Mersenne31Field>(config, matches),
        "stark252" => start::<velox::fields::Stark252Field>(config, matches),
        "bn254" => start::<velox::fields::BN254Field>(config, matches),
        other => bail!(
            "unknown field {:?}; expected one of m61, m61base, m31, m31base, stark252, bn254",
            other
        ),
    }
}

fn start<F: ProtocolField>(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    let values = matches
        .value_of("values")
        .map(|v| v.parse::<usize>().map_err(|_| anyhow::anyhow!("--values must be a positive integer")))
        .transpose()?
        .unwrap_or(4);
    let app = RevealProbe::<F>::new(config.num_nodes, config.id, values)?;
    // The application talks to the Planner, which is what the engine hosts.
    velox::spawn(config, velox::Planner::new(app)?, &EngineOptions::from_matches(matches)?)
}
