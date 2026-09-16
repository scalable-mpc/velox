//! Evaluate a Bristol-style `.arith` arithmetic circuit on the Velox engine.
//!
//! The application owns its binary: velox supplies argument parsing, config
//! loading, the tokio runtime, the syncer and signal handling, and this file
//! supplies what is specific to this application — parsing the circuit and
//! giving it this party's input wires.

use bristol_circuit::BristolCircuit;
use velox::{bail, ArgMatches, EngineOptions, ExitSender, FieldElement, Node, ProtocolField, Result};

fn main() -> Result<()> {
    // `--circuit` belongs to this application. It used to sit in the engine's
    // shared argument list alongside anonymous broadcast's `--messages`, so each
    // node had to be handed a flag the other application ignored.
    let matches = velox::engine_args("bristol-circuit")
        .arg(velox::arg(
            "circuit",
            "x",
            "path to the .arith circuit to evaluate",
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

/// Start the circuit over whichever field `--field` names.
///
/// Each arm is the *same* application at a different field. See the note in
/// `anonymous_broadcast`'s binary for why the arms are spelled out rather than
/// abstracted behind a trait.
fn start_over_field(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    match matches.value_of("field").unwrap_or("m61") {
        "m61" | "mersenne61" => start::<velox::fields::DefaultField>(config, matches),
        "stark252" => start::<velox::fields::Stark252Field>(config, matches),
        "bn254" => start::<velox::fields::BN254Field>(config, matches),
        "m61base" => start::<velox::fields::Mersenne61Field>(config, matches),
        "m31" | "mersenne31" => start::<velox::fields::Mersenne31Degree8ExtensionField>(config, matches),
        "m31base" => start::<velox::fields::Mersenne31Field>(config, matches),
        other => bail!(
            "unknown field {:?}; expected one of m61, m61base, m31, m31base, stark252, bn254",
            other
        ),
    }
}

fn start<F: ProtocolField>(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    let path = matches
        .value_of("circuit")
        .ok_or_else(|| anyhow::anyhow!("--circuit is required"))?;

    // A malformed circuit is fatal: every party must evaluate the same circuit,
    // so there is nothing sensible to fall back to.
    let app = BristolCircuit::<F>::from_file(config.num_nodes, config.id, path)?;

    let wires = app.inputs_per_party();
    let app = if wires == 0 {
        log::info!("Circuit {} gives party {} no input wires", path, config.id);
        app
    } else {
        app.with_inputs(read_input_wires::<F>(config.id, wires))
    };

    velox::spawn(config, app, &EngineOptions::from_matches(matches)?)
}

/// This party's input wires, read as decimal integers.
///
/// An arithmetic circuit's wires carry field elements, where anonymous broadcast
/// reads ASCII payloads. A short or missing file is not fatal — the application
/// pads with random values, which exercises the circuit but computes nothing
/// meaningful.
fn read_input_wires<F: ProtocolField>(id: usize, count: usize) -> Vec<FieldElement<F>> {
    let primary = format!("testdata/inputs/circuit_input_{}.txt", id);
    let fallback = format!("circuit_input_{}.txt", id);
    velox::input::read_numeric_input_from_files::<F>(&primary, &fallback, count).unwrap_or_else(|e| {
        log::error!(
            "Error reading circuit input files: {}, falling back to random inputs",
            e
        );
        Vec::new()
    })
}
