//! Run anonymous broadcast on the Velox engine.
//!
//! The application owns its binary: velox supplies argument parsing, config
//! loading, the tokio runtime, the syncer and signal handling, and this file
//! supplies the one thing that is specific to anonymous broadcast — building an
//! `AnonymousBroadcast<F>` and giving it this party's messages.

use anonymous_broadcast::AnonymousBroadcast;
use velox::{bail, ArgMatches, EngineOptions, ExitSender, FieldElement, Node, ProtocolField, Result};

fn main() -> Result<()> {
    // `--messages` is the anonymity set size, so it belongs to this application
    // rather than to the engine's shared argument list.
    let matches = velox::engine_args("anonymous-broadcast")
        .arg(velox::arg(
            "messages",
            "t",
            "size of the anonymity set; must be a power of two",
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

/// Start the mixing network over whichever field `--field` names.
///
/// Each arm is the *same* application at a different field — the engine, the
/// application layer and the ACSS/Sh2t modules are all generic over
/// `F: ProtocolField`. The arms are spelled out rather than hidden behind a
/// trait because `AnonymousBroadcast<F>` is a different type per field and
/// `--field` picks the field at runtime; collapsing that needs a generic
/// associated type, which is a lot of machinery for four lines.
fn start_over_field(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    match matches.value_of("field").unwrap_or("m61") {
        "m61" | "mersenne61" => start::<velox::fields::DefaultField>(config, matches),
        "stark252" => start::<velox::fields::Stark252Field>(config, matches),
        "bn254" => start::<velox::fields::BN254Field>(config, matches),
        // Shares over the 61-bit base field, DZK proofs lifted into its degree-4
        // extension. One share carries 7 bytes of text rather than 28, so longer
        // input lines fall back to random values.
        "m61base" => start::<velox::fields::Mersenne61Field>(config, matches),
        // The Mersenne-31 pair: the Fp8 tower (24 bytes of text per share) and
        // its 31-bit base field (3 bytes, so nearly every line falls back to a
        // random value — it exists for the comparison layer, not for text).
        "m31" | "mersenne31" => start::<velox::fields::Mersenne31Degree8ExtensionField>(config, matches),
        "m31base" => start::<velox::fields::Mersenne31Field>(config, matches),
        other => bail!(
            "unknown field {:?}; expected one of m61, m61base, m31, m31base, stark252, bn254",
            other
        ),
    }
}

fn start<F: ProtocolField>(config: Node, matches: &ArgMatches) -> Result<ExitSender> {
    let anonymity_set = velox::parse_usize(matches, "messages")?;
    if !anonymity_set.is_power_of_two() || anonymity_set < 2 {
        bail!(
            "--messages is the anonymity set size and must be a power of two of at least 2, got {}",
            anonymity_set
        );
    }

    let app = AnonymousBroadcast::<F>::new(
        config.num_nodes,
        config.num_faults,
        config.id,
        anonymity_set,
    );
    let inputs = read_messages::<F>(config.id, app.inputs_per_party());

    velox::spawn(config, app.with_inputs(inputs), &EngineOptions::from_matches(matches)?)
}

/// This party's messages into the mixing network, read as ASCII payloads through
/// `F::encode_ascii`.
///
/// A short or missing file is not fatal — the application pads with random
/// values, which still exercises the circuit but carries no message.
fn read_messages<F: ProtocolField>(id: usize, count: usize) -> Vec<FieldElement<F>> {
    let primary = format!("testdata/inputs/input_{}.txt", id);
    let fallback = format!("input_{}.txt", id);
    velox::input::read_input_from_files::<F>(&primary, &fallback, count).unwrap_or_else(|e| {
        log::error!("Error reading input files: {}, falling back to random inputs", e);
        Vec::new()
    })
}
