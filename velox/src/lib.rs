//! The Velox engine, as one dependency.
//!
//! An application owns its own `main.rs` and imports this crate to run itself on
//! the engine. Everything an application needs is re-exported here — the
//! [`Application`] trait, [`FieldElement`], the [`Node`] config type — so an
//! application takes a single dependency rather than five spread across two
//! repositories, each rev-pinned separately and free to skew. That matters most
//! once applications live in their own repositories: `application`, `fields` and
//! `mpc` come from velox, but `Node`, `Replica` and `util::io` come from
//! `secure-distributed-computing-protocols`, and an application should not have
//! to know that.
//!
//! # What a binary looks like
//!
//! ```ignore
//! fn main() -> velox::Result<()> {
//!     let matches = velox::engine_args("my-app")
//!         .arg(velox::arg("widgets", "w", "number of widgets", true))
//!         .get_matches();
//!
//!     velox::run_node(|| {
//!         velox::init_logging()?;
//!         let config = velox::load_config(&matches)?;
//!         match matches.value_of("protocol").unwrap_or("mpc") {
//!             "sync" => velox::spawn_syncer(&config, &matches),
//!             "mpc"  => match matches.value_of("field").unwrap_or("m61") {
//!                 "m61" => start::<velox::fields::DefaultField>(config, &matches),
//!                 // … one arm per field
//!                 other => velox::bail!("unknown field {:?}", other),
//!             },
//!             other => velox::bail!("unknown protocol {:?}", other),
//!         }
//!     })
//! }
//!
//! fn start<F: velox::ProtocolField>(config: Node, m: &velox::ArgMatches)
//!     -> velox::Result<velox::ExitSender>
//! {
//!     let app = MyApplication::<F>::new(config.num_nodes, config.id);
//!     velox::spawn(config, app, &velox::EngineOptions::from_matches(m)?)
//! }
//! ```
//!
//! The four-arm `match` on `--field` is deliberately left in the application
//! rather than hidden behind a trait here. Collapsing it needs a generic
//! associated type — the application type is `MyApplication<F>`, a different
//! type per field, and `--field` picks `F` at runtime — and that buys a dozen
//! lines of deduplication at the cost of a language feature every reader has to
//! know, worse compiler errors, and a trait that cannot be used as `dyn`.

use std::future::Future;

use anyhow::anyhow;
use fnv::FnvHashMap;
use signal_hook::{
    consts::{SIGINT, SIGTERM},
    iterator::Signals,
};

pub mod syncer;
pub use syncer::Syncer;

// ---------------------------------------------------------------------------
// Re-exports: one dependency for an application.
// ---------------------------------------------------------------------------

/// The trait an application implements, and the data model its hooks exchange.
pub use application::{Application, DepthInput, PreprocessingCounts, RandomWireShares, RandomWires};

/// The field abstraction the whole protocol is generic over, and the concrete
/// fields `--field` selects between (`fields::{DefaultField, Mersenne61Field,
/// Stark252Field, BN254Field, BLS12381ScalarField}`).
pub use fields::{self, ProtocolField};

/// The element type the trait's data model is built on.
pub use lambdaworks_math::field::element::FieldElement;

/// The engine, for applications that need more than the helpers below.
pub use mpc;

/// Reading a party's inputs from disk. The *file naming* is the application's
/// business — anonymous broadcast and an arithmetic circuit read different
/// files — but the parsing is not.
pub use mpc::input;

/// Deployment description: party count, fault threshold, this party's id, the
/// network map. Defined in `secure-distributed-computing-protocols`, and
/// re-exported so an application needs no dependency on that repository.
pub use config::Node;

pub use anyhow::{bail, Result};
pub use clap::{App, Arg, ArgMatches};

/// Handle that shuts the node down when sent on.
pub type ExitSender = tokio::sync::oneshot::Sender<()>;

// ---------------------------------------------------------------------------
// Arguments.
// ---------------------------------------------------------------------------

/// An argument in the shape the engine and its applications use: long flag named
/// after the argument, a short flag, help text, takes a value.
pub fn arg(name: &'static str, short: &'static str, help: &'static str, required: bool) -> Arg<'static, 'static> {
    Arg::with_name(name)
        .short(short)
        .long(name)
        .help(help)
        .takes_value(true)
        .required(required)
}

/// The arguments every node takes, whichever application it runs.
///
/// Application-specific flags are *not* here. `--messages` used to be, even
/// though it is anonymous broadcast's anonymity set size, so a node running an
/// arithmetic circuit still had to be handed a value it ignored — and
/// `main()` unwrapped it with `.expect()`, so omitting it was a panic rather
/// than an error. Applications add their own with `.arg(velox::arg(…))`.
pub fn engine_args(name: &'static str) -> App<'static, 'static> {
    App::new(name)
        .version("1.0")
        .about("A Velox MPC node")
        .arg(arg("config", "c", "config file with this node's startup information", true))
        .arg(arg("ip", "i", "file of IPs overriding the config's network map", false))
        .arg(arg("protocol", "p", "mpc (default), or sync to run the syncer", false))
        .arg(arg("syncer", "y", "IPs for the syncer to connect to", false))
        .arg(arg("comp", "o", "compression factor for multiplication gate verification", true))
        .arg(arg("rand_batches", "r", "sub-batches per random-sharing group", false).long("rand-batches"))
        .arg(arg("field", "f", "field to run over: m61 (default), m61base, stark252, bn254", false))
        .arg(arg("byz", "b", "Byzantine faulty or normal node", false).long("byzantine"))
}

/// Parse a required numeric argument.
pub fn parse_usize(matches: &ArgMatches, name: &str) -> Result<usize> {
    let value = matches
        .value_of(name)
        .ok_or_else(|| anyhow!("--{} is required", name))?;
    value
        .parse::<usize>()
        .map_err(|_| anyhow!("--{} must be a number, got {:?}", name, value))
}

// ---------------------------------------------------------------------------
// Start-up.
// ---------------------------------------------------------------------------

pub fn init_logging() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_utc_timestamps()
        .init()
        .map_err(|e| anyhow!("failed to initialise the logger: {}", e))?;
    log::set_max_level(log::LevelFilter::Info);
    Ok(())
}

/// Load the node config named by `--config`, applying the `--ip` override.
pub fn load_config(matches: &ArgMatches) -> Result<Node> {
    let path = matches
        .value_of("config")
        .ok_or_else(|| anyhow!("--config is required"))?;
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("config file {:?} has no extension", path))?;

    let owned = path.to_string();
    let mut config = match extension {
        "json" => Node::from_json(owned),
        "dat" => Node::from_bin(owned),
        "toml" => Node::from_toml(owned),
        "yaml" => Node::from_yaml(owned),
        other => bail!("unknown config extension {:?}; expected json, dat, toml or yaml", other),
    };

    config
        .validate()
        .map_err(|e| anyhow!("the decoded config is not valid: {}", e))?;
    if let Some(ip_file) = matches.value_of("ip") {
        log::info!("Overriding the config's network map from {}", ip_file);
        config.update_config(util::io::file_to_ips(ip_file.to_string()));
    }
    Ok(config)
}

/// The engine settings every application passes through untouched.
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    pub compression_factor: usize,
    pub num_rand_batches: usize,
    pub node_normal: bool,
}

impl EngineOptions {
    pub fn from_matches(matches: &ArgMatches) -> Result<Self> {
        Ok(Self {
            compression_factor: parse_usize(matches, "comp")?,
            num_rand_batches: match matches.value_of("rand_batches") {
                Some(value) => value
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--rand-batches must be a number, got {:?}", value))?,
                None => mpc::NUM_RAND_BATCHES,
            },
            node_normal: match matches.value_of("byz").unwrap_or("false") {
                "true" => true,
                "false" => false,
                other => bail!("--byzantine must be true or false, got {:?}", other),
            },
        })
    }
}

/// Hand an application to the engine.
///
/// Nothing in this body names a concrete field or a concrete application: the
/// engine, the application layer and the ACSS/Sh2t modules are all generic over
/// `F: ProtocolField`, so the caller's per-field match arms are the *same*
/// protocol instantiated four times.
pub fn spawn<F: ProtocolField, A: Application<F>>(
    config: Node,
    app: A,
    options: &EngineOptions,
) -> Result<ExitSender> {
    mpc::Context::spawn(
        config,
        app,
        options.compression_factor,
        options.num_rand_batches,
        options.node_normal,
    )
    .map_err(|e| {
        log::error!("Error starting MPC protocol: {}", e);
        e
    })
}

/// Start the syncer, which sequences a run rather than taking part in it.
///
/// It is not an application — it computes nothing and holds no shares — but it
/// ships in the application binaries because that is where the binaries are, and
/// a deployment that has to distribute one executable per host would rather not
/// distribute two.
pub fn spawn_syncer(config: &Node, matches: &ArgMatches) -> Result<ExitSender> {
    let syncer_file = matches
        .value_of("syncer")
        .ok_or_else(|| anyhow!("--protocol sync needs --syncer"))?;
    let mut net_map = FnvHashMap::default();
    for (index, ip) in util::io::file_to_ips(syncer_file.to_string()).into_iter().enumerate() {
        net_map.insert(index, ip);
    }
    Syncer::spawn(net_map, config.client_addr.clone())
        .map_err(|e| anyhow!("failed to start the syncer: {}", e))
}

// ---------------------------------------------------------------------------
// The runtime.
// ---------------------------------------------------------------------------

/// Build the tokio runtime, run `body` inside it, and block until a termination
/// signal arrives.
///
/// `Context::spawn` and `Syncer::spawn` are synchronous but call `tokio::spawn`
/// internally, so they have to be entered from a runtime context. Owning the
/// runtime here is what lets an application crate depend on velox alone — no
/// tokio, no `#[tokio::main]`, no `async fn main`.
pub fn run_node(body: impl FnOnce() -> Result<ExitSender>) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to build the tokio runtime: {}", e))?;
    let _guard = runtime.enter();

    let exit_tx = body()?;
    wait_for_shutdown(exit_tx)
}

/// Block until SIGINT or SIGTERM, then shut the node down.
pub fn wait_for_shutdown(exit_tx: ExitSender) -> Result<()> {
    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    signals.forever().next();
    log::error!("Received termination signal");
    exit_tx
        .send(())
        .map_err(|_| anyhow!("Server already shut down"))?;
    log::error!("Shutting down server");
    Ok(())
}

/// Escape hatch for an application that needs to await something itself.
pub fn block_on<T>(future: impl Future<Output = T>) -> Result<T> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to build the tokio runtime: {}", e))?;
    Ok(runtime.block_on(future))
}