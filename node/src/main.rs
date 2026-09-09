use anyhow::{anyhow, Result};
use clap::{load_yaml, App};
use config::Node;
use fnv::FnvHashMap;
use node::Syncer;
use signal_hook::{
    consts::{SIGINT, SIGTERM},
    iterator::Signals,
};
use std::{net::{SocketAddr, SocketAddrV4}};
use tokio::sync::oneshot;

/// Field the node runs the protocol over, chosen by `--field` at startup.
///
/// The engine, the application layer and the ACSS/Sh2t modules are all generic
/// over `F: fields::ProtocolField`, so each arm below is the *same* protocol
/// instantiated at a different field — adding a fourth is one `impl` plus one
/// line here, with nothing in the protocol touched.
fn spawn_mpc_over_field(
    field: &str,
    config: Node,
    mixing_batch_size: usize,
    compression_factor: usize,
    num_rand_batches: usize,
    node_normal: bool,
    circuit_path: Option<&str>,
) -> Result<oneshot::Sender<()>> {
    match field {
        "m61" | "mersenne61" => spawn_mpc::<fields::DefaultField>(
            config, mixing_batch_size, compression_factor, num_rand_batches, node_normal, circuit_path),
        "stark252" => spawn_mpc::<fields::Stark252Field>(
            config, mixing_batch_size, compression_factor, num_rand_batches, node_normal, circuit_path),
        "bn254" => spawn_mpc::<fields::BN254Field>(
            config, mixing_batch_size, compression_factor, num_rand_batches, node_normal, circuit_path),
        // Shares over the 61-bit base field, DZK proofs lifted into its
        // degree-4 extension. One share carries 7 bytes of text rather than 28,
        // so longer input lines fall back to random values.
        "m61base" => spawn_mpc::<fields::Mersenne61Field>(
            config, mixing_batch_size, compression_factor, num_rand_batches, node_normal, circuit_path),
        other => Err(anyhow!(
            "unknown field {:?}; expected one of m61, m61base, stark252, bn254", other)),
    }
}

/// Start the MPC protocol over field `F`, running whichever application the
/// command line selected.
///
/// `--circuit <path>` picks `BristolCircuit`, which evaluates the arithmetic
/// circuit in that `.arith` file; without it the node runs the anonymous
/// broadcast mixing network as before. Both are `Application` implementations
/// over the same engine, so the choice is one branch here and nothing else.
fn spawn_mpc<F: fields::ProtocolField>(
    config: Node,
    mixing_batch_size: usize,
    compression_factor: usize,
    num_rand_batches: usize,
    node_normal: bool,
    circuit_path: Option<&str>,
) -> Result<oneshot::Sender<()>> {
    match circuit_path {
        Some(path) => {
            let app = build_bristol_circuit::<F>(&config, path)?;
            spawn_with_app(config, app, compression_factor, num_rand_batches, node_normal)
        }
        None => {
            let app = build_anonymous_broadcast::<F>(&config, mixing_batch_size);
            spawn_with_app(config, app, compression_factor, num_rand_batches, node_normal)
        }
    }
}

/// Hand an application to the engine. Nothing in this body names a concrete
/// field or a concrete application.
fn spawn_with_app<F: fields::ProtocolField, A: application::Application<F>>(
    config: Node,
    app: A,
    compression_factor: usize,
    num_rand_batches: usize,
    node_normal: bool,
) -> Result<oneshot::Sender<()>> {
    mpc::Context::spawn(
        config,
        app,
        compression_factor,
        num_rand_batches,
        node_normal,
    ).or_else(|e| {
        log::error!("Error starting MPC protocol: {}", e);
        Err(e)
    })
}

/// Anonymous broadcast over a butterfly mixing network, with this party's
/// messages read as ASCII payloads through `F::encode_ascii`.
fn build_anonymous_broadcast<F: fields::ProtocolField>(
    config: &Node,
    mixing_batch_size: usize,
) -> application::AnonymousBroadcast<F> {
    let app = application::AnonymousBroadcast::<F>::new(
        config.num_nodes,
        config.num_faults,
        config.id,
        mixing_batch_size,
    );

    // This party's messages into the mixing network. A short or missing input
    // file is not fatal — the application pads with random values.
    let file_location_1 = format!("testdata/inputs/input_{}.txt", config.id);
    let file_location_2 = format!("input_{}.txt", config.id);
    let inputs = mpc::input::read_input_from_files::<F>(
        file_location_1.as_str(),
        file_location_2.as_str(),
        app.inputs_per_party(),
    ).unwrap_or_else(|e| {
        log::error!("Error reading input files: {}, falling back to random inputs", e);
        Vec::new()
    });
    app.with_inputs(inputs)
}

/// The arithmetic circuit in `circuit_path`, with this party's input wires read
/// as decimal integers.
///
/// A malformed circuit file is fatal — every party must evaluate the same
/// circuit, so there is nothing sensible to fall back to. A short or missing
/// *input* file is not: the application pads with random values, which still
/// exercises the circuit.
fn build_bristol_circuit<F: fields::ProtocolField>(
    config: &Node,
    circuit_path: &str,
) -> Result<application::BristolCircuit<F>> {
    let app = application::BristolCircuit::<F>::from_file(config.num_nodes, config.id, circuit_path)?;

    let num_inputs = app.inputs_per_party();
    if num_inputs == 0 {
        log::info!("Circuit {} gives party {} no input wires", circuit_path, config.id);
        return Ok(app);
    }

    let file_location_1 = format!("testdata/inputs/circuit_input_{}.txt", config.id);
    let file_location_2 = format!("circuit_input_{}.txt", config.id);
    let inputs = mpc::input::read_numeric_input_from_files::<F>(
        file_location_1.as_str(),
        file_location_2.as_str(),
        num_inputs,
    ).unwrap_or_else(|e| {
        log::error!("Error reading circuit input files: {}, falling back to random inputs", e);
        Vec::new()
    });
    Ok(app.with_inputs(inputs))
}

#[tokio::main]
async fn main() -> Result<()> {
    log::error!("{}", std::env::current_dir().unwrap().display());
    let yaml = load_yaml!("cli.yml");
    let m = App::from_yaml(yaml).get_matches();
    //println!("{:?}",m);
    let conf_str = m
        .value_of("config")
        .expect("unable to convert config file into a string");
    let vss_type = m
        .value_of("protocol")
        .expect("Unable to detect protocol to run");
    let syncer_file = m
        .value_of("syncer")
        .expect("Unable to parse syncer ip file");
    let mixing_batch_size = m
        .value_of("messages")
        .expect("Unable to parse number of batches")
        .parse::<usize>().unwrap();
    let compression_factor = m
        .value_of("comp")
        .expect("Unable to parse compression factor")
        .parse::<usize>().unwrap();
    // Optional; defaults to mpc::NUM_RAND_BATCHES when not supplied.
    let num_rand_batches = m
        .value_of("rand_batches")
        .map(|v| v.parse::<usize>().expect("Unable to parse number of random-sharing batches"))
        .unwrap_or(mpc::NUM_RAND_BATCHES);
    // let broadcast_msgs_file = m
    //     .value_of("bfile")
    //     .expect("Unable to parse broadcast messages file");
    // Optional; the Mersenne-61 Fp4 field the protocol has always used.
    let field_name = m.value_of("field").unwrap_or("m61");
    // Optional; a `.arith` arithmetic circuit to evaluate. Absent, the node runs
    // the anonymous broadcast mixing network.
    let circuit_path = m.value_of("circuit");
    let byz_flag = m.value_of("byz").expect("Unable to parse Byzantine flag");
    let node_normal: bool = match byz_flag {
        "true" => true,
        "false" => false,
        _ => {
            panic!("Byz flag invalid value");
        }
    };
    let conf_file = std::path::Path::new(conf_str);
    let str = String::from(conf_str);
    let mut config = match conf_file
        .extension()
        .expect("Unable to get file extension")
        .to_str()
        .expect("Failed to convert the extension into ascii string")
    {
        "json" => Node::from_json(str),
        "dat" => Node::from_bin(str),
        "toml" => Node::from_toml(str),
        "yaml" => Node::from_yaml(str),
        _ => panic!("Invalid config file extension"),
    };

    simple_logger::SimpleLogger::new()
        .with_utc_timestamps()
        .init()
        .unwrap();
    log::set_max_level(log::LevelFilter::Info);
    config.validate().expect("The decoded config is not valid");
    if let Some(f) = m.value_of("ip") {
        let f_str = f.to_string();
        log::info!("Logging the file f {}", f_str);
        config.update_config(util::io::file_to_ips(f.to_string()));
    }
    // let string_to_hex_string = |s: &str| -> String {
    //     let mut hex_string = String::new();
    //     for byte in s.as_bytes() {
    //         hex_string.push_str(&format!("{:02x}", byte));
    //     }
    //     hex_string
    // };
    // let largefield_ele = LargeField::from_hex(string_to_hex_string("ABCDEFGHIJKLMNOPQRST").as_str()).unwrap();
    
    // log::info!("Printing converted field element {:?}", largefield_ele);
    
    // let reverse_conversion = |fe: &LargeField| -> String {
    //     let bytes = fe.to_bytes_be();
    //     let s: String = bytes.iter().map(|&b| b as char).collect();
    //     s
    // };
    // log::info!("Printing reverse converted field element {:?}", reverse_conversion(&largefield_ele));
    let config = config;
    // Start the Reliable Broadcast protocol
    let exit_tx;
    match vss_type {
        // "acs" => {
        //     exit_tx = 
        //         acs::Context::spawn(config, 
        //             batches, 
        //             per_batch, 
        //             true,
        //             node_normal
        //         ).unwrap();
        // }
        "mpc" => {
            // The circuit lives in the application; the engine only drives the
            // protocol phases around it. `--field` picks which finite field
            // both of them run over, and `--circuit` picks which application:
            // a `.arith` file, or the built-in mixing network.
            exit_tx = spawn_mpc_over_field(
                field_name,
                config,
                mixing_batch_size,
                compression_factor,
                num_rand_batches,
                node_normal,
                circuit_path,
            )?;
        }
        // "sh2t" => {
        //     let (_req_sender,req_receiver) = channel(10000);
        //     let (out_sender,_out_receiver) = channel(10000);
        //     exit_tx =
        //         sh2t::Context::spawn(
        //             config, 
        //             req_receiver, 
        //             out_sender, 
        //             node_normal
        //         ).unwrap();
        // }
        "sync" => {
            let f_str = syncer_file.to_string();
            log::info!("Logging the file f {}", f_str);
            let ip_str = util::io::file_to_ips(f_str);
            let mut net_map = FnvHashMap::default();
            let mut idx = 0;
            for ip in ip_str {
                net_map.insert(idx, ip.clone());
                idx += 1;
            }
            //let client_addr = net_map.get(&(net_map.len()-1)).unwrap();
            //exit_tx = Syncer::spawn(net_map, config.client_addr.clone(),broadcast_msgs_file.to_string()).unwrap();
            exit_tx = Syncer::spawn(net_map, config.client_addr.clone()).unwrap();
        }
        _ => {
            log::error!(
                "Matching VSS not provided {}, canceling execution",
                vss_type
            );
            return Ok(());
        }
    }
    //let exit_tx = pedavss_cc::node::Context::spawn(config).unwrap();
    // Implement a waiting strategy
    let mut signals = Signals::new(&[SIGINT, SIGTERM])?;
    signals.forever().next();
    log::error!("Received termination signal");
    exit_tx
        .send(())
        .map_err(|_| anyhow!("Server already shut down"))?;
    log::error!("Shutting down server");
    Ok(())
}

pub fn to_socket_address(ip_str: &str, port: u16) -> SocketAddr {
    let addr = SocketAddrV4::new(ip_str.parse().unwrap(), port);
    addr.into()
}
