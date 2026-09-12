//! Bristol-style arithmetic circuits: the `.arith` file format, its typed
//! model, and a parser that levelises a circuit by multiplicative depth.
//!
//! The format is documented in `docs/CIRCUIT_FORMAT.md`. These types are the
//! Velox port of `cc_types/src/{circuit,depth,gate}.rs` and
//! `cc_types/src/utils.rs` from the `scalable_mpc` repository; each module's
//! header records what the port changed and why.
//!
//! Nothing here talks to the MPC engine: a circuit is a format and an IR, not a
//! protocol. The application that evaluates one on the engine is
//! `bristol_circuit::BristolCircuit`, in its own crate.

/// A wire in the arithmetic circuit, identified by its index.
///
/// Lives here rather than beside the engine's data model. The `Application`
/// trait never mentions wires -- it exchanges flat vectors of sharings -- so the
/// only users of this alias are this crate and the application that evaluates
/// circuits.
pub type Wire = usize;

mod gate;
pub use gate::{Gate, GateType};

mod depth;
pub use depth::Depth;

#[allow(clippy::module_inception)]
mod circuit;
pub use circuit::Circuit;

mod parser;
pub use parser::{parse_circuit, parse_circuit_file};

mod eval;
pub use eval::evaluate_circuit;
