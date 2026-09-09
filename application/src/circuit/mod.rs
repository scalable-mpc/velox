//! Bristol-style arithmetic circuits: the `.arith` file format, its typed
//! model, and a parser that levelises a circuit by multiplicative depth.
//!
//! The format is documented in `docs/CIRCUIT_FORMAT.md`. These types are the
//! Velox port of `cc_types/src/{circuit,depth,gate}.rs` and
//! `cc_types/src/utils.rs` from the `scalable_mpc` repository; each module's
//! header records what the port changed and why.
//!
//! Nothing here talks to the MPC engine. The application that evaluates a
//! parsed circuit against the [`Application`](crate::Application) trait is
//! [`BristolCircuit`](crate::BristolCircuit).

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
