//! Arithmetic circuits: the typed model, levelisation by multiplicative depth,
//! and a cleartext evaluator.
//!
//! The gate set is `ADD`, `SUB`, `MUL` and the Planner's operations — `LT`,
//! `DRELU`, `RELU`, `MAX`, `MIN`, `TRUNC d`, `FMUL d` — whose semantics are
//! over signed integers read out of a Mersenne-prime field
//! (`fields::MersennePrimeField`): `(p−1)/2` and below is non-negative, above
//! it is `x − p`.
//!
//! This is the IR, not a file format. [`Circuit::from_gates`] takes gates in
//! topological order and groups them into the levels the protocol evaluates one
//! round at a time; [`evaluate_circuit`] runs a circuit in the clear as the
//! reference an MPC evaluation is checked against. A *frontend* -- something that
//! reads `.arith` files, or a compiler emitting gates directly -- lives with
//! whatever consumes it; the `.arith` reader is
//! `bristol_circuit::parse_circuit_file`, since that format and that application
//! go together.
//!
//! These types are the Velox port of `cc_types/src/{circuit,depth,gate}.rs` from
//! the `scalable_mpc` repository; each module's header records what the port
//! changed and why.
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
pub use depth::{Depth, OpGroup};

#[allow(clippy::module_inception)]
mod circuit;
pub use circuit::Circuit;

mod eval;
pub use eval::{evaluate_circuit, from_signed, to_signed, trunc};
