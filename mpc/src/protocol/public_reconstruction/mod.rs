//! Public reconstruction of a batch of sharings, in `O(1)` field elements per
//! value per party.
//!
//! Three things in the engine open sharings to everyone: the linear
//! multiplication protocol (masked products), random bit generation (the
//! squares of random sharings) and an application's `Reveal`. They differ
//! only in the degree of what they open, in whether the L1 step needs a
//! privacy term, and in what happens to the values afterwards. This module is
//! the shared exchange — the embedding, the two levels, the hash agreement —
//! parameterised by [`ReconConfig`] and dispatched on [`ReconKind`]; the
//! callers keep their preamble and their completion.
//!
//! - [`math`] is the arithmetic, as pure functions with an in-process test of
//!   the whole exchange over a local Shamir sharing.
//! - [`state`] is the per-depth state, attached to the depth's
//!   `SingleDepthState`, and the memory discipline every consumer inherits.
//! - `protocol` is the driver on `Context`.
//!
//! What this does not cover: the quadratic multiplication protocol (unused —
//! `multiplication_switch_threshold` is 0), and the small direct broadcasts
//! the verification phase, the common coin and the output layer use.

pub mod config;
pub mod math;
mod protocol;
pub mod state;

pub use config::{ReconConfig, ReconKind};
pub use state::ReconState;
