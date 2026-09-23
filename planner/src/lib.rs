//! The Planner: the layer between an application and the MPC engine.
//!
//! Towards the engine, the Planner is an `Application` that speaks the
//! engine's three batch types — multiply, reveal, masked multiply. Towards
//! the application, it hosts a [`PlannerApplication`] and offers comparison,
//! min/max, truncation and fixed-point multiplication as single operations,
//! one [`OpDepthInput`] per op-depth, managing the edaBits those operations
//! consume.
//!
//! | op-depth contains | rounds |
//! |---|---|
//! | `Add` only | 0 |
//! | `Mul`, `Reveal`, `MaskReveal`, `Truncate`, `FixedMul` only | 1 |
//! | `Compare`, `ComparePub` | 2 + carry-tree levels (8 at ℓ = 61, 7 at 31) |
//! | `Max`, `Min`, `MaxPub`, `MinPub` | one more |
//!
//! # Layers, top to bottom
//!
//! - [`api::application`] — what an application programs against.
//! - [`api::engine`] — the engine's own API: the `Application` trait and
//!   its data model, which the engine hosts and the Planner implements.
//! - [`planner`] — the Planner itself: the engine's `Application`, running
//!   one op-depth at a time, one engine depth at a time.
//! - [`plan`] — the declared op-depths compiled into engine rounds.
//! - [`ops`] — the operations, one file each on one template, each owning
//!   its steps.
//! - [`primitives`] — edaBits and the carry tree.
//!
//! The Planner is generic over every protocol field. `Mul`, `Add` and
//! `Reveal` run over any of them; the comparison family, `Truncate`,
//! `FixedMul` and `MaskReveal` need a Mersenne prime field `p = 2^ℓ − 1`,
//! which the Planner reads at runtime through `ProtocolField::MERSENNE_BITS`
//! and refuses, by name, when the field is anything else.

pub mod api;
pub mod plan;
pub mod ops;
pub mod planner;
pub mod primitives;

pub use api::application::{OpParams, Op, OpDepthInput, OpResult, OpType, PlannerApplication, PlannerCounts};
pub use api::engine::{Application, DefaultApplication, DepthInput, PreprocessingCounts, RandomWireShares, RandomWires};
pub use plan::Plan;
pub use planner::Planner;
pub use primitives::edabit::EdaBit;
