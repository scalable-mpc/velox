//! The Planner: a bridge between an application and the MPC engine.
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
//! | mixed | the longest; `Mul`s ride in the first multiply step |
//!
//! # Layers, top to bottom
//!
//! - [`api::application`] — what an application programs against.
//! - [`api::engine`] — the engine's own API: the `Application` trait and
//!   its data model, which the engine hosts and the Planner implements.
//! - [`bridge`] — the Planner as an engine `Application`.
//! - [`planner`] — the state machine between the two: schedule an op-depth,
//!   send each step's batch, deliver results, complete.
//! - [`plan`] — the declared op-depths compiled into engine rounds.
//! - [`ops`] — the operations, one file each on one template, each owning
//!   its steps.
//! - [`primitives`] — edaBits and the carry tree.
//!
//! Everything is generic over a Mersenne-prime field
//! (`fields::MersennePrimeField`): the comparison and truncation tricks need
//! `p = 2^ℓ − 1`.

pub mod api;
pub mod bridge;
pub mod plan;
pub mod ops;
pub mod planner;
pub mod primitives;

pub use api::application::{OpParams, Op, OpDepthInput, OpResult, OpType, PlannerApplication, PlannerCounts};
pub use api::engine::{Application, DefaultApplication, DepthInput, PreprocessingCounts, RandomWireShares, RandomWires};
pub use plan::Plan;
pub use planner::Planner;
pub use primitives::edabit::EdaBit;
