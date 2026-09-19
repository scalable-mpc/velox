//! The Planner's two faces.
//!
//! - [`engine`]: the API the engine drives — the `Application` trait and the
//!   data its hooks exchange (`DepthInput`, `PreprocessingCounts`,
//!   `RandomWires`). An application may implement it directly; the Planner
//!   implements it on behalf of a `PlannerApplication` (see `crate::bridge`).
//! - [`application`]: the API an application programs against through the
//!   Planner — operations, op-depths, results, and the `PlannerApplication`
//!   hooks.
//!
//! Neither face holds arithmetic: the operations live in `ops`, the plan
//! in `plan`, and the state machine that connects the two in `planner`.

pub mod application;
pub mod engine;
