//! Protocol data-model types exposed by the [`Application`](crate::Application)
//! trait's hooks. Kept minimal: only the types the trait's `DepthInput` carries.

use crate::SmallField;

mod sharing;
pub use sharing::*;

mod mult_input;
pub use mult_input::*;

mod network_routing_input;
pub use network_routing_input::*;

/// A wire in the arithmetic circuit, identified by its index.
pub type Wire = usize;

/// What an application schedules for a given circuit depth: an optional
/// multiplication batch and/or an optional network-routing request. The MPC
/// engine consumes this after each application hook returns.
pub struct DepthInput {
    pub mult: Option<Multiplication<SmallField>>,
    pub network_routing: Option<NetworkRouting<SmallField>>,
}

impl DepthInput {
    /// Build a `DepthInput` carrying neither a multiplication nor a network
    /// routing request — useful when an Application hook has nothing to
    /// schedule for the current depth.
    pub fn empty() -> Self {
        Self {
            mult: None,
            network_routing: None,
        }
    }

    /// Build a `DepthInput` carrying a multiplication request; no network routing.
    pub fn from_mult(mult: Multiplication<SmallField>) -> Self {
        Self {
            mult: Some(mult),
            network_routing: None,
        }
    }

    /// Build a `DepthInput` carrying a network routing request; no multiplication.
    pub fn from_network_routing(network_routing: NetworkRouting<SmallField>) -> Self {
        Self {
            mult: None,
            network_routing: Some(network_routing),
        }
    }

    /// Returns the contained `NetworkRouting` request if one is present.
    pub fn network_routing(&self) -> Option<NetworkRouting<SmallField>> {
        self.network_routing.clone()
    }

    /// Returns the contained `Multiplication` request if one is present.
    pub fn mult(&self) -> Option<Multiplication<SmallField>> {
        self.mult.clone()
    }

    /// True iff neither a multiplication nor a network routing request is set.
    pub fn is_empty(&self) -> bool {
        self.mult.is_none() && self.network_routing.is_none()
    }
}
