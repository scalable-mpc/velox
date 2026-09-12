//! One multiplicative level of a circuit.
//!
//! Ported from `cc_types/src/depth.rs`, restructured for the levelisation fix.
//! The original grouped gates by a depth counted over *all* gates, so a level
//! could hold nothing but `ADD` gates and still cost a protocol round. Here a
//! level holds both kinds of gate and states when each runs:
//!
//!  - [`Depth::mult_gates`] are batched into this level's single multiplication
//!    request — one round, whatever the batch size.
//!  - [`Depth::linear_gates`] are evaluated locally *after* that round returns
//!    (for level 0, as soon as the input wires are assembled). They are held in
//!    file order, which the parser has checked is topological, so evaluating
//!    them front to back never reads a wire that is not yet written.

use crate::Gate;

/// The gates at one multiplicative level.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Depth {
    /// Gates costing a multiplication, all scheduled in one batch.
    pub mult_gates: Vec<Gate>,
    /// Linear gates unblocked by this level, in topological order.
    pub linear_gates: Vec<Gate>,
}

impl Depth {
    /// Creates a level from its two gate groups.
    pub fn new(mult_gates: Vec<Gate>, linear_gates: Vec<Gate>) -> Self {
        Self {
            mult_gates,
            linear_gates,
        }
    }

    /// Creates an empty level.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Files a gate into the group matching its type.
    pub fn add_gate(&mut self, gate: Gate) {
        if gate.is_multiplicative() {
            self.mult_gates.push(gate);
        } else {
            self.linear_gates.push(gate);
        }
    }

    /// Total number of gates at this level.
    pub fn num_gates(&self) -> usize {
        self.mult_gates.len() + self.linear_gates.len()
    }

    /// Number of gates this level's multiplication batch carries — the number
    /// of multiplications it costs.
    pub fn num_mul_gates(&self) -> usize {
        self.mult_gates.len()
    }

    /// Number of linear gates at this level.
    pub fn num_add_gates(&self) -> usize {
        self.linear_gates.len()
    }

    /// Returns true if this level has no gates at all.
    pub fn is_empty(&self) -> bool {
        self.num_gates() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_gate_files_by_type() {
        let mut depth = Depth::empty();
        assert!(depth.is_empty());

        depth.add_gate(Gate::mul(0, 1, 2, 1));
        depth.add_gate(Gate::add(2, 3, 4, 1));

        assert_eq!(depth.num_gates(), 2);
        assert_eq!(depth.num_mul_gates(), 1);
        assert_eq!(depth.num_add_gates(), 1);
        assert!(!depth.is_empty());
    }
}
