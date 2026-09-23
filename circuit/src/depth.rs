//! One level of a circuit.
//!
//! Ported from `cc_types/src/depth.rs`, restructured for the levelisation fix
//! and then for the Planner. The original grouped gates by a depth counted
//! over *all* gates, so a level could hold nothing but `ADD` gates and still
//! cost a protocol round. Here a level holds both kinds of gate and states
//! when each runs:
//!
//!  - [`Depth::op_groups`] are the gates that cost rounds, grouped by type in
//!    order of first appearance in the file. Each group is one vector
//!    operation — one Planner op-depth, the Planner running one type per
//!    op-depth. A level with more than one type runs its groups one after
//!    another.
//!  - [`Depth::linear_gates`] are evaluated locally *after* the level's last
//!    group returns (for level 0, as soon as the input wires are assembled).
//!    They are held in file order, which the parser has checked is
//!    topological, so evaluating them front to back never reads a wire that
//!    is not yet written.

use crate::{Gate, GateType};

/// The gates of one type at one level: one vector operation, and so one
/// Planner op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpGroup {
    pub gate_type: GateType,
    pub gates: Vec<Gate>,
}

impl OpGroup {
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}

/// The gates at one level.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Depth {
    /// Gates costing a round, one group per type, in order of first
    /// appearance.
    pub op_groups: Vec<OpGroup>,
    /// Linear gates unblocked by this level, in topological order.
    pub linear_gates: Vec<Gate>,
}

impl Depth {
    /// Creates a level from its multiplication gates and its linear gates.
    pub fn new(mult_gates: Vec<Gate>, linear_gates: Vec<Gate>) -> Self {
        let mut depth = Self { op_groups: Vec::new(), linear_gates: Vec::new() };
        for gate in mult_gates {
            depth.add_gate(gate);
        }
        depth.linear_gates = linear_gates;
        depth
    }

    /// Creates an empty level.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Files a gate: a linear gate onto the list, a round-costing gate into
    /// its type's group, opening the group if this is the type's first gate.
    pub fn add_gate(&mut self, gate: Gate) {
        if gate.is_linear() {
            self.linear_gates.push(gate);
            return;
        }
        match self.op_groups.iter_mut().find(|group| group.gate_type == gate.gate_type) {
            Some(group) => group.gates.push(gate),
            None => self.op_groups.push(OpGroup { gate_type: gate.gate_type, gates: vec![gate] }),
        }
    }

    /// Total number of gates at this level.
    pub fn num_gates(&self) -> usize {
        self.num_mult_gates() + self.linear_gates.len()
    }

    /// Number of round-costing gates at this level, of every type.
    pub fn num_mult_gates(&self) -> usize {
        self.op_groups.iter().map(|group| group.len()).sum()
    }

    /// Number of `MUL` gates at this level.
    pub fn num_mul_gates(&self) -> usize {
        self.op_groups.iter().filter(|group| group.gate_type == GateType::Mul).map(|group| group.len()).sum()
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

    /// Groups open in order of first appearance and collect every later
    /// gate of their type; a different `d` is a different type.
    #[test]
    fn op_groups_form_by_type_in_first_appearance_order() {
        let mut depth = Depth::empty();
        depth.add_gate(Gate::binary(GateType::Lt, 0, 1, 5, 1));
        depth.add_gate(Gate::mul(0, 1, 6, 1));
        depth.add_gate(Gate::unary(GateType::Trunc(4), 2, 7, 1));
        depth.add_gate(Gate::binary(GateType::Lt, 2, 3, 8, 1));
        depth.add_gate(Gate::unary(GateType::Trunc(8), 3, 9, 1));

        let types: Vec<GateType> = depth.op_groups.iter().map(|g| g.gate_type).collect();
        assert_eq!(types, vec![GateType::Lt, GateType::Mul, GateType::Trunc(4), GateType::Trunc(8)]);
        assert_eq!(depth.op_groups[0].len(), 2);
        assert_eq!(depth.num_mult_gates(), 5);
        assert_eq!(depth.num_mul_gates(), 1);
    }
}
