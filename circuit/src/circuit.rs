//! A parsed Bristol-style arithmetic circuit.
//!
//! Ported from `cc_types/src/circuit.rs`. Two representational changes came
//! with the port:
//!
//!  - `depths` becomes [`levels`](Circuit::levels), indexed by *multiplicative*
//!    level rather than by a depth counted over every gate. Level 0 holds only
//!    linear gates — the ones evaluable straight off the input wires — so the
//!    number of protocol rounds is [`Circuit::multiplicative_depth`], one less
//!    than the number of levels.
//!  - `output_wires` becomes an ordered `Vec` rather than a hash set. The file
//!    lists the outputs in a specific order and reconstruction hands them back
//!    in that order, so a set would lose the mapping from position to wire.
//!
//! Input wires stay implicit: the format numbers them `0..total_inputs`,
//! dealer by dealer in the order of [`Circuit::inputs_per_party`].

use crate::{Depth, Gate, Wire};

/// An arithmetic circuit levelised by multiplicative depth.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Gates grouped by multiplicative level; index `r` holds level `r`.
    /// Always non-empty — a circuit with no gates still has level 0.
    levels: Vec<Depth>,
    /// Total number of wires the file declares.
    num_wires: usize,
    /// Number of input wires each input party supplies, in party order.
    inputs_per_party: Vec<usize>,
    /// Output wire indices, in the order the file lists them.
    output_wires: Vec<Wire>,
}

impl Circuit {
    /// Creates a circuit from already-levelised gates.
    pub fn new(
        levels: Vec<Depth>,
        num_wires: usize,
        inputs_per_party: Vec<usize>,
        output_wires: Vec<Wire>,
    ) -> Self {
        let levels = if levels.is_empty() {
            vec![Depth::empty()]
        } else {
            levels
        };
        Self {
            levels,
            num_wires,
            inputs_per_party,
            output_wires,
        }
    }

    /// Builds a circuit from gates in topological order, grouping them into
    /// multiplicative levels.
    ///
    /// A wire's level is the longest chain of multiplication gates on any path
    /// reaching it: input wires sit at level 0, a multiplication gate is one past
    /// the highest of its inputs, and a linear gate inherits the highest of its
    /// inputs without advancing. So a level's multiplication gates all become
    /// evaluable at the same round, and its linear gates are the ones that round
    /// unblocks. One pass suffices because the gates are in topological order,
    /// which is the caller's to guarantee — `bristol_circuit`'s parser checks it
    /// while reading the file.
    ///
    /// Levelisation lives here rather than with any one frontend: it is a
    /// property of the circuit, not of the syntax a circuit was written in, so a
    /// second frontend emitting these gates directly gets it for free.
    pub fn from_gates(
        gates: Vec<Gate>,
        num_wires: usize,
        inputs_per_party: Vec<usize>,
        output_wires: Vec<Wire>,
    ) -> Self {
        let mut wire_levels = vec![0usize; num_wires];
        let mut levelled: Vec<Gate> = Vec::with_capacity(gates.len());
        let mut multiplicative_depth = 0;

        for mut gate in gates.into_iter() {
            let inputs_level = wire_levels[gate.input_left].max(wire_levels[gate.input_right]);
            gate.level = if gate.is_multiplicative() {
                inputs_level + 1
            } else {
                inputs_level
            };
            wire_levels[gate.output] = gate.level;
            if gate.is_multiplicative() {
                multiplicative_depth = multiplicative_depth.max(gate.level);
            }
            levelled.push(gate);
        }

        // Levels 0..=multiplicative_depth. A linear gate can never exceed the
        // last multiplicative level: it inherits an input's level, and every wire
        // level is set by the gate writing it.
        let mut levels = vec![Depth::empty(); multiplicative_depth + 1];
        for gate in levelled.into_iter() {
            levels[gate.level].add_gate(gate);
        }

        Self::new(levels, num_wires, inputs_per_party, output_wires)
    }

    /// Creates an empty circuit.
    pub fn empty() -> Self {
        Self::new(Vec::new(), 0, Vec::new(), Vec::new())
    }

    /// The circuit's levels, index `r` being level `r`.
    pub fn levels(&self) -> &[Depth] {
        &self.levels
    }

    /// The gates at level `r`, or `None` if the circuit has no such level.
    pub fn level(&self, r: usize) -> Option<&Depth> {
        self.levels.get(r)
    }

    /// Number of levels, including level 0.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Number of multiplication rounds the circuit costs — the longest chain of
    /// multiplication gates in it. Zero for a purely linear circuit.
    pub fn multiplicative_depth(&self) -> usize {
        self.levels.len() - 1
    }

    /// Total number of gates.
    pub fn num_gates(&self) -> usize {
        self.levels.iter().map(|d| d.num_gates()).sum()
    }

    /// Total number of linear (addition) gates.
    pub fn num_add_gates(&self) -> usize {
        self.levels.iter().map(|d| d.num_add_gates()).sum()
    }

    /// Total number of gates costing a multiplication. This is what the
    /// multiplication preprocessing is sized against.
    pub fn num_mul_gates(&self) -> usize {
        self.levels.iter().map(|d| d.num_mul_gates()).sum()
    }

    /// Total number of wires the file declares.
    pub fn num_wires(&self) -> usize {
        self.num_wires
    }

    /// Number of input wires each input party supplies, in party order.
    pub fn inputs_per_party(&self) -> &[usize] {
        &self.inputs_per_party
    }

    /// Number of input wires party `party` supplies; zero for parties beyond
    /// the input parties the file names.
    pub fn inputs_of_party(&self, party: usize) -> usize {
        self.inputs_per_party.get(party).copied().unwrap_or(0)
    }

    /// Number of parties the file expects inputs from.
    pub fn num_input_parties(&self) -> usize {
        self.inputs_per_party.len()
    }

    /// Total number of input wires, across all parties.
    pub fn num_inputs(&self) -> usize {
        self.inputs_per_party.iter().sum()
    }

    /// Output wire indices, in file order.
    pub fn output_wires(&self) -> &[Wire] {
        &self.output_wires
    }

    /// Number of output wires.
    pub fn num_outputs(&self) -> usize {
        self.output_wires.len()
    }

    /// Returns true if `wire` is an input wire. Input wires are the first
    /// [`num_inputs`](Circuit::num_inputs) wire indices, by the format's rule.
    pub fn is_input_wire(&self, wire: Wire) -> bool {
        wire < self.num_inputs()
    }

    /// Returns true if `wire` is one of the circuit's outputs.
    pub fn is_output_wire(&self, wire: Wire) -> bool {
        self.output_wires.contains(&wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gate;

    /// `w2 = w0 · w1` with both inputs dealt by different parties.
    fn single_mul() -> Circuit {
        let levels = vec![
            Depth::empty(),
            Depth::new(vec![Gate::mul(0, 1, 2, 1)], Vec::new()),
        ];
        Circuit::new(levels, 3, vec![1, 1], vec![2])
    }

    #[test]
    fn counts_are_taken_over_every_level() {
        let circuit = single_mul();

        assert_eq!(circuit.num_levels(), 2);
        assert_eq!(circuit.multiplicative_depth(), 1);
        assert_eq!(circuit.num_gates(), 1);
        assert_eq!(circuit.num_mul_gates(), 1);
        assert_eq!(circuit.num_add_gates(), 0);
        assert_eq!(circuit.num_inputs(), 2);
        assert_eq!(circuit.num_outputs(), 1);
        assert_eq!(circuit.inputs_of_party(0), 1);
        assert_eq!(circuit.inputs_of_party(7), 0, "parties past the header deal nothing");
    }

    #[test]
    fn wire_roles() {
        let circuit = single_mul();

        assert!(circuit.is_input_wire(0) && circuit.is_input_wire(1));
        assert!(!circuit.is_input_wire(2));
        assert!(circuit.is_output_wire(2) && !circuit.is_output_wire(0));
    }

    /// A circuit of only linear gates costs no multiplication round at all.
    #[test]
    fn purely_linear_circuit_has_zero_multiplicative_depth() {
        let levels = vec![Depth::new(Vec::new(), vec![Gate::add(0, 1, 2, 0)])];
        let circuit = Circuit::new(levels, 3, vec![1, 1], vec![2]);

        assert_eq!(circuit.multiplicative_depth(), 0);
        assert_eq!(circuit.num_levels(), 1);
    }

    #[test]
    fn empty_circuit_still_has_level_zero() {
        let circuit = Circuit::empty();

        assert_eq!(circuit.num_levels(), 1);
        assert_eq!(circuit.multiplicative_depth(), 0);
        assert_eq!(circuit.num_gates(), 0);
    }
}
