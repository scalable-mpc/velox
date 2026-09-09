//! Gates of a Bristol-style arithmetic circuit.
//!
//! Ported from `cc_types/src/gate.rs` in the `scalable_mpc` repository,
//! essentially unchanged: the gate set is `ADD` and `MUL`, both binary, and the
//! `depth` field becomes [`level`](Gate::level) because levels are now counted
//! over multiplication gates alone (see [`crate::circuit::parse_circuit_file`]).
//!
//! The format also specifies an `INNERP` gate, which this implementation does
//! not yet support — see `docs/CIRCUIT_FORMAT.md`. Adding it means widening this
//! struct's two fixed input wires to a list, and doing the engine-side work the
//! doc describes.

use crate::types::Wire;

/// Type of arithmetic gate operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    /// Addition gate: `output = input_left + input_right`. Local, free.
    Add,
    /// Multiplication gate: `output = input_left * input_right`. One multiplication.
    Mul,
}

impl GateType {
    /// True for gates the multiplication protocol has to run. Linear gates are
    /// evaluated locally on the sharings and cost neither a round nor
    /// preprocessing, which is why levels are counted over these gates alone.
    pub fn is_multiplicative(&self) -> bool {
        matches!(self, GateType::Mul)
    }
}

/// A gate in the arithmetic circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gate {
    /// The operation this gate performs.
    pub gate_type: GateType,
    /// Left input wire index.
    pub input_left: Wire,
    /// Right input wire index.
    pub input_right: Wire,
    /// Output wire index.
    pub output: Wire,
    /// Multiplicative level of this gate — the longest chain of multiplication
    /// gates on any path reaching it. Assigned by the parser, not by the file.
    pub level: usize,
}

impl Gate {
    /// Creates a gate. `level` is overwritten during levelisation.
    pub fn new(
        gate_type: GateType,
        input_left: Wire,
        input_right: Wire,
        output: Wire,
        level: usize,
    ) -> Self {
        Self {
            gate_type,
            input_left,
            input_right,
            output,
            level,
        }
    }

    /// Creates an addition gate.
    pub fn add(input_left: Wire, input_right: Wire, output: Wire, level: usize) -> Self {
        Self::new(GateType::Add, input_left, input_right, output, level)
    }

    /// Creates a multiplication gate.
    pub fn mul(input_left: Wire, input_right: Wire, output: Wire, level: usize) -> Self {
        Self::new(GateType::Mul, input_left, input_right, output, level)
    }

    /// This gate's input wires.
    pub fn inputs(&self) -> [Wire; 2] {
        [self.input_left, self.input_right]
    }

    /// Returns true if this is an addition gate.
    pub fn is_add(&self) -> bool {
        matches!(self.gate_type, GateType::Add)
    }

    /// Returns true if this is a multiplication gate.
    pub fn is_mul(&self) -> bool {
        matches!(self.gate_type, GateType::Mul)
    }

    /// Returns true if this gate costs a multiplication.
    pub fn is_multiplicative(&self) -> bool {
        self.gate_type.is_multiplicative()
    }

    /// Returns true if this gate is evaluated locally on the sharings.
    pub fn is_linear(&self) -> bool {
        !self.is_multiplicative()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_gate_accessors() {
        let gate = Gate::mul(0, 1, 2, 1);

        assert_eq!(gate.gate_type, GateType::Mul);
        assert_eq!(gate.inputs(), [0, 1]);
        assert_eq!(gate.output, 2);
        assert_eq!(gate.level, 1);
        assert!(gate.is_mul() && gate.is_multiplicative() && !gate.is_linear());
    }

    #[test]
    fn add_gate_is_linear() {
        let gate = Gate::add(0, 1, 2, 1);
        assert!(gate.is_add() && gate.is_linear() && !gate.is_multiplicative());
    }
}
