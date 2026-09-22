//! Gates of a Bristol-style arithmetic circuit.
//!
//! Ported from `cc_types/src/gate.rs` in the `scalable_mpc` repository and
//! since widened: the gate set was `ADD` and `MUL`, both binary, and now also
//! holds `SUB` and the comparison, truncation and fixed-point gates the
//! Planner runs (issue #5). Inputs are a list rather than a fixed pair, since
//! `DRELU`, `RELU` and `TRUNC` take one wire; the `depth` field is
//! [`level`](Gate::level) because levels are counted over the gates that
//! cost a round (see `Circuit::from_gates`).
//!
//! The format also specifies an `INNERP` gate, which this implementation does
//! not yet support — see `docs/CIRCUIT_FORMAT.md`.

use crate::Wire;

/// Type of gate. `Trunc` and `FMul` carry `d`, the number of low bits
/// dropped; two `TRUNC` gates with different `d` are different types, which
/// is what puts them in different op groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    /// `out = a + b`. Local, free.
    Add,
    /// `out = a − b`. Local, free.
    Sub,
    /// `out = a · b`. One multiplication.
    Mul,
    /// `out = [a < b]`, as 0 or 1. Signed comparison.
    Lt,
    /// `out = [a ≥ 0]`, as 0 or 1.
    DRelu,
    /// `out = max(a, 0)`.
    Relu,
    /// `out = max(a, b)`.
    Max,
    /// `out = min(a, b)`.
    Min,
    /// `out = Trunc_d(a)`: drop the low `d` bits, keep the sign; error `±2`.
    Trunc(usize),
    /// `out = Trunc_d(a · b)`: a fixed-point multiplication with `d`
    /// fractional bits; error `±2`.
    FMul(usize),
}

impl GateType {
    /// True for the gates evaluated locally on the sharings, which cost
    /// neither a round nor preprocessing.
    pub fn is_linear(&self) -> bool {
        matches!(self, GateType::Add | GateType::Sub)
    }

    /// True for the gates the protocol has to run — everything that is not
    /// linear. Levels are counted over these gates alone.
    pub fn is_multiplicative(&self) -> bool {
        !self.is_linear()
    }

    /// Input wires the gate takes.
    pub fn arity(&self) -> usize {
        match self {
            GateType::DRelu | GateType::Relu | GateType::Trunc(_) => 1,
            _ => 2,
        }
    }

    /// The type's token in a `.arith` file, without its parameter.
    pub fn name(&self) -> &'static str {
        match self {
            GateType::Add => "ADD",
            GateType::Sub => "SUB",
            GateType::Mul => "MUL",
            GateType::Lt => "LT",
            GateType::DRelu => "DRELU",
            GateType::Relu => "RELU",
            GateType::Max => "MAX",
            GateType::Min => "MIN",
            GateType::Trunc(_) => "TRUNC",
            GateType::FMul(_) => "FMUL",
        }
    }
}

impl std::fmt::Display for GateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateType::Trunc(d) | GateType::FMul(d) => write!(f, "{} {}", self.name(), d),
            _ => f.write_str(self.name()),
        }
    }
}

/// A gate in the arithmetic circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gate {
    /// The operation this gate performs.
    pub gate_type: GateType,
    /// Input wire indices, as many as the type's arity.
    pub inputs: Vec<Wire>,
    /// Output wire index.
    pub output: Wire,
    /// Level of this gate — the longest chain of round-costing gates on any
    /// path reaching it. Assigned by the levelisation, not by the file.
    pub level: usize,
}

impl Gate {
    /// Creates a gate. `level` is overwritten during levelisation.
    pub fn new(gate_type: GateType, inputs: Vec<Wire>, output: Wire, level: usize) -> Self {
        debug_assert_eq!(inputs.len(), gate_type.arity(), "{} takes {} inputs", gate_type, gate_type.arity());
        Self { gate_type, inputs, output, level }
    }

    /// A binary gate.
    pub fn binary(gate_type: GateType, left: Wire, right: Wire, output: Wire, level: usize) -> Self {
        Self::new(gate_type, vec![left, right], output, level)
    }

    /// A unary gate.
    pub fn unary(gate_type: GateType, input: Wire, output: Wire, level: usize) -> Self {
        Self::new(gate_type, vec![input], output, level)
    }

    /// Creates an addition gate.
    pub fn add(left: Wire, right: Wire, output: Wire, level: usize) -> Self {
        Self::binary(GateType::Add, left, right, output, level)
    }

    /// Creates a multiplication gate.
    pub fn mul(left: Wire, right: Wire, output: Wire, level: usize) -> Self {
        Self::binary(GateType::Mul, left, right, output, level)
    }

    /// The first input wire.
    pub fn left(&self) -> Wire {
        self.inputs[0]
    }

    /// The second input wire; a unary gate has none.
    pub fn right(&self) -> Option<Wire> {
        self.inputs.get(1).copied()
    }

    /// Returns true if this is an addition gate.
    pub fn is_add(&self) -> bool {
        matches!(self.gate_type, GateType::Add)
    }

    /// Returns true if this is a multiplication gate.
    pub fn is_mul(&self) -> bool {
        matches!(self.gate_type, GateType::Mul)
    }

    /// Returns true if this gate costs a round.
    pub fn is_multiplicative(&self) -> bool {
        self.gate_type.is_multiplicative()
    }

    /// Returns true if this gate is evaluated locally on the sharings.
    pub fn is_linear(&self) -> bool {
        self.gate_type.is_linear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_gate_accessors() {
        let gate = Gate::mul(0, 1, 2, 1);

        assert_eq!(gate.gate_type, GateType::Mul);
        assert_eq!(gate.inputs, vec![0, 1]);
        assert_eq!((gate.left(), gate.right()), (0, Some(1)));
        assert_eq!(gate.output, 2);
        assert_eq!(gate.level, 1);
        assert!(gate.is_mul() && gate.is_multiplicative() && !gate.is_linear());
    }

    #[test]
    fn add_and_sub_are_linear() {
        assert!(Gate::add(0, 1, 2, 1).is_linear());
        assert!(Gate::binary(GateType::Sub, 0, 1, 2, 1).is_linear());
    }

    #[test]
    fn op_gates_cost_a_round() {
        let relu = Gate::unary(GateType::Relu, 0, 1, 1);
        assert_eq!((relu.left(), relu.right()), (0, None));
        for t in [GateType::Lt, GateType::DRelu, GateType::Relu, GateType::Max, GateType::Min, GateType::Trunc(4), GateType::FMul(4)] {
            assert!(t.is_multiplicative(), "{t}");
        }
        assert_eq!(GateType::Trunc(4).arity(), 1);
        assert_eq!(GateType::FMul(4).arity(), 2);
        assert_ne!(GateType::Trunc(4), GateType::Trunc(8), "different d, different op group");
        assert_eq!(GateType::FMul(16).to_string(), "FMUL 16");
        assert_eq!(GateType::Max.to_string(), "MAX");
    }
}
