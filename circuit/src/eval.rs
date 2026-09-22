//! Cleartext evaluation of a parsed circuit.
//!
//! This is the reference the MPC evaluation is checked against: it runs the
//! same levels in the same order, on field elements rather than on sharings.
//! It is also a cheap way to sanity-check a `.arith` file before committing a
//! cluster run to it.
//!
//! It needs a Mersenne-prime field, because the comparison and truncation
//! gates are defined on the signed integers read out of one.
//!
//! `TRUNC` and `FMUL` are exact here; the protocol computes them to within
//! `±2` (Liu et al. §3), so a check against this reference allows that on
//! those wires.

use anyhow::{bail, Result};
use fields::{MersennePrimeField, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

use crate::{Circuit, GateType};

/// The signed integer a field element stands for: `x` for `x ≤ (p−1)/2`,
/// `x − p` above.
pub fn to_signed<F: MersennePrimeField>(e: &FieldElement<F>) -> i128 {
    let c = F::to_canonical_u64(e) as i128;
    let p = F::MODULUS as i128;
    if c <= (p - 1) / 2 { c } else { c - p }
}

/// The field element standing for a signed integer.
pub fn from_signed<F: ProtocolField>(v: i128) -> FieldElement<F> {
    if v >= 0 {
        FieldElement::<F>::from(v as u64)
    } else {
        -FieldElement::<F>::from((-v) as u64)
    }
}

/// `Trunc_d` on a signed integer: shift the magnitude, keep the sign.
pub fn trunc(v: i128, d: usize) -> i128 {
    if v >= 0 { v >> d } else { -((-v) >> d) }
}

/// Evaluates `circuit` on `inputs`, which are the input wires in wire order —
/// party by party, in the order of [`Circuit::inputs_per_party`].
///
/// Returns the output wire values in the order the file lists them.
pub fn evaluate_circuit<F: ProtocolField + MersennePrimeField>(
    circuit: &Circuit,
    inputs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>> {
    if inputs.len() != circuit.num_inputs() {
        bail!(
            "circuit takes {} input wires, {} values supplied",
            circuit.num_inputs(),
            inputs.len()
        );
    }

    let mut wires: Vec<Option<FieldElement<F>>> = vec![None; circuit.num_wires()];
    for (wire, value) in inputs.iter().enumerate() {
        wires[wire] = Some(value.clone());
    }

    let read = |wires: &Vec<Option<FieldElement<F>>>, wire: usize| -> Result<FieldElement<F>> {
        match wires[wire].as_ref() {
            Some(value) => Ok(value.clone()),
            // Unreachable on a parsed circuit — the parser rejects a gate that
            // reads an unwritten wire — but this function is also callable on a
            // hand-built `Circuit`.
            None => bail!("gate reads wire {} before it is written", wire),
        }
    };

    // Level order mirrors the protocol: the level's op groups first, in
    // order, then the linear gates that level unblocks.
    for level in circuit.levels().iter() {
        let grouped = level.op_groups.iter().flat_map(|group| group.gates.iter());
        for gate in grouped.chain(level.linear_gates.iter()) {
            let a = read(&wires, gate.left())?;
            let b = match gate.right() {
                Some(wire) => Some(read(&wires, wire)?),
                None => None,
            };
            let (x, y) = (to_signed::<F>(&a), b.as_ref().map(to_signed::<F>));
            let bit = |c: bool| from_signed::<F>(c as i128);
            let value = match gate.gate_type {
                GateType::Add => a + b.unwrap(),
                GateType::Sub => a - b.unwrap(),
                GateType::Mul => a * b.unwrap(),
                GateType::Lt => bit(x < y.unwrap()),
                GateType::DRelu => bit(x >= 0),
                GateType::Relu => from_signed::<F>(x.max(0)),
                GateType::Max => from_signed::<F>(x.max(y.unwrap())),
                GateType::Min => from_signed::<F>(x.min(y.unwrap())),
                GateType::Trunc(d) => from_signed::<F>(trunc(x, d)),
                GateType::FMul(d) => from_signed::<F>(trunc(x * y.unwrap(), d)),
            };
            wires[gate.output] = Some(value);
        }
    }

    circuit
        .output_wires()
        .iter()
        .map(|wire| read(&wires, *wire))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Depth, Gate};
    use fields::{Mersenne31Field, Mersenne61Field};

    type F = Mersenne61Field;

    fn elem(x: i128) -> FieldElement<F> {
        from_signed::<F>(x)
    }

    #[test]
    fn signed_view_round_trips() {
        for v in [0i128, 1, -1, 12345, -12345, (1i128 << 59), -(1i128 << 59)] {
            assert_eq!(to_signed::<F>(&elem(v)), v);
            assert_eq!(to_signed::<Mersenne31Field>(&from_signed::<Mersenne31Field>(v >> 30)), v >> 30);
        }
        assert_eq!(trunc(77, 4), 4);
        assert_eq!(trunc(-77, 4), -4);
        assert_eq!(trunc(-1, 4), 0);
    }

    /// Every op gate on a hand-built level, against integer arithmetic.
    #[test]
    fn op_gates_follow_integer_semantics() {
        let level = Depth::new(
            vec![
                Gate::binary(GateType::Lt, 0, 1, 2, 1),
                Gate::unary(GateType::DRelu, 0, 3, 1),
                Gate::unary(GateType::Relu, 0, 4, 1),
                Gate::binary(GateType::Max, 0, 1, 5, 1),
                Gate::binary(GateType::Min, 0, 1, 6, 1),
                Gate::unary(GateType::Trunc(3), 1, 7, 1),
                Gate::binary(GateType::FMul(3), 0, 1, 8, 1),
                Gate::mul(0, 1, 9, 1),
            ],
            vec![Gate::binary(GateType::Sub, 0, 1, 10, 1), Gate::add(9, 10, 11, 1)],
        );
        let circuit = Circuit::new(vec![Depth::empty(), level], 12, vec![1, 1], (2..12).collect());

        let out = evaluate_circuit::<F>(&circuit, &[elem(-9), elem(20)]).unwrap();
        let got: Vec<i128> = out.iter().map(to_signed::<F>).collect();
        //           lt drelu relu max min trunc(20,3) fmul(-180,3) mul  sub   add
        assert_eq!(got, vec![1, 0, 0, 20, -9, 2, -22, -180, -29, -209]);

        let out = evaluate_circuit::<F>(&circuit, &[elem(7), elem(-3)]).unwrap();
        let got: Vec<i128> = out.iter().map(to_signed::<F>).collect();
        assert_eq!(got, vec![0, 1, 7, 7, -3, 0, -2, -21, 10, -11]);
    }
}
