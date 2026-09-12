//! Cleartext evaluation of a parsed circuit.
//!
//! This is the reference the MPC evaluation is checked against: it runs the
//! same levels in the same order, on field elements rather than on sharings.
//! It is also a cheap way to sanity-check a `.arith` file before committing a
//! cluster run to it.

use anyhow::{bail, Result};
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

use crate::{Circuit, GateType};

/// Evaluates `circuit` on `inputs`, which are the input wires in wire order —
/// party by party, in the order of [`Circuit::inputs_per_party`].
///
/// Returns the output wire values in the order the file lists them.
pub fn evaluate_circuit<F: ProtocolField>(
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

    // Level order mirrors the protocol: the level's multiplications first, then
    // the linear gates that level unblocks.
    for level in circuit.levels().iter() {
        for gate in level.mult_gates.iter().chain(level.linear_gates.iter()) {
            let left = read(&wires, gate.input_left)?;
            let right = read(&wires, gate.input_right)?;
            let value = match gate.gate_type {
                GateType::Add => left + right,
                GateType::Mul => left * right,
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
