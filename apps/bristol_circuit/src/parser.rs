//! Parser for `.arith` circuit files.
//!
//! Ported from `cc_types/src/utils.rs::parse_circuit_file`. The format is
//! unchanged (see `docs/CIRCUIT_FORMAT.md`); what changed is everything the
//! original left open:
//!
//!  1. **Levelisation counts multiplication gates only.** The original added a
//!     level for every gate, `ADD` included, which inflates the round count —
//!     `polynomial_eval.arith` came out at 4 depths, two of them holding
//!     nothing but additions, against a true multiplicative depth of 2.
//!  2. **Errors name the line they came from**, and are `anyhow::Error` like
//!     the rest of the workspace rather than a bespoke `ParseError`.
//!  3. **The structural invariants are checked** rather than assumed: wire
//!     indices stay inside the declared `num_wires`, every gate input is
//!     already defined when the gate is read (which is exactly the topological
//!     order the format requires), no wire is written twice, and every output
//!     wire is defined by the end.
//!
//! The gate set is `ADD`, `SUB`, `MUL` and the op gates `LT`, `DRELU`,
//! `RELU`, `MAX`, `MIN`, `TRUNC d`, `FMUL d`; `d` follows the type token.
//! `INNERP` is still rejected, as it was in the original — but with a
//! diagnostic saying so rather than a generic parse failure. It needs work on
//! the engine's verification pipeline, not just here; see
//! `docs/CIRCUIT_FORMAT.md`.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

use circuit::{Circuit, Gate, GateType, Wire};

/// Number of header lines before the gate definitions.
const HEADER_LINES: usize = 5;

/// Parses a `.arith` circuit file into a levelised [`Circuit`].
pub fn parse_circuit_file<P: AsRef<Path>>(path: P) -> Result<Circuit> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read circuit file {}", path.display()))?;
    parse_circuit(&text, &path.display().to_string())
}

/// Parses circuit text. `source` names the origin for error messages — a file
/// path from [`parse_circuit_file`], anything descriptive from a test.
pub fn parse_circuit(text: &str, source: &str) -> Result<Circuit> {
    // Blank lines and `#` comments carry no content but do occupy line numbers,
    // so the original line number rides along with each retained line. Without
    // it every diagnostic below would point at the wrong place in the file —
    // the fixtures open with five comment lines.
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .collect();

    if lines.len() < HEADER_LINES {
        bail!(
            "{}: circuit file has {} content lines, fewer than the {} header lines the format requires",
            source,
            lines.len(),
            HEADER_LINES
        );
    }

    let header = parse_usize_line(source, lines[0])?;
    if header.len() != 2 {
        bail!(
            "{}:{}: header line must be `<num_gates> <num_wires>`, got {} values",
            source,
            lines[0].0,
            header.len()
        );
    }
    let (expected_num_gates, num_wires) = (header[0], header[1]);

    let num_input_parties = parse_single_usize(source, lines[1], "number of input parties")?;
    let inputs_per_party = parse_usize_line(source, lines[2])?;
    if inputs_per_party.len() != num_input_parties {
        bail!(
            "{}:{}: header declares {} input parties but lists {} input counts",
            source,
            lines[2].0,
            num_input_parties,
            inputs_per_party.len()
        );
    }

    let num_inputs: usize = inputs_per_party.iter().sum();
    if num_inputs > num_wires {
        bail!(
            "{}:{}: input parties supply {} wires, more than the {} wires the header declares",
            source,
            lines[2].0,
            num_inputs,
            num_wires
        );
    }

    let num_outputs = parse_single_usize(source, lines[3], "number of outputs")?;
    let output_wires: Vec<Wire> = parse_usize_line(source, lines[4])?;
    if output_wires.len() != num_outputs {
        bail!(
            "{}:{}: header declares {} outputs but lists {} output wires",
            source,
            lines[4].0,
            num_outputs,
            output_wires.len()
        );
    }
    for wire in output_wires.iter() {
        if *wire >= num_wires {
            bail!(
                "{}:{}: output wire {} is outside the {} wires the header declares",
                source,
                lines[4].0,
                wire,
                num_wires
            );
        }
    }

    // A wire is "defined" once something writes it: an input party at the top of
    // the numbering, or a gate that has already been read. Checking membership
    // as each gate is read enforces the format's topological-order requirement
    // and, on the output side, the single-writer rule — the original silently
    // treated an undefined input as level 0 and let two gates share an output.
    let mut defined = vec![false; num_wires];
    for wire in 0..num_inputs {
        defined[wire] = true;
    }

    let mut gates: Vec<Gate> = Vec::with_capacity(expected_num_gates);
    for line in lines.iter().skip(HEADER_LINES) {
        let gate = parse_gate_line(source, *line, num_wires)?;
        for input in gate.inputs.iter() {
            if !defined[*input] {
                bail!(
                    "{}:{}: gate reads wire {} before anything writes it; gates must be listed in \
                     topological order and inputs must be party inputs or earlier gate outputs",
                    source,
                    line.0,
                    input
                );
            }
        }
        if defined[gate.output] {
            bail!(
                "{}:{}: wire {} is written more than once (or is a party input wire); each wire \
                 has exactly one writer",
                source,
                line.0,
                gate.output
            );
        }
        defined[gate.output] = true;
        gates.push(gate);
    }

    if gates.len() != expected_num_gates {
        bail!(
            "{}: header declares {} gates but the file defines {}",
            source,
            expected_num_gates,
            gates.len()
        );
    }

    for (index, wire) in output_wires.iter().enumerate() {
        if !defined[*wire] {
            bail!(
                "{}:{}: output {} names wire {}, which no gate or party writes",
                source,
                lines[4].0,
                index,
                wire
            );
        }
    }

    Ok(Circuit::from_gates(gates, num_wires, inputs_per_party, output_wires))
}

/// Parses a whitespace-separated line of non-negative integers.
fn parse_usize_line(source: &str, (line_no, line): (usize, &str)) -> Result<Vec<usize>> {
    line.split_whitespace()
        .map(|token| {
            token.parse::<usize>().map_err(|_| {
                anyhow::anyhow!("{}:{}: expected a non-negative integer, got {:?}", source, line_no, token)
            })
        })
        .collect()
}

/// Parses a line holding exactly one non-negative integer.
fn parse_single_usize(source: &str, line: (usize, &str), what: &str) -> Result<usize> {
    let values = parse_usize_line(source, line)?;
    if values.len() != 1 {
        bail!(
            "{}:{}: expected a single value for the {}, got {}",
            source,
            line.0,
            what,
            values.len()
        );
    }
    Ok(values[0])
}

/// Parses one gate definition:
/// `<num_inputs> <num_outputs> <input_wires...> <output_wire> <TYPE> [<d>]`.
///
/// The level is left at 0; [`Circuit::from_gates`] assigns the real one.
fn parse_gate_line(source: &str, (line_no, line): (usize, &str), num_wires: usize) -> Result<Gate> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        bail!(
            "{}:{}: gate line has {} fields, fewer than the 4 a gate needs \
             (`<num_inputs> <num_outputs> <input_wires...> <output_wire> <TYPE> [<d>]`)",
            source,
            line_no,
            parts.len()
        );
    }

    let num_inputs = parse_field(source, line_no, parts[0], "num_inputs")?;
    let num_outputs = parse_field(source, line_no, parts[1], "num_outputs")?;
    if num_outputs != 1 {
        bail!(
            "{}:{}: gate declares {} outputs; every gate in this format has exactly one",
            source,
            line_no,
            num_outputs
        );
    }

    // 2 header fields + num_inputs input wires + 1 output wire + 1 type token,
    // then the parameter a parameterised type takes.
    let type_index = num_inputs + 3;
    let Some(type_token) = parts.get(type_index) else {
        bail!(
            "{}:{}: gate declares {} inputs, so the type should be field {}, but the line holds only {}",
            source,
            line_no,
            num_inputs,
            type_index + 1,
            parts.len()
        );
    };
    let params = &parts[type_index + 1..];

    let param = |what: &str| -> Result<usize> {
        match params {
            [d] => {
                let d = parse_field(source, line_no, d, what)?;
                if d == 0 {
                    bail!("{}:{}: {} must be at least 1", source, line_no, what);
                }
                Ok(d)
            }
            _ => bail!(
                "{}:{}: {} takes one parameter after the type token, the number of bits to drop; got {}",
                source,
                line_no,
                type_token,
                params.len()
            ),
        }
    };
    let gate_type = match *type_token {
        "ADD" => GateType::Add,
        "SUB" => GateType::Sub,
        "MUL" => GateType::Mul,
        "LT" => GateType::Lt,
        "DRELU" => GateType::DRelu,
        "RELU" => GateType::Relu,
        "MAX" => GateType::Max,
        "MIN" => GateType::Min,
        "TRUNC" => GateType::Trunc(param("TRUNC's d")?),
        "FMUL" => GateType::FMul(param("FMUL's d")?),
        // Part of the format, but not yet runnable here: the engine's tuple
        // verification records one operand pair per gate, so an inner product's
        // output would be checked against its first term alone and an honest run
        // would be rejected. Supporting it is engine work, not parser work.
        "INNERP" => bail!(
            "{}:{}: INNERP gates are not supported yet; expand the inner product into MUL and ADD \
             gates (see docs/CIRCUIT_FORMAT.md)",
            source,
            line_no
        ),
        other => bail!(
            "{}:{}: unknown gate type {:?}; expected one of ADD, SUB, MUL, LT, DRELU, RELU, MAX, MIN, TRUNC, FMUL",
            source,
            line_no,
            other
        ),
    };
    if !params.is_empty() && !matches!(gate_type, GateType::Trunc(_) | GateType::FMul(_)) {
        bail!(
            "{}:{}: {} takes no parameter, but {} follow the type token",
            source,
            line_no,
            gate_type,
            params.len()
        );
    }

    if num_inputs != gate_type.arity() {
        bail!(
            "{}:{}: {} gate declares {} inputs; {} takes exactly {}",
            source,
            line_no,
            gate_type.name(),
            num_inputs,
            gate_type.name(),
            gate_type.arity()
        );
    }

    let mut wires = Vec::with_capacity(num_inputs + 1);
    for (offset, token) in parts[2..type_index].iter().enumerate() {
        let what = if offset < num_inputs { format!("input wire {}", offset) } else { "output wire".to_string() };
        let wire = parse_field(source, line_no, token, &what)?;
        if wire >= num_wires {
            bail!(
                "{}:{}: {} is wire {}, outside the {} wires the header declares",
                source,
                line_no,
                what,
                wire,
                num_wires
            );
        }
        wires.push(wire);
    }
    let output = wires.pop().expect("an output wire follows the inputs");

    Ok(Gate::new(gate_type, wires, output, 0))
}

/// Parses one numeric field of a gate line.
fn parse_field(source: &str, line_no: usize, token: &str, what: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("{}:{}: expected a non-negative integer for {}, got {:?}", source, line_no, what, token)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture circuits live at the workspace root, but unit tests run with the
    /// crate directory as the working directory.
    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/circuits")
            .join(name)
    }

    fn parse_fixture(name: &str) -> Result<Circuit> {
        parse_circuit_file(fixture(name))
    }

    #[test]
    fn simple_mul_has_one_multiplicative_level() {
        let circuit = parse_fixture("simple_mul.arith").unwrap();

        assert_eq!(circuit.num_gates(), 1);
        assert_eq!(circuit.num_mul_gates(), 1);
        assert_eq!(circuit.num_inputs(), 2);
        assert_eq!(circuit.num_outputs(), 1);
        assert_eq!(circuit.multiplicative_depth(), 1);
        assert!(circuit.level(0).unwrap().is_empty());
        assert_eq!(circuit.level(1).unwrap().num_mul_gates(), 1);
    }

    /// A circuit of only linear gates costs no multiplication round, so it has
    /// nothing but level 0. The original parser gave it a depth per ADD gate.
    #[test]
    fn simple_add_costs_no_round() {
        let circuit = parse_fixture("simple_add.arith").unwrap();

        assert_eq!(circuit.num_add_gates(), 1);
        assert_eq!(circuit.num_mul_gates(), 0);
        assert_eq!(circuit.multiplicative_depth(), 0);
        assert_eq!(circuit.level(0).unwrap().num_add_gates(), 1);
    }

    /// The regression this port exists for. `f(x) = a·x² + b·x + c` is
    /// `x²` and `b·x` at level 1, then `a·x²` at level 2, with the two trailing
    /// additions folded into level 2. Counting depth over every gate — as
    /// `calculate_wire_depths` did — reported 4 depths, of which two held only
    /// ADD gates and would each have scheduled an empty multiplication batch.
    #[test]
    fn polynomial_eval_is_two_deep_not_four() {
        let circuit = parse_fixture("polynomial_eval.arith").unwrap();

        assert_eq!(circuit.num_gates(), 5);
        assert_eq!(circuit.num_mul_gates(), 3);
        assert_eq!(circuit.num_add_gates(), 2);
        assert_eq!(circuit.multiplicative_depth(), 2, "multiplicative depth, not gate depth");
        assert_eq!(circuit.num_levels(), 3);

        // Level 1: x² (wire 4) and b·x (wire 6). Level 2: a·x² (wire 5) and
        // both additions.
        assert_eq!(circuit.level(0).unwrap().num_gates(), 0);
        assert_eq!(circuit.level(1).unwrap().num_mul_gates(), 2);
        assert_eq!(circuit.level(1).unwrap().num_add_gates(), 0);
        assert_eq!(circuit.level(2).unwrap().num_mul_gates(), 1);
        assert_eq!(circuit.level(2).unwrap().num_add_gates(), 2);

        let level_one_outputs: Vec<usize> = circuit
            .level(1)
            .unwrap()
            .op_groups[0]
            .gates
            .iter()
            .map(|gate| gate.output)
            .collect();
        assert_eq!(level_one_outputs, vec![4, 6]);
    }

    /// Two MULs feeding one ADD: the additions ride along with the level that
    /// unblocks them rather than taking a round of their own.
    #[test]
    fn linear_gates_ride_the_level_that_unblocks_them() {
        let circuit = parse_fixture("inner_product_2.arith").unwrap();

        assert_eq!(circuit.multiplicative_depth(), 1);
        assert_eq!(circuit.level(1).unwrap().num_mul_gates(), 2);
        assert_eq!(circuit.level(1).unwrap().num_add_gates(), 1);
    }

    /// `INNERP` is part of the format but not yet runnable: the engine's tuple
    /// verification records one operand pair per gate, so an inner product's
    /// output would be checked against its first term alone. The two fixtures
    /// that use it are kept, and must be rejected with a diagnostic that says
    /// what to do instead — not a generic parse failure.
    #[test]
    fn innerp_fixtures_are_rejected_with_a_pointed_message() {
        for name in ["inner_product_2_innerp.arith", "inner_product_4_innerp.arith"] {
            let error = parse_fixture(name)
                .expect_err(&format!("{} must not parse while INNERP is unsupported", name))
                .to_string();
            assert!(
                error.contains("INNERP gates are not supported yet")
                    && error.contains("expand the inner product into MUL and ADD"),
                "{}: got {:?}",
                name,
                error
            );
        }
    }

    /// The expanded spelling of the same inner product does parse, and costs one
    /// multiplication per term.
    #[test]
    fn expanded_inner_product_parses() {
        let expanded = parse_fixture("inner_product_2.arith").unwrap();
        assert_eq!(expanded.num_mul_gates(), 2);
        assert_eq!(expanded.multiplicative_depth(), 1);
    }

    #[test]
    fn chained_multiplications_take_a_level_each() {
        let circuit = parse_fixture("multiply_three.arith").unwrap();

        assert_eq!(circuit.multiplicative_depth(), 2);
        assert_eq!(circuit.inputs_per_party(), &[1, 1, 1]);
        assert_eq!(circuit.num_input_parties(), 3);
    }

    /// Every gate type parses, with `d` after the type token, and the op
    /// gates are levelised and grouped like multiplications.
    #[test]
    fn op_gates_parse_and_group() {
        let circuit = parse_fixture("comparison.arith").unwrap();

        assert_eq!(circuit.num_gates(), 11);
        assert_eq!((circuit.num_mult_gates(), circuit.num_mul_gates(), circuit.num_add_gates()), (9, 1, 2));
        assert_eq!(circuit.multiplicative_depth(), 2);
        let level_one: Vec<GateType> = circuit.level(1).unwrap().op_groups.iter().map(|g| g.gate_type).collect();
        assert_eq!(
            level_one,
            vec![GateType::Mul, GateType::Lt, GateType::Relu, GateType::Max, GateType::Min, GateType::FMul(4), GateType::DRelu]
        );
        assert_eq!(circuit.level(1).unwrap().op_groups[1].len(), 2, "both LT gates in one group");
        assert_eq!(circuit.level(2).unwrap().op_groups[0].gate_type, GateType::Trunc(4));
        assert_eq!(circuit.num_op_groups(), 8);
    }

    /// Outputs come back in file order, so the order has to survive parsing —
    /// the original stored them in a hash set.
    #[test]
    fn output_wire_order_is_preserved() {
        let text = "\
2 5
1
2
2
4 3

2 1 0 1 3 ADD
2 1 0 1 4 MUL
";
        let circuit = parse_circuit(text, "inline").unwrap();
        assert_eq!(circuit.output_wires(), &[4, 3]);
    }

    /// Every malformed fixture must be rejected, and the diagnostic must name
    /// the line the problem is on — the fixtures open with a comment line, so a
    /// parser counting only content lines would point one line short.
    #[test]
    fn malformed_fixtures_are_rejected_with_line_numbers() {
        let cases = [
            ("truncated_header.arith", "header lines"),
            ("bad_header.arith", ":2:"),
            ("party_count_mismatch.arith", ":4:"),
            ("output_count_mismatch.arith", ":6:"),
            ("gate_count_mismatch.arith", "declares 2 gates"),
            ("wire_out_of_range.arith", ":8:"),
            ("not_topological.arith", "topological order"),
            ("duplicate_wire_write.arith", "written more than once"),
            ("unknown_gate_type.arith", "unknown gate type"),
            ("binary_gate_arity.arith", "MUL takes exactly 2"),
            ("unary_gate_arity.arith", "RELU takes exactly 1"),
            ("trunc_without_d.arith", "takes one parameter"),
            ("trunc_zero_d.arith", "at least 1"),
            ("add_with_param.arith", "takes no parameter"),
            ("multi_output_gate.arith", "exactly one"),
            ("non_numeric_field.arith", "input wire 0"),
            ("undefined_output_wire.arith", "which no gate or party writes"),
        ];

        for (name, expected) in cases {
            let error = parse_fixture(&format!("invalid/{}", name))
                .expect_err(&format!("{} must not parse", name))
                .to_string();
            assert!(
                error.contains(expected),
                "{}: expected the error to mention {:?}, got {:?}",
                name,
                expected,
                error
            );
        }
    }

    /// A missing file is an error rather than a panic, and says which file.
    #[test]
    fn missing_file_is_reported() {
        let error = parse_circuit_file("testdata/circuits/does_not_exist.arith")
            .expect_err("a missing file must not parse")
            .to_string();
        assert!(error.contains("does_not_exist.arith"), "got {:?}", error);
    }
}
