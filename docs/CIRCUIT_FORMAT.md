# Arithmetic Circuit Format

Velox evaluates arithmetic circuits given as text files with the `.arith`
extension, through the [`BristolCircuit`](../application/src/bristol_circuit/app.rs)
application. The format is inspired by
[Bristol Fashion](https://nigelsmart.github.io/MPC-Circuits/) but simplified for
arithmetic MPC over a finite field: gates are `ADD` and `MUL` rather than
`AND`/`XOR`/`INV`, and wires carry field elements rather than bits.

The format also specifies an `INNERP` gate. It is **not supported yet** — see
[INNERP](#innerp-not-supported-yet) — and the parser rejects it, pointing at the
expanded `MUL`/`ADD` spelling instead.

This document is the Velox port of `circuits/CIRCUIT_FORMAT.md` from the
`scalable_mpc` repository. The file syntax is unchanged and the circuits written
against it still parse; what changed is on the implementation side, and is
recorded under [What this port changed](#what-this-port-changed).

Run a circuit with the `bristol_circuit` binary — each application owns its own:

```
./target/release/bristol_circuit --config testdata/10/nodes-$i.json \
    --circuit testdata/circuits/polynomial_eval.arith --comp 2 ...
```

or through the test harness, which picks the binary from `CIRCUIT`:

```
CIRCUIT=testdata/circuits/polynomial_eval.arith bash scripts/test.sh 10 16 2
```

Anonymous broadcast is the separate `anonymous_broadcast` binary, whose
`--messages` flag is its anonymity set size.

## File Structure

```
<num_gates> <num_wires>
<num_input_parties>
<party_1_inputs> <party_2_inputs> ... <party_n_inputs>
<num_outputs>
<output_wire_1> <output_wire_2> ... <output_wire_m>

<gate_definitions>
```

Blank lines and lines beginning with `#` are ignored anywhere in the file.
Diagnostics quote the real line number in the file, comments included.

### Header

| Line | Content | Description |
|------|---------|-------------|
| 1 | `<num_gates> <num_wires>` | Total number of gates and wires in the circuit |
| 2 | `<num_input_parties>` | Number of parties providing inputs |
| 3 | `<n_1> <n_2> ... <n_p>` | Number of input wires from each party |
| 4 | `<num_outputs>` | Number of output wires |
| 5 | `<o_1> <o_2> ... <o_m>` | Wire indices of the outputs, **in output order** |

### Wire Numbering

Wires are numbered from 0:

- Wires `0` to `n_1 - 1`: party 0's inputs
- Wires `n_1` to `n_1 + n_2 - 1`: party 1's inputs
- ... and so on for each input party
- Remaining wires: intermediate results

Input assignment is **positional**: party `i` in the deployment supplies the
`i`-th block. A party the header does not name supplies no input wires and deals
nothing.

### Gate Definitions

```
<num_inputs> <num_outputs> <input_wires...> <output_wire> <gate_type>
```

| Field | Description |
|-------|-------------|
| `num_inputs` | Number of input wires (2 for `ADD`/`MUL`; `2n` for the unsupported `INNERP`) |
| `num_outputs` | Always 1 |
| `input_wires` | Space-separated input wire indices |
| `output_wire` | Output wire index |
| `gate_type` | `ADD` or `MUL` (`INNERP` is specified but unsupported) |

## Gate Types

### ADD

```
2 1 <wire_a> <wire_b> <wire_out> ADD
```

`wire_out = wire_a + wire_b`. Linear, so it is evaluated locally on the
sharings: it costs no round and no preprocessing.

### MUL

```
2 1 <wire_a> <wire_b> <wire_out> MUL
```

`wire_out = wire_a · wire_b`. Costs one multiplication.

### INNERP (not supported yet)

```
<2n> 1 <a_0> ... <a_{n-1}> <b_0> ... <b_{n-1}> <wire_out> INNERP
```

`wire_out = Σ a_i · b_i`, with the first `n` input wires the left vector and the
last `n` the right one. The gate exists so that matrix multiplication and
convolutions cost one multiplication per output element rather than one per
term, which is where the format's performance argument lives.

**The parser rejects it**, naming the line and suggesting the expanded
`MUL`/`ADD` spelling. It is not simply a parser gap:

- The *multiplication* protocol can already do it: `lin_mult`/`quad_mult` take a
  batch as `Vec<Vec<_>>`, one inner vector per gate, and an inner product is just
  a gate whose vectors are longer than one.
- The *verification* pipeline cannot. `lin_mult`/`quad_mult` record one operand
  pair per gate for tuple verification (`x[0]`, `y[0]`), so an inner-product
  gate's triple would be `(a_0, b_0, Σ a_j·b_j)`, which does not satisfy
  `z = x·y`. An honest run then fails the multiplication-constraint check.

Supporting it means widening `DepthInput::Multiply` to carry operand vectors
rather than scalars, recording all of a gate's operands for verification,
weighting a whole gate by one delinearization challenge power rather than one
pair, and sizing the verification budget in operand pairs rather than gates.
That is engine work and is deliberately out of scope here.

The two fixtures that use the gate — `inner_product_2_innerp.arith` and
`inner_product_4_innerp.arith` — are kept as the regression targets for when it
lands; `inner_product_2.arith` is the same computation spelled out, and runs
today.

## Structural Rules

The parser enforces all of these and names the offending line:

- Gates are listed in **topological order**: every input wire of a gate is
  either a party input wire or the output of an earlier gate.
- Every wire has **at most one writer**. A gate may not write a party input wire
  or a wire an earlier gate already wrote.
- Every wire index is below the header's `num_wires`, and every output wire is
  actually written by something.
- The gate count, the input-party count, and the output count all match what the
  header declares.
- `ADD`/`MUL` take exactly 2 inputs, and every gate has exactly one output.

## Depth Is Counted Over Multiplication Gates

A circuit's cost in protocol rounds is its **multiplicative depth**: the longest
chain of multiplication gates in it. Linear gates are free — they are applied to
the sharings locally — so they do not advance the depth.

Concretely, a wire's level is:

- `0` for a party input wire,
- `max(input levels) + 1` for the output of a `MUL` gate,
- `max(input levels)` for the output of an `ADD` gate.

`BristolCircuit` runs one multiplication batch per level, containing *every*
`MUL` gate at that level, and evaluates the linear gates a level unblocks in
between batches. So a level costs one round however wide it is, and
the batching is where the protocol's efficiency comes from.

Take `polynomial_eval.arith` below: `x²` and `b·x` are independent and share
level 1, `a·x²` is level 2, and both additions ride along with level 2. Two
rounds, not five, and not the four a depth counted over every gate would report.

## Example: Inner Product

`(a_0 · b_0) + (a_1 · b_1)`, spelled out
(`testdata/circuits/inner_product_2.arith`):

```
3 7
2
2 2
1
6

2 1 0 2 4 MUL
2 1 1 3 5 MUL
2 1 4 5 6 ADD
```

Two multiplications, multiplicative depth 1 — the `ADD` rides along with level 1.

## Example: Polynomial Evaluation

`f(x) = a·x² + b·x + c`, with party 0 holding the coefficients and party 1 the
evaluation point (`testdata/circuits/polynomial_eval.arith`):

```
5 9
2
3 1
1
8

2 1 3 3 4 MUL
2 1 0 4 5 MUL
2 1 1 3 6 MUL
2 1 5 6 7 ADD
2 1 7 2 8 ADD
```

- Party 0: wires 0, 1, 2 (`a`, `b`, `c`); party 1: wire 3 (`x`).
- Wire 4 = `x²`, wire 6 = `b·x` — **level 1**, one batch of two gates.
- Wire 5 = `a·x²`, wire 7 = `a·x² + b·x`, wire 8 = the output — **level 2**.

## Inputs

Each party reads its input wires from `testdata/inputs/circuit_input_<id>.txt`,
falling back to `circuit_input_<id>.txt`: one decimal integer per line, blank
lines and `#` comments skipped, in the wire order the header assigns that party.
Values are reduced modulo the field's order, and may be written wider than 64
bits. A short or missing file is not fatal — the remaining wires are filled with
random values, which still exercises the circuit but computes nothing meaningful.

Unlike anonymous broadcast, which takes any `k` of the sharings the honest
parties produce, a circuit binds wires to dealers positionally. So **every input
party's ACSS must terminate** before the circuit can start; there is no
substitute for a missing dealer's wire. That is inherent to input-carrying MPC
rather than a property of this format.

## Circuit Agreement

Every party must evaluate the same circuit, or their sharings mean different
things. The circuit file is distributed **out of band**, like the node config
files, and each party is pointed at it with `--circuit`. That is adequate for a
research prototype; a deployment wanting more would hash-agree on the file
during setup.

## Comparison with Bristol Format

| Feature | Bristol Fashion | This Format |
|---------|-----------------|-------------|
| Domain | Binary (bits) | Arithmetic (field elements) |
| Gates | AND, XOR, INV, EQ, EQW, MAND | ADD, MUL (INNERP specified, unsupported) |
| Use case | Garbled circuits, GMW | Secret sharing MPC |
| Constants | Via EQ gate | Not supported |

## What This Port Changed

The format specification and the parser came from `scalable_mpc`; the evaluator
did not exist there — `parse_circuit_file` had no callers. Porting closed these
gaps, each of which has a regression test:

1. **Depth is counted over multiplication gates.** The original
   `calculate_wire_depths` added a level for every gate, `ADD` included, so
   `polynomial_eval.arith` came out at 4 depths — two of them holding nothing
   but additions, each of which would have scheduled an empty multiplication
   batch and paid a round for it. Its true multiplicative depth is 2.
2. **Structural validation.** The original reported no line numbers, treated an
   undefined input wire as level 0 rather than rejecting it (so nothing checked
   the topological order the format requires), never used the header's
   `num_wires` to bound wire indices, and let two gates write the same wire.
   All of these are now errors that name their line.
3. **Output order is preserved.** Output wires were stored in a hash set, which
   loses the order the file lists them in; reconstruction hands outputs back in
   that order, so they are now an ordered list.

## Not Supported

- **`INNERP` gates**, for the verification-pipeline reason given
  [above](#innerp-not-supported-yet).
- **Constants.** `CONST`, `SMUL` and `SUB` were listed as future extensions in
  the original spec and remain unimplemented. `SUB` and public-constant affine
  gates are local and cheap to add; fixed-point constants are a larger question,
  since they imply truncation, which Velox does not have.
- **Comparison gates.** `LT`, `DReLU`, `ReLU` and `Max` need the comparison
  protocol tracked in issue #5. This format is where they will be expressed once
  the protocol exists; nothing here forecloses adding the gate types.
- **Boolean Bristol Fashion.** Classic Bristol files are boolean
  (`XOR`/`AND`/`INV`) and need bit sharings. Out of scope, but the parser
  dispatches on the gate-type token, so adding them is additive.
- **Packing.** Velox sharings are unpacked: one wire is one sharing. There is no
  SIMD packing of several wires into one sharing.

## References

- [Bristol Fashion MPC Circuits](https://nigelsmart.github.io/MPC-Circuits/)
- [MP-SPDZ Circuit Documentation](https://mp-spdz.readthedocs.io/en/latest/)
- [SCALE-MAMBA](https://github.com/KULeuven-COSIC/SCALE-MAMBA)
