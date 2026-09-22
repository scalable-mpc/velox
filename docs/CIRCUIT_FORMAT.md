# Arithmetic Circuit Format

Velox evaluates arithmetic circuits given as text files with the `.arith`
extension, through the [`BristolCircuit`](../apps/bristol_circuit/src/lib.rs)
application. The format is inspired by
[Bristol Fashion](https://nigelsmart.github.io/MPC-Circuits/) but simplified for
arithmetic MPC over a finite field: wires carry field elements rather than
bits, and the gates are `ADD`, `SUB`, `MUL` and the *op gates* — `LT`,
`DRELU`, `RELU`, `MAX`, `MIN`, `TRUNC d`, `FMUL d` — which the Planner
(`planner/README.md`) runs as single operations over a Mersenne-prime field.

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
CIRCUIT=testdata/circuits/polynomial_eval.arith FIELD=m61base bash scripts/test.sh 10 16 2
CIRCUIT=testdata/circuits/comparison.arith FIELD=m61base bash scripts/test.sh 10 16 10
```

**Which field.** The application runs on the Planner, which needs a
Mersenne-prime field: `--field m61base` (the default) or `m31base`. The
engine's other fields are refused at startup with a message saying so.

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
| `num_inputs` | Number of input wires: 1 for `DRELU`/`RELU`/`TRUNC`, 2 for the rest (`2n` for the unsupported `INNERP`) |
| `num_outputs` | Always 1 |
| `input_wires` | Space-separated input wire indices |
| `output_wire` | Output wire index |
| `gate_type` | One of the types below (`INNERP` is specified but unsupported) |
| `d` | `TRUNC` and `FMUL` only: the number of low bits dropped, after the type token |

## Gate Types

### ADD

```
2 1 <wire_a> <wire_b> <wire_out> ADD
```

`wire_out = wire_a + wire_b`. Linear, so it is evaluated locally on the
sharings: it costs no round and no preprocessing.

### SUB

```
2 1 <wire_a> <wire_b> <wire_out> SUB
```

`wire_out = wire_a − wire_b`. Linear, local, free.

### MUL

```
2 1 <wire_a> <wire_b> <wire_out> MUL
```

`wire_out = wire_a · wire_b`. Costs one multiplication.

### The op gates

These read their operands as **signed integers**: a field element `x ≤ (p−1)/2`
is `x`, anything above is `x − p`. That needs `p = 2^ℓ − 1` (the Planner's
comparison and truncation tricks depend on it), which is why they run over
`m61base`/`m31base` only. Domains: every operand and every product must satisfy
`|v| < 2^{ℓ−2}` — `2^59` at ℓ = 61, `2^29` at ℓ = 31.

| Gate | Line | Semantics | Rounds (ℓ = 61 / 31) | Mults per gate | Random bits per gate |
|---|---|---|---|---|---|
| `LT` | `2 1 a b out LT` | `out = [a < b]`, 0 or 1 | 8 / 7 | 86 / 42 | ℓ |
| `DRELU` | `1 1 a out DRELU` | `out = [a ≥ 0]`, 0 or 1 | 8 / 7 | 86 / 42 | ℓ |
| `RELU` | `1 1 a out RELU` | `out = max(a, 0)` | 9 / 8 | 87 / 43 | ℓ |
| `MAX` | `2 1 a b out MAX` | `out = max(a, b)` | 9 / 8 | 87 / 43 | ℓ |
| `MIN` | `2 1 a b out MIN` | `out = min(a, b)` | 9 / 8 | 87 / 43 | ℓ |
| `TRUNC d` | `1 1 a out TRUNC d` | `out = Trunc_d(a)`, drop the low `d` bits keeping the sign | 1 | 0 | ℓ |
| `FMUL d` | `2 1 a b out FMUL d` | `out = Trunc_d(a · b)`, a fixed-point multiplication with `d` fractional bits | 1 | 1 (masked) | ℓ |

`TRUNC` and `FMUL` are exact up to an additive error in `{0, ±1, ±2}` (Liu et
al., USENIX Security 2024, §3); `d` must lie in `1 ..= ℓ − 3`. A check against
the cleartext reference (`circuit::evaluate_circuit`) allows that on those
wires. The other op gates are exact. Each op gate consumes one edaBit — ℓ of
the engine's random bits — which the Planner reserves from the circuit's
declaration.

`RELU` is `MAX` against the public constant 0 and `DRELU` is `1 − LT` against
it, so both cost what their binary form costs; a circuit does not need a
constant wire to express them.

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

## Depth Is Counted Over Round-Costing Gates

A circuit's level structure follows its **multiplicative depth**: the longest
chain of round-costing gates — `MUL` and the op gates — in it. Linear gates
(`ADD`, `SUB`) are free — they are applied to the sharings locally — so they do
not advance the depth.

Concretely, a wire's level is:

- `0` for a party input wire,
- `max(input levels) + 1` for the output of a `MUL` or op gate,
- `max(input levels)` for the output of an `ADD` or `SUB` gate.

Within a level, gates are grouped by type into **op groups**, in order of
first appearance in the file (a different `d` is a different type).
`BristolCircuit` runs each op group as one Planner op-depth — one vector
operation, one multiplication round for a `MUL` group — and evaluates the
linear gates a level unblocks once its last group is in. So a group costs its
op's rounds however wide it is, and the grouping is where the protocol's
efficiency comes from.

**A level with several types runs its op groups one after another.** The
Planner takes one operation type per op-depth, so a level holding one `MUL`
and one `LT` costs 1 + 8 rounds, not 8. Where rounds matter, keep a level to
one type: put independent `MUL`s and `LT`s at different levels, or accept the
serialisation. The round count of a circuit is the sum over its op groups of
the group's type's rounds.

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

## Example: The Op Gates

`testdata/circuits/comparison.arith` uses every op gate once, plus `SUB`, on
`a, b` from party 0 and `c, d` from party 1:

```
11 15
2
2 2
9
4 5 7 8 9 10 11 13 14

2 1 0 1 4 MUL
2 1 0 2 5 LT
2 1 1 3 6 SUB
1 1 6 7 RELU
2 1 0 2 8 MAX
2 1 1 3 9 MIN
1 1 4 10 TRUNC 4
2 1 2 3 11 FMUL 4
1 1 6 12 DRELU
2 1 5 12 13 ADD
2 1 1 3 14 LT
```

Level 0 is the `SUB`. Level 1 has seven op groups in file order — `MUL`, `LT`
(both `LT` gates), `RELU`, `MAX`, `MIN`, `FMUL 4`, `DRELU` — then the `ADD`;
level 2 is the `TRUNC 4` of the product. Eight op-depths, 46 engine rounds at
ℓ = 61 (41 at ℓ = 31). With `3` and `5` in `testdata/inputs/circuit_input_0.txt`
and `11` and `-7` in `circuit_input_1.txt` (input files are not tracked —
`.gitignore` excludes `*.txt` — so write them locally) the outputs are
`[15, 1, 12, 11, −7, 0 ± 2, −4 ± 2, 2, 0]`; the parties log them as signed
integers when the output is reconstructed.

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
- Wire 4 = `x²`, wire 6 = `b·x` — **level 1**, one op group of two gates.
- Wire 5 = `a·x²`, wire 7 = `a·x² + b·x`, wire 8 = the output — **level 2**.

## Inputs

Each party reads its input wires from `testdata/inputs/circuit_input_<id>.txt`,
falling back to `circuit_input_<id>.txt`: one decimal integer per line, a
leading `-` for a negative value, blank lines and `#` comments skipped, in the
wire order the header assigns that party. Values are reduced modulo the field's
order, and may be written wider than 64 bits. A short or missing file is not fatal — the remaining wires are filled with
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
| Gates | AND, XOR, INV, EQ, EQW, MAND | ADD, SUB, MUL, LT, DRELU, RELU, MAX, MIN, TRUNC, FMUL (INNERP specified, unsupported) |
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
4. **The op gates and `SUB`** (issue #5). Gates carry a list of inputs rather
   than a fixed pair, levels hold op groups by type, and the application moved
   from the engine's `Application` trait onto the Planner
   (`PlannerApplication`), which is now the only thing it talks to. The
   cleartext evaluator gained the signed-integer semantics
   (`circuit::evaluate_circuit`) and with them the Mersenne-prime bound.

## Not Supported

- **`INNERP` gates**, for the verification-pipeline reason given
  [above](#innerp-not-supported-yet).
- **Constants.** `CONST` and `SMUL` were listed as future extensions in the
  original spec and remain unimplemented; public-constant affine gates are
  local and cheap to add. (`SUB` landed with the op gates.)
- **Comparison against a secret-shared constant other than 0.** `RELU` and
  `DRELU` compare against the public 0; `LT`/`MAX`/`MIN` take two wires. A
  comparison against another public constant is `LT` against a wire holding
  it, which needs `CONST`.
- **Boolean Bristol Fashion.** Classic Bristol files are boolean
  (`XOR`/`AND`/`INV`) and need bit sharings. Out of scope, but the parser
  dispatches on the gate-type token, so adding them is additive.
- **Packing.** Velox sharings are unpacked: one wire is one sharing. There is no
  SIMD packing of several wires into one sharing.

## References

- [Bristol Fashion MPC Circuits](https://nigelsmart.github.io/MPC-Circuits/)
- [MP-SPDZ Circuit Documentation](https://mp-spdz.readthedocs.io/en/latest/)
- [SCALE-MAMBA](https://github.com/KULeuven-COSIC/SCALE-MAMBA)
