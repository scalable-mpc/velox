# The Planner

The design — why the comparison opens nothing, the truncation formula, what
the engine gained — is in [`docs/comparison.md`](../docs/comparison.md); this
file is the API.

The Planner is the layer between an application and the Velox MPC engine. The
engine (`mpc`) speaks three batch types — *multiply*, *reveal*, *masked
multiply* — one batch per circuit depth. The Planner turns those into richer
vector operations an application can ask for in a single call: comparison,
min/max, ReLU, truncation and fixed-point multiplication, on top of plain
multiplication, addition and reveals. It also owns the preprocessing those
operations consume (edaBits) and lays it out so that every party draws the
same material for the same operation.

Everything in this crate is arithmetic on sharings the caller already holds;
nothing here touches the network. The engine hosts the Planner as one of its
`Application`s; the Planner hosts the application as a `PlannerApplication`.

```
    application            ─── implements ───►  planner::api::application::PlannerApplication
        │                                              (Op, OpDepthInput, OpResult, PlannerCounts)
        ▼
    planner::Planner       ─── implements ───►  planner::api::engine::Application
        │  (plan, ops, primitives)                     (DepthInput, PreprocessingCounts, RandomWires)
        ▼
    mpc::Context           (the engine: preprocessing, multiplication, reveal, verification, output)
```

The Planner is generic over every protocol field. `Mul`, `Add` and `Reveal`
run over any of them; the comparison family, `Truncate`, `FixedMul` and
`MaskReveal` need a Mersenne prime field `p = 2^ℓ − 1` (ℓ = 61 or 31, i.e.
`m61base` / `m31base`), because they depend on `bits(p − b) = ¬bits(b)` and
`msb(v) = lsb(2v)`. The Planner reads that at runtime through
`ProtocolField::MERSENNE_BITS` and refuses those ops by name over any other
field — when the plan is compiled, before preprocessing. Every application in
the workspace — `anonymous_broadcast`, `btx_setup` (over BLS12-381),
`reveal_probe`, `bristol_circuit`, `erc20` — is a `PlannerApplication`; none
implements the engine's `Application` itself.

Contents:

1. [Using the Planner from an application](#1-using-the-planner-from-an-application)
2. [The operations](#2-the-operations)
3. [Lifecycle of a run](#3-lifecycle-of-a-run)
4. [Struct and trait reference](#4-struct-and-trait-reference)
   - [`api::application`](#41-apiapplication--the-application-facing-api)
   - [`api::engine`](#42-apiengine--the-engine-facing-api)
   - [`planner`](#43-planner--the-planner-as-the-engines-application)
   - [`plan`](#44-plan--the-compiled-schedule)
   - [`ops`](#45-ops--the-operations)
   - [`primitives`](#46-primitives--edabits-and-the-carry-tree)
5. [Adding an operation](#5-adding-an-operation)
6. [Tests](#6-tests)
7. [Status](#7-status)

---

## 1. Using the Planner from an application

An application does four things.

### 1.1 Declare the circuit's shape: `PlannerCounts`

Before preprocessing, the Planner needs the *type* and *width* of the
operation each op-depth will run — its `OpParams` — plus any random wires the
application reads for itself and the number of output wires. From this the
Planner compiles a `Plan`, sizes the engine's preprocessing, and reserves its
edaBits. The declaration must be pure and identical at every party: it is read
once and every party lays out the same preprocessing slices from it.

```rust
use planner::{OpParams, OpType, PlannerCounts};

fn preprocessing_count(&self) -> PlannerCounts {
    PlannerCounts::new(
        vec![
            OpParams::new(OpType::Mul, 1024),        // op-depth 1
            OpParams::new(OpType::MaxPub, 1024),     // op-depth 2: ReLU
            OpParams::new(OpType::Truncate, 1024),   // op-depth 3
            OpParams::new(OpType::Reveal, 16),       // op-depth 4
        ],
        16, // output wires the engine reconstructs at the end
    )
    .with_random_wires(0, 0) // random bits / sharings the app consumes itself
}
```

The rules the declaration imposes on the run:

- **One operation per op-depth, of one type.** Op-depth `d` runs exactly one
  `Op`, whose `op_type()` must equal the declared type.
- **At most the declared width.** The op may cover fewer elements than
  declared (it then uses a prefix of the depth's edaBit slice, the same prefix
  everywhere) but never more. Declaring zero elements is rejected at compile
  time; running zero elements is allowed and its steps are skipped.
- **Op-depths run in order**, 1, 2, 3, … Scheduling depth `d + 2` after `d`,
  or scheduling while another op-depth is still in flight, is an error.

### 1.2 Implement `PlannerApplication`

The hooks have the same shape as the engine's `Application`, one level up:
each returns an `OpDepthInput` — the next op-depth to run, `Waiting`, or
`Done(outputs)` — and completed op-depths come back as an `OpResult` rather
than as raw products.

```rust
use anyhow::Result;
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;
use planner::{Op, OpDepthInput, OpResult, PlannerApplication, PlannerCounts, RandomWireShares};

#[async_trait]
impl<F: ProtocolField> PlannerApplication<F> for MyApp<F> {
    fn preprocessing_count(&self) -> PlannerCounts { /* as above */ }

    // Secrets this party deals to the circuit's input wires (default: none).
    async fn inputs(&mut self) -> Vec<FieldElement<F>> { self.my_inputs.clone() }

    // A party's input sharing terminated; its shares arrive here.
    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>)
        -> Result<OpDepthInput<F>>
    {
        self.inputs_of.insert(party, shares);
        Ok(OpDepthInput::Waiting) // until preprocessing is done and all inputs are in
    }

    // Preprocessing done. `wires` holds only what *this app* asked for in
    // `with_random_wires`; the Planner has already taken its own bits.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        let (x, y) = self.first_operands();
        OpDepthInput::op(1, Op::Mul { x, y })
    }

    // Op-depth `depth` finished; `result` is what the op produced.
    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        match depth {
            1 => {
                let products = result.shares()?;
                let zeros = vec![FieldElement::<F>::zero(); products.len()];
                OpDepthInput::op(2, Op::MaxPub { a: products, c: zeros })   // ReLU
            }
            2 => OpDepthInput::op(3, Op::Truncate { x: result.shares()?, d: 16 }),
            3 => {
                let t = result.shares()?;
                self.kept = t.clone();
                OpDepthInput::op(4, Op::Reveal { x: self.blinded(&t[..16]) })
            }
            4 => {
                let opened = result.public()?;
                log::info!("opened {} values", opened.len());
                Ok(OpDepthInput::Done(self.kept[..16].to_vec()))
            }
            _ => anyhow::bail!("no op-depth {depth}"),
        }
    }

    // Everything is verified and agreed on; `outputs` are the reconstructed
    // `Done` wires (default: ignore).
    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> { Ok(()) }
}
```

Errors returned from a hook abort the circuit with the message logged; they
never turn into a hang. `Waiting` means "nothing to schedule yet" — the engine
will call again when the next event arrives.

### 1.3 Host it on the engine

`Planner::new(app)` compiles the plan (it fails if the declaration is
malformed); the result is an engine `Application`, spawned like any other:

```rust
let planner = planner::Planner::new(MyApp::<F>::new(config.num_nodes, config.id))?;
velox::spawn(config, planner, &EngineOptions::from_matches(matches)?)
```

Field selection is the app's. A declaration that uses a Mersenne-only op
over another field makes `Planner::new` fail with a message naming the op;
`--field m61base` / `m31base` are the fields that run everything.

### 1.4 Read results: `OpResult`

| Variant | Produced by | Meaning |
|---|---|---|
| `Shares(Vec)` | `Mul`, `Add`, `Compare*`, `Max*`, `Min*`, `Truncate`, `FixedMul` | degree-`t` sharings, elementwise in operand order |
| `Public(Vec)` | `Reveal` | field elements every party holds identically |
| `Masked { public, mask }` | `MaskReveal` | the public `x_i + r_i` and, alongside, the `EdaBit` of each `r_i` (its sharing and its bit sharings) |

`result.shares()?` / `result.public()?` unwrap the expected variant or return
an error naming what arrived.

### 1.5 Contracts the application must honour

- **Domains.** `Compare`, `Max`, `Min` and their `Pub` variants need
  `|a|, |b| < 2^{ℓ−2}` as signed integers, so that `a − b` keeps its sign in
  the top bit. `Truncate` needs `|x| < 2^{ℓ−2}`; `FixedMul` needs
  `|x · y| < 2^{ℓ−2}`. Both truncating ops are exact up to an additive error in
  `{0, ±1, ±2}` (Liu et al. §3), and `d` must lie in `1 ..= ℓ − 3`.
- **Reveal privacy.** `Op::Reveal { x }` opens `x` as is. What a party learns
  in the engine's two-level reconstruction is the sharing polynomial of a
  public combination of the revealed values, which is harmless only when those
  polynomials are uniformly random — so `x` must already be blinded by a fresh
  random sharing. `MaskReveal` is the safe form: the Planner adds its own
  random `r` and hands back `r`'s edaBit so the application can undo it.
- **Public operands** (`c` in `ComparePub`, `MaxPub`, `MinPub`) are plain
  field elements every party holds in the clear; under Shamir the arithmetic on
  them is the same as on sharings, so the same op file serves both.
- **Random wires.** Random bits the application asks for arrive as sharings of
  `±1` (the engine's native form); random sharings are uniform field elements.
  The Planner's own edaBits are taken off the front of the bit stream before
  the application sees it.

## 2. The operations

Per element of the vector op. ℓ = 61 figures first, ℓ = 31 in brackets where
they differ. "Rounds" is engine depths, i.e. network round trips.

| `Op` | Result | Rounds | Engine steps | Mults / element | edaBits / element | Domain |
|---|---|---|---|---|---|---|
| `Mul { x, y }` | `Shares(x·y)` | 1 | multiply | 1 | 0 | — |
| `Add { x, y }` | `Shares(x+y)` | 0 | none (local) | 0 | 0 | — |
| `Reveal { x }` | `Public(x)` | 1 | reveal | 0 | 0 | `x` fresh-blinded |
| `MaskReveal { x }` | `Masked{x+r, eda(r)}` | 1 | reveal | 0 | 1 | — |
| `Truncate { x, d }` | `Shares(Trunc_d(x) ± 2)` | 1 | reveal of `x + 2^{ℓ−2} + r` | 0 | 1 | `|x| < 2^{ℓ−2}` |
| `FixedMul { x, y, d }` | `Shares(Trunc_d(x·y) ± 2)` | 1 | masked multiply, mask `r + 2^{ℓ−2}` | 1 (masked) | 1 | `|x·y| < 2^{ℓ−2}` |
| `Compare { a, b }` / `ComparePub { a, c }` | `Shares([a < b])` | 8 [7] | reveal, 6 [5] tree levels, 1 multiply | 86 [42] | 1 | `|a|,|b| < 2^{ℓ−2}` |
| `Max { a, b }` / `MaxPub { a, c }` | `Shares(max(a,b))` | 9 [8] | Compare's steps + 1 multiply | 87 [43] | 1 | same |
| `Min { a, b }` / `MinPub { a, c }` | `Shares(min(a,b))` | 9 [8] | Compare's steps + 1 multiply | 87 [43] | 1 | same |

`Relu(x)` is `MaxPub { a: x, c: 0 }`. One edaBit costs ℓ engine random bits.

The comparison family shares one pipeline, `DReLU(v) = [v ≥ 0]` on `v = a − b`
(Liu et al. Protocol 5.1 with the bitwise less-than replaced by
Catrina–de Hoogh's `CarryOutL` tree, which opens nothing):

1. reveal `y = 2v + r`, `r` an edaBit; over a Mersenne prime
   `msb(v) = lsb(2v) = y_0 ⊕ r_0 ⊕ [y < r]`;
2. the carry tree over the public bits of `y` and the sharings `¬[r]` gives
   `[c] = [y < r]` in `⌈log₂ ℓ⌉` multiply rounds of widths 30, 29, 15, 7, 3, 1
   at ℓ = 61 (15, 15, 7, 3, 1 at ℓ = 31);
3. one multiply `[b]·[c]` with `[b] = y_0 ⊕ [r_0]` (affine), then
   `DReLU = 1 − ([b] + [c] − 2[b][c])`.

`Compare = 1 − DReLU(a − b)`; `Max = DReLU·(a − b) + b`;
`Min = a − DReLU·(a − b)`.

The truncating ops share the local half of Liu et al.'s ΠTrunc: on the opened
`c = v + 2^{ℓ−2} + r`,

```
[Trunc_d(v)] = Trunc_d(c) − [Trunc_d(r)] + (1 − [r_msb]) · c_msb · (2^{ℓ−d} − 1) − 2^{ℓ−d−2}
```

with `[Trunc_d(r)]` and `[r_msb]` read off the edaBit's bits.

## 3. Lifecycle of a run

The engine drives; the Planner translates each hook in both directions.

```
engine                          Planner                                       application
──────                          ───────                                       ───────────
preprocessing_count()  ───────► plan.preprocessing_counts()  ◄─── compiled from app.preprocessing_count()
random_wires()         ───────► plan.random_wires()                  (= app's wires + ℓ bits per edaBit)
inputs()               ───────────────────────────────────────────► app.inputs()
input_sharing_termination(p,s) ────────────────────────────────────► app.input_sharing_termination(p,s)
                                start_scheduled_op_depth ◄─────────── returns OpDepthInput
on_preprocessing_complete(w) ─► take planner_bits() off w.bits, fill EdaBitPool
                                ──────────────────────────────────► app.on_preprocessing_complete(rest)
                                start_scheduled_op_depth ◄─────────── returns OpDepthInput
                                   Op{depth, op}: check, draw edaBits, build the Operation
                                   next_engine_depth_or_finish_op_depth
◄──── DepthInput::{Multiply|Reveal|MaskedMultiply}{engine depth, …}
on_depth_complete(d, results)  ─► results to the op's step;
on_reveal_complete(d, values)  ─►   next_engine_depth_or_finish_op_depth …
                                   no steps left: the op's OpResult
                                ──────────────────────────────────► app.on_depth_complete(op_depth, result)
                                start_scheduled_op_depth ◄─────────── returns the next Op / Waiting / Done
◄──── DepthInput::Done(outputs)
   [tuple and reveal verification, output reconstruction]
on_output(outputs)     ───────────────────────────────────────────► app.on_output(outputs)
```

Numbering: the application counts **op-depths** from 1; the Planner assigns
each op's steps consecutive **engine depths**, also from 1, across the whole
plan (an `Add` takes none). The engine keys its per-depth preprocessing slice
by engine depth, so the plan — computed identically at every party — is what
binds an op's step to the same masks everywhere.

## 4. Struct and trait reference

### 4.1 `api::application` — the application-facing API

**`trait PlannerApplication<F: ProtocolField>: Send + 'static`**

| Hook | Required | Returns | When |
|---|---|---|---|
| `fn preprocessing_count(&self) -> PlannerCounts` | yes | the declaration | once, before preprocessing; pure, same at every party |
| `async fn inputs(&mut self) -> Vec<FieldElement<F>>` | no (default empty) | this party's input secrets | start of input sharing |
| `async fn input_sharing_termination(&mut self, party, shares) -> Result<OpDepthInput<F>>` | yes | next move | party `party`'s input ACSS terminated |
| `async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>>` | yes | next move | preprocessing done; `wires` are the app's own random wires |
| `async fn on_depth_complete(&mut self, depth, result: OpResult<F>) -> Result<OpDepthInput<F>>` | yes | next move | op-depth `depth` finished |
| `async fn on_output(&mut self, outputs) -> Result<()>` | no (default `Ok`) | — | run verified and agreed; `outputs` are the reconstructed `Done` wires |

**`enum OpDepthInput<F>`** — what the application tells the Planner to do next.

| Variant | Meaning |
|---|---|
| `Waiting` | nothing to schedule yet |
| `Op { depth, op }` | run `op` as op-depth `depth` (from 1); build with `OpDepthInput::op(depth, op)`, which rejects depth 0 |
| `Done(Vec<FieldElement<F>>)` | the circuit is finished; these sharings are its outputs, to be verified and reconstructed |

`is_waiting()`; `Debug` prints shapes, never shares.

**`enum Op<F>`** — one vector operation, elementwise. Fields are `Vec<FieldElement<F>>`
(sharings, or public elements for `c`); `d: usize` is the number of fractional
bits dropped.

| Variant | Fields |
|---|---|
| `Mul` | `x, y` |
| `Add` | `x, y` |
| `Compare` | `a, b` |
| `ComparePub` | `a, c` |
| `Max` / `Min` | `a, b` |
| `MaxPub` / `MinPub` | `a, c` |
| `Reveal` | `x` |
| `MaskReveal` | `x` |
| `Truncate` | `x, d` |
| `FixedMul` | `x, y, d` |

Methods: `op_type() -> OpType`; `len()` / `is_empty()` (elements, from the
first operand); `params() -> OpParams` (type + width, what a declaration is
compared to); `validate(ell: Option<usize>) -> Result<()>` (operand vectors
agree in length; the field can run the op — `ell` is `F::MERSENNE_BITS`;
`d ∈ 1..=ℓ−3`). `Debug` prints `Type(n elements[, d=…])`.

**`enum OpType`** — `Mul, Add, Compare, ComparePub, Max, Min, MaxPub, MinPub, Reveal, MaskReveal, Truncate, FixedMul`.
`Copy + Eq + Hash`. `needs_mersenne()` is false for `Mul`, `Add`, `Reveal` and
true for the rest; `edabits_per_element()` is 0 for `Mul`, `Add`, `Reveal`
and 1 for everything else.

**`struct OpParams { op_type: OpType, elements: usize }`** — an op without its
operands; what an application declares for each op-depth before the operands
exist. `new(op_type, elements)`; `edabits()` = `elements ×
edabits_per_element`; `covers(&run)` is true when `run` has the same type and
at most this width.

**`struct PlannerCounts { ops: Vec<OpParams>, rand_bits: usize, sharings: usize, output: usize }`** —
the declaration. `new(ops, output)` (no app-level random wires);
`with_random_wires(rand_bits, sharings)`; `depth()` = number of op-depths.

**`enum OpResult<F>`** — `Shares(Vec)`, `Public(Vec)`, `Masked { public: Vec, mask: Vec<EdaBit<F>> }`
(see §1.4). `shares()` / `public()` unwrap or error; `Debug` prints counts.

### 4.2 `api::engine` — the engine-facing API

This is the engine's own API, hosted by `mpc::Context` and implemented by the
Planner (and by apps that bypass it). Documented here because the Planner is
its primary implementor; an application programming against the Planner never
touches it.

**`trait Application<F: ProtocolField>: Send + 'static`**

| Hook | Default | Purpose |
|---|---|---|
| `fn preprocessing_count(&self) -> PreprocessingCounts` | required | per-engine-depth gate profile; pure, read once |
| `fn random_wires(&self) -> RandomWires` | none | random bits / sharings the app reads as wires |
| `async fn inputs(&mut self) -> Vec<FieldElement<F>>` | empty | input secrets |
| `async fn input_sharing_termination(&mut self, party, shares) -> Result<DepthInput<F>>` | required | a party's input sharing arrived |
| `async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>>` | required | preprocessing done |
| `async fn on_depth_complete(&mut self, depth, results) -> Result<DepthInput<F>>` | required | a multiply depth's products |
| `async fn on_reveal_complete(&mut self, depth, values) -> Result<DepthInput<F>>` | `Err` | a reveal or masked-multiply depth's public values |
| `async fn on_output(&mut self, outputs) -> Result<()>` | `Ok` | verified, agreed output |

**`enum DepthInput<F>`** — what the application tells the engine to do next.
One batch per depth number, of one kind.

| Variant | Preprocessing charged | Result hook |
|---|---|---|
| `Waiting` | — | — |
| `Multiply { depth, x, y }` | one mask per gate + zero sharings per chunk (`gates_per_depth`) | `on_depth_complete` |
| `Reveal { depth, values }` | none (declares 0 gates); `([v], v)` recorded for verification | `on_reveal_complete` |
| `MaskedMultiply { depth, x, y, mask }` | zero sharings only, no mask (`masked_gates_per_depth`); tuple `(x, y, c − [mask])` verified | `on_reveal_complete` |
| `Done(outputs)` | — | `on_output`, after verification |

Constructors `multiply(depth, x, y)`, `reveal(depth, values)`,
`masked_multiply(depth, x, y, mask)` validate depth ≥ 1 and matching lengths;
`is_waiting()`.

**`struct PreprocessingCounts { gates_per_depth: Vec<usize>, masked_gates_per_depth: Vec<usize>, output: usize }`** —
the engine's per-depth profile, from which it reserves each depth a fixed
slice of the preprocessing pool. `new(gates_per_depth, output)`;
`with_masked_gates(v)`; `mult_gates()` (plain + masked: every one is a
verified tuple); `plain_gates()`; `depth()`; `gates_at_depth(d)` /
`masked_gates_at_depth(d)` (`None` past the declared depths).

**`struct RandomWires { bits: usize, sharings: usize }`** — random wires the
application reads as sharings; `new(bits, sharings)`.

**`struct RandomWireShares<F> { bits: Vec, sharings: Vec }`** — the delivery:
`bits` are sharings of `±1`, `sharings` uniform elements. `new`, `empty()`.

**`struct DefaultApplication<F>`** — a no-op application (10 depths × 1000
gates, 100 outputs, 10 000 random bits) for benchmarking the bare engine.

### 4.3 `planner` — the Planner as the engine's `Application`

One file: the struct, its two functions, and the engine hooks.

**`struct Planner<F, A: PlannerApplication<F>>`**

| Field | Role |
|---|---|
| `app: A` | the hosted application |
| `execution_plan: Plan` | the compiled schedule |
| `edabit_pool: EdaBitPool<F>` | edaBits, sliced per op-depth |
| `current_engine_run: Option<DepthRun<F>>` | the op-depth in flight, if any |
| `completed: usize` | highest op-depth completed |

**`struct DepthRun<F>`** (private) — `op_depth`, `op: Box<dyn Operation<F>>`,
`step` (the op's next step), `out: Option<(engine_depth, len)>` (the step
whose engine depth is out, and how many results it returns).

Public: `new(app) -> Result<Self>` (compiles the plan, logs its size);
`plan()`; `app()`. Private, each called from more than one place:

- `start_scheduled_op_depth(app_input)` — acts on what an application hook
  returned. `Waiting` and `Done` go straight to the engine. For an `Op` it
  rejects a concurrent or out-of-order op-depth, validates the op, checks
  the declaration `covers` it, draws the op-depth's edaBits, builds the
  `Operation`, and returns its first engine depth.
- `next_engine_depth_or_finish_op_depth()` — the running op's operands for
  its next step, checked against the batch type the plan expects and
  returned as a `DepthInput` at that step's engine depth (steps with no
  operands are passed over). When the op has no steps left, it hands the
  op's `OpResult` to the application's `on_depth_complete` and passes the
  answer to `start_scheduled_op_depth`.

`impl Application<F> for Planner<F, A>`: `preprocessing_count` /
`random_wires` read the plan; `inputs`, `input_sharing_termination` and
`on_output` pass through to the application; `on_preprocessing_complete`
takes `plan.planner_bits()` off the front of the delivered random bits,
fills the edaBit pool and passes the rest on; `on_depth_complete` checks the
engine depth and result count against the step that is out, gives the
results to the op and moves on; `on_reveal_complete` is the same hook.

### 4.4 `plan` — the compiled schedule

**`struct EngineRound { step: OpStep, engine_depth: usize, gates: usize }`** —
one engine depth: a step of an op, placed and sized to the declared width
(`gates = step.num_engine_operands × elements`).

**`struct OpDepthPlan { params: OpParams, rounds: Vec<EngineRound> }`** — one
op-depth; `rounds()` is its count.

**`struct Plan`** — every op-depth end to end, plus `ell`, `output`, and the
application's own `rand_bits` / `sharings`.

| Method | Returns |
|---|---|
| `compile(&PlannerCounts, ell: Option<usize>) -> Result<Plan>` | the plan; errors on an op-depth of zero elements, and on a Mersenne-only op when `ell` is `None` |
| `op_depth(d) -> Option<&OpDepthPlan>` | op-depth `d` (from 1) |
| `op_depths() -> &[OpDepthPlan]` | all |
| `ell()` | `Some(ℓ)` over a Mersenne prime field, `None` otherwise |
| `engine_depths()` | total engine depths |
| `edabits_per_depth() -> Vec<usize>` / `edabits_total()` | edaBit layout |
| `planner_bits()` | engine random bits the Planner keeps: `ℓ × edabits_total` |
| `preprocessing_counts() -> PreprocessingCounts` | the engine profile: plain gates at multiply rounds, masked at masked-multiply rounds, 0 at reveals |
| `random_wires() -> RandomWires` | `planner_bits + app bits`, app sharings |

### 4.5 `ops` — the operations

**`enum EngineOperationType`** — `Reveal`, `Multiply`, `MaskedMultiply`: the
three things the engine can run.

**`struct OpStep { engine_op: EngineOperationType, num_engine_operands: usize }`** —
one step of an op, engine-agnostic: which batch, and how many gates (or values
opened) *per element*. `const fn new`.

**`fn steps_of(op_type, ell) -> Vec<OpStep>`** — the steps of an op type;
what the plan and the op share.

**`enum EngineOperands<F>`** — what an op puts into one step:
`Reveal(Vec)`, `Multiply { x, y }`, `MaskedMultiply { x, y, mask }`.
`batch_type()`, `len()` (results the step returns), `is_empty()`,
`multiply(pairs)` builds a multiply batch from `(x, y)` pairs.

**`trait Operation<F>: Send`** — the template every op implements:

```rust
fn operands(&self, step: usize) -> EngineOperands<F>;                 // what goes into step `step`
fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()>; // what came back
fn into_result(self: Box<Self>) -> OpResult<F>;                       // the application's result
```

**`fn build(op, edabits) -> Result<Box<dyn Operation<F>>>`** — constructs the
op's state, drawing its edaBits (one per element where needed) from the
iterator; errors if it runs out. `Pub` variants map to the same struct as
their shared sibling.

State structs, one file each. Each file reads: header (semantics, rounds,
steps, preprocessing) → `steps()` → the struct → `impl Operation`.

| File | Struct | Fields | Steps |
|---|---|---|---|
| `mul.rs` | `Mul` | `x, y, out` | 0: multiply |
| `add.rs` | `Add` | `out` (summed at construction) | none |
| `reveal.rs` | `Reveal` | `x, out` | 0: reveal |
| `mask_reveal.rs` | `MaskReveal` | `x, eda, out` | 0: reveal `x + r` |
| `truncate.rs` | `Truncate` | `x, d, eda, out` | 0: reveal `x + 2^{ℓ−2} + r`; `unmask` on completion |
| `fixed_mul.rs` | `FixedMul` | `x, y, d, eda, out` | 0: masked multiply with mask `r + 2^{ℓ−2}`; `unmask` on completion |
| `compare.rs` | `Compare` | `drelu: DReLU` | DReLU's; result `1 − DReLU` |
| `max.rs` | `Max` | `diff, other, drelu, out` | DReLU's, then multiply `DReLU · diff`; `+ other` |
| `min.rs` | `Min` | `a, diff, drelu, out` | DReLU's, then multiply `DReLU · diff`; `a −` |

Shared pipelines (not `Operation`s themselves):

- `drelu.rs` — **`struct DReLU<F>`**: `tree: CarryTree`, `v`, `eda`,
  `slots: Vec<Vec<Slot<F>>>` (per element, the tree's current level),
  `b` (`y_0 ⊕ [r_0]`), `c` (`[y < r]`), `out` (`[v ≥ 0]`). `new(v, eda)`,
  `steps()` (= levels + 2), `result()`, and `operands` / `on_step_complete`
  with the template's shape so `Compare`/`Max`/`Min` delegate their leading
  steps. `fn steps(ell) -> Vec<OpStep>` gives reveal, one multiply per tree
  level of that level's width, one multiply.
- `fixed_point.rs` — the local half of ΠTrunc: `pow2(exp)`, `offset()`
  (`2^{ℓ−2}`), and `unmask(c, eda, d)` (steps 6–8 of Protocol 3.1, `Trunc_d`
  of the public `c` included).

`E<F>` is the crate's shorthand for `FieldElement<F>`; `no_such_step(op_type,
step)` is the error an op returns for a step outside its list.

### 4.6 `primitives` — edaBits and the carry tree

**`mod mersenne`** — `ell::<F>()`, `canonical::<F>(e)`,
`bit::<F>(e, i)`: the Mersenne arithmetic the ops need, read through
`ProtocolField::MERSENNE_BITS` / `mersenne_canonical`. They panic over a
non-Mersenne field, which cannot happen past `Plan::compile` and
`Op::validate`, the two places that refuse Mersenne-only ops cleanly.

**`struct EdaBit<F: ProtocolField> { value: FieldElement<F>, bits: Vec<FieldElement<F>> }`** —
a random `r` with sharings of its bits, LSB first. Assembled locally from ℓ
engine sign bits: `[b] = (1 + [s]) / 2`, `[r] = Σ 2^i [b_i]`.

| Method | |
|---|---|
| `from_signs(&[±1 sharings]) -> Result<Self>` | needs exactly ℓ; composes the value from the bits |
| `msb()` | `[r_{ℓ−1}]` |
| `trunc_shift(d)` | `[Trunc_d(r)]` from the bits (Theorem 3.1) |

**`struct EdaBitPool<F>`** — the circuit's edaBits, laid out per op-depth in
fixed slices so op-depth `d` draws the same material at every party.
`plan(&per_depth)`, `bits_needed()` (ℓ per edaBit), `fill(&signs)`,
`for_depth(depth, count)` (the first `count` of the depth's
slice; read, not drained).

**`struct CarryTree { k, levels: Vec<Vec<Node>> }`** — `CarryOutL`
specialised to a public first operand: the schedule of a `k`-bit less-than as
a balanced tree of the operator `(P_L, G_L) ∘ (P_H, G_H) = (P_L·P_H, G_H +
P_H·G_L)`, with a leaf pair costing one multiplication (`g_i = a_i(1 − p_i)`)
and the carry-in folded into leaf 0 so the whole left spine costs one each.

| Method | |
|---|---|
| `new(k)` | the schedule |
| `bits()`, `num_levels()`, `level_widths()` | 61 → `[30, 29, 15, 7, 3, 1]` (85); 31 → `[15, 15, 7, 3, 1]` (41) |
| `leaves(a_bits, not_b) -> Vec<Slot>` | level 0 from public `a` and sharings `¬[b]` |
| `level_operands(level, slots) -> Vec<(x, y)>` | the level's multiplications |
| `level_absorb(level, slots, products) -> Vec<Slot>` | the next level |
| `carry(slots)` | the carry out, once one slot is left; `[a < b] = 1 − carry` |

**`enum CombineType`** — `LeafPair` (1 mult), `Spine` (1 mult, `P = 0`
below), `Inner` (2 mults). **`enum Node`** — `Combine { lo, hi, combine }` or
`Carry(usize)` (an odd slot carried up). **`struct Slot<F> { p, g, leaf_a:
Option<u64> }`** — a `(P, G)` pair; `leaf_a` marks an affine leaf.

## 5. Adding an operation

1. Add the variant to `Op` and `OpType` in `api/application.rs`; extend
   `op_type`, `len`, `validate`, `Debug`, and `edabits_per_element`.
2. Write `ops/<name>.rs` on the template: header, `steps()` (or `steps(ell)`),
   the state struct, `impl Operation`. Every step must be one engine batch of
   one type, and `operands(step)` must return that type.
3. Register it: `pub mod` in `ops/mod.rs`, an arm in `steps_of`, an arm in
   `build` (drawing `n` edaBits if the op needs them).
4. Cover it in `tests/plaintext.rs` against an integer reference at ℓ = 61 and
   ℓ = 31, and in `plan.rs`'s `rounds_per_type`.

The plan and the preprocessing budget follow automatically from `steps()`.

## 6. Tests

- Unit tests live beside the code: `api/application.rs` (validation,
  `covers`), `plan.rs` (rounds per type, level-by-level layout, end-to-end
  placement), `primitives/edabit.rs` (bits and `Trunc_d` against integers, the
  pool's slicing), `primitives/carry_tree.rs` (exhaustive to 8 bits,
  randomised at 31 and 61).
- `tests/plaintext.rs` drives the whole Planner with a *plaintext engine*
  (sharings are linear, so a plaintext value is a valid degree-0 sharing:
  `Multiply` is the product, `Reveal` the identity, `MaskedMultiply` is `x·y +
  mask`). A scripted `PlannerApplication` runs each op and checks it against
  its integer reference: the comparison family exhaustively on small values and
  randomised at ℓ = 61 and 31, truncation and `FixedMul` within ±2, reveal /
  mask-reveal / mul / add exactly, a multi-depth circuit against its plan, and
  undeclared ops refused.

```
cargo test -p planner
```

End-to-end on the real engine: `apps/reveal_probe` exercises `Mul` and
`Reveal` over any field; `apps/bristol_circuit` over `m61base` or
`m31base` runs a `.arith` circuit on the Planner (`docs/CIRCUIT_FORMAT.md`,
`testdata/circuits/comparison.arith` uses every op).

## 7. Status

Every engine primitive the Planner uses — `Multiply`, `Reveal`,
`MaskedMultiply` — runs on the network, and the verification phase checks
multiplication tuples (plain and masked) and revealed values.
`apps/bristol_circuit` is the Planner-hosted application: a `.arith` circuit
with `LT`/`DRELU`/`RELU`/`MAX`/`MIN`/`TRUNC`/`FMUL` gates, one op-depth per
group of gates of one type (`circuit::OpGroup`).
