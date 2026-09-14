# Comparison, truncation and fixed-point multiplication on Velox — plan

Tracks GitHub issue #5. Progress is kept in `docs/comparison-progress.md`.

## Sources

- Liu, Xie, Yu. *Scalable Multi-Party Computation Protocols for Machine
  Learning in the Honest-Majority Setting.* USENIX Security 2024. Basis for
  `ΠSolvedBits`, `ΠDReLU` (§5.1), `ΠTrunc` and `ΠFixed-Mult` (§3).
- Catrina, de Hoogh. *Improved Primitives for Secure Multiparty Integer
  Computation.* SCN 2010. Basis for `BitLTL` (Protocol 4.1) and `CarryOutL`
  (Table 3).

## Why not Liu et al.'s bitwise less-than as written

`ΠBitwise-LT` computes prefix-ORs with `ΠPreMult`, whose step 1 opens
`c_i = a_i · r_i'` via `ΠMultPub`. The `a_i` are *bits*: when `a_i = 0` the
opened value is 0 whatever `r_i'` is. In `ΠBitwise-LT` those bits are
`¬(y_i ⊕ r_i)`, so the opening reveals which bits of the mask `r` equal the
public bits of `y`, hence `r`, hence the secret `a = (y − r)/2`.

## Replacement: BitLTL / CarryOutL

`[a < b] = 1 − CarryOut(a, ¬b, c_in = 1)` — the carry out of the top bit of
`2^k + a − b`. Both operands are `< 2^ℓ`, so it slots straight into
`ΠDReLU` step 3 as `[c] ← BitLTL(y, [r]_B)`. The Mersenne complement trick
(`bits(p − b) = ¬bits(b)`) is the same one BitLTL uses for `2^k − 1 − b`.

The carry out is a k-ary fold of the associative carry-lookahead operator

    (P_L, G_L) ∘ (P_H, G_H) = (P_L · P_H,  G_H + P_H · G_L)     L = lower block, H = upper block

with leaves `p_i = a_i ⊕ b_i`, `g_i = a_i ∧ b_i`, evaluated as a balanced
binary tree: `⌈log₂ k⌉` rounds, `k − 1` combines. Nothing is ever opened;
every product is an ordinary secure multiplication, so the privacy argument is
the multiplication protocol's and the malicious guarantee comes from Velox's
tuple verification for free.

Two specialisations because `a` (the revealed `y`) is public:

- **Leaf pairs cost one multiplication.** `g_i = a_i · (1 − p_i)`, so for a
  pair of leaves `P = p_L p_H` (1 mult) and
  `G = a_H (1 − p_H) + a_L (p_H − P)` is local.
- **Carry-in folded into the spine.** Treat `c_in` as bit position 0 with
  `(P, G) = (0, 1)`; combining with leaf 1 is local and leaves `P = 0`. Every
  node containing position 0 (the left spine, root included) therefore has
  `P = 0` for free and costs the single multiplication `P_H · G_L`. The final
  `G + P · c_in` step disappears.

Cost at `ℓ = 61`: 6 depths, level widths **30, 29, 15, 7, 3, 1** = **85**
multiplications per comparison (CdH's generic bound is `2k − 2 = 120`; the
earlier Sklansky prefix-OR plan was ~180 at the same depth). At `ℓ = 31`:
5 depths, 41 multiplications.

## Operations, in Velox terms

Everything reduces to two engine primitives — a batched public **Reveal** of
degree-t sharings (new) and **Multiply** (existing) — plus **solved bits**:
ℓ random 0/1 bits assembled locally into `[r] = Σ 2^i [r_i]` from the ±1 signs
`rand_bit` already produces (`b = (1 + s)/2`). The `r = 2^ℓ − 1 ≡ 0` event has
probability `2^{-ℓ}` and is ignored, as in the paper.

| Op | Protocol | Rounds | Comm / op (per party, ℓ=61) | Preprocessing / op |
|---|---|---|---|---|
| `DReLU(a)` | reveal `y = 2a + r`; `b = y_0 ⊕ [r_0]`; `c = BitLTL(y, [r]_B)`; out `1 − (b ⊕ c)` | 1 + 6 + 1 = 8 | ~86 mults ≈ 260 elems + 1 reveal | 61 bits, ~86 masks, ~43 zero-sharings |
| `Trunc_d(a)` | ΠTrunc: reveal `c = a + 2^{ℓ−2} + r`, rest local; error `±2` | 1 | ~3 elems | 61 bits |
| `FixedMul_d(a,b)` v1 | `Mul` then `Trunc` | 2 | mult + reveal | 61 bits + mask |
| `FixedMul_d(a,b)` v2 | ΠFixed-Mult via a `MaskedMultiply` engine primitive | 1 | as a mult | follow-up |
| `Lt(a,b)` | `1 − DReLU(a − b)` | 8 | | |
| `Relu(a)`, `Max(a,b)` | DReLU then one multiplication | 9 | | |

A Velox multiplication depth is the L1/L2 pair plus hash broadcast, ~3 field
elements per gate per party in the linear protocol (`2n/(2t+1)`).

## Architecture

```
apps/*              implements OpsApplication:  Op::{Mul, Compare, Lt, Relu, Max, Trunc, FixedMul}
   │                per op-depth; one result per op back
   ▼
ops crate           OpsAdapter<F: MersennePrimeField, A: OpsApplication<F>> — implements Application<F>
                    compiles OpCounts into a flat engine depth layout, runs the per-op state
                    machines, assembles solved bits from the engine's signs
   │
   ▼
application crate   Application trait, unchanged except DepthInput::Reveal + on_reveal_complete
   │
   ▼
mpc engine          Multiply (existing) + Reveal (new) at flat, numbered depths; verifies both
```

The engine's view is what it is today: a flat sequence of depths, each one
`Multiply` batch or one `Reveal` batch, with `gates_per_depth` declared up
front. A reveal depth declares 0 gates, which `ApplicationPreprocessing::plan`
already accepts. No sub-depth numbering, no engine-side comparison state, no
new preprocessing types.

### Engine delta (complete)

- **E1.** `DepthInput::Reveal { depth, values }` — batched public
  reconstruction of degree-t sharings using the L1/L2 chunk trick factored out
  of `lin_mult.rs` / `rand_bit.rs` into `protocol/reveal/`. Results return via
  a new hook `on_reveal_complete(depth, values)` whose default impl returns
  `Err`, so `AnonymousBroadcast` is untouched. Every `([y], y)` pair is
  recorded in `verf_state`.
- **E2.** Reveal verification: in the verification phase, open one
  coin-weighted combination `Σ ρ_i ([y_i] − y_i)` through the O(n²)
  degree-checked path the output layer uses; abort if nonzero. Privacy does
  not depend on it (`y` is uniform and everything downstream stays shared),
  correctness does: without it a corrupt party can shift a revealed `y`
  consistently at every honest party via the redundancy-free L2 step.
  Alternative considered: chunks of `t+1` read as a degree-t polynomial so L1
  and L2 both detect errors — ~6 elements/value, no verification hook.
- **E3.** `delinearization_depth` derived from the declared profile instead of
  the hard-coded 5000.

### The ops layer

```rust
pub enum Op<F> {
    Mul(x, y), Compare(x) /* [x ≥ 0] */, Lt(a, b), Relu(x), Max(a, b),
    Trunc { x, d }, FixedMul { x, y, d },
}
pub struct OpBatch<F> { pub depth: usize, pub ops: Vec<Op<F>> }
pub struct OpCounts { pub ops_per_depth: Vec<OpProfile>, pub rand_bits: usize, pub output: usize }

pub trait OpsApplication<F>: Send + 'static {
    fn preprocessing_count(&self) -> OpCounts;
    async fn inputs(&mut self) -> Vec<FieldElement<F>>;
    async fn input_sharing_termination(&mut self, party, shares) -> Result<OpInput<F>>;
    async fn on_preprocessing_complete(&mut self, rand_bits) -> Result<OpInput<F>>;
    async fn on_depth_complete(&mut self, depth, results: Vec<FieldElement<F>>) -> Result<OpInput<F>>;
}
// OpInput = Waiting | Batch(OpBatch) | Done(outputs)
```

`Program::plan(&OpCounts) -> Layout` runs once at every party from the
agreed profile and fixes, per op-depth, its span of engine depths and what
each engine depth is. The span depends only on which kinds are present:

| op-depth contains | engine depths | rounds |
|---|---|---|
| `Mul` only | M | 1 |
| `Trunc` only | R | 1 |
| `Compare` (± `Mul`, `Lt`) | R, M×6 (tree), M (final XOR) | 8 |
| `Relu` / `Max` | Compare + M | 9 |
| `Trunc` + `Mul` | R, M | 2 |
| `FixedMul` (v1) | M, R | 2 |

Rule: one engine depth of reveals first (all ops' `y` / `c` values), then
multiplication depths; plain `Mul`s ride in the first multiplication depth of
the batch. Mixing kinds costs at most one extra round for the `Mul`s, never
for the long op. `gates_per_depth` for the engine comes from this table and
`CarryTree`'s level widths.

`OpsAdapter` holds the `Layout`, a cursor into the current op-depth's span,
and per-op state (the `(P, G)` slot vectors for comparisons; `[r]`,
`[r_msb]`, `[r']` for truncations). Each `on_depth_complete` /
`on_reveal_complete` from the engine advances every in-flight op one engine
depth and returns the next engine batch; when the span ends it calls the inner
app's `on_depth_complete(op_depth, results)` with one result per op in order.
The adapter keeps `ℓ × (#ops needing bits)` of the engine's random bits and
passes the rest to the inner app.

### Testing without a network

Sharings are linear, so a plaintext value is a valid sharing (a degree-0
polynomial). A fake engine where `Multiply` is the product and `Reveal` is the
identity drives the entire adapter — layout, scheduling, carry tree,
truncation — against integer references, exhaustively for small ℓ and
randomised at 61. Only the engine's own reveal is outside that harness; the
`testdata/10` fixture covers it.

## Tasks

One PR each. Ask before every commit.

- **T0 — preprocessing undercount fix.** `TODO.md`: "runs out one sharing
  short at depth 5005" (`rand_sh.rs` sizing /
  `multiplication_preprocessing_requirement`). Prerequisite: three new
  consumers on top of an off-by-one are unattributable.
- **T1 — `fields`: `MersennePrimeField: ProtocolField`** with `const BITS`,
  `to_canonical_u64`, `from_u64`; implemented for `Mersenne61Field` only
  (there is no M31 *base* field, only the Fp8 tower). Compile-time restriction
  at no cost to the engine.
- **T2 — `application`: `DepthInput::Reveal` + `on_reveal_complete`**
  (default `Err`), docs. Pure data; both apps compile unchanged.
- **T3 — `mpc`: reveal primitive (E1).** Factor the L1/L2 machinery,
  once-guards, hash agreement, record pairs. Regression via the existing
  fixture since `lin_mult` / `rand_bit` route through it.
- **T4 — `ops` crate, offline.** `CarryTree`, solved-bit assembly,
  ΠTrunc / ΠFixed-Mult local steps, `Op` / `OpBatch` / `OpCounts`,
  `Program::plan`, per-op state machines, `OpsAdapter`, plaintext fake-engine
  test suite. The bulk of the work; reviewable without running parties.
- **T5 — `mpc`: reveal verification (E2)** and the derived
  `delinearization_depth` (E3).
- **T6 — Bristol app onto `OpsApplication`.** Gate types `DRELU`, `LT`,
  `RELU`, `MAX`, `TRUNC d`, `FMUL d`; leveliser emits `OpCounts`;
  `docs/CIRCUIT_FORMAT.md`; test circuit under `testdata/circuits/`;
  end-to-end on `testdata/10` with `--field m61base`.
- **T7 — benchmark + `docs/comparison.md`** (carry-tree derivation, layout
  table, why nothing is opened), in the style of `docs/simd-m61-avx2.md`.

## Follow-ups (out of scope)

- `MaskedMultiply` engine primitive for the 1-round ΠFixed-Mult.
- `DepthInput::Parallel` so a batch's reveals and plain `Mul`s share a round.
- Folding the final XOR into the tree root with the two-layer DN trick
  (Liu et al. §5.2, Π2L-DN): 8 → 7 rounds for DReLU.
- Mersenne-31 base field as a `ProtocolField` (note `2^{-31}` per-comparison
  probability of `r ≡ 0`, which reveals `a`).

## Open decisions

- E2: deferred reveal check (recommended) vs. detect-at-L2 chunking
  (~6 elements/value).
- FixedMul v1 (2 rounds) acceptable for the first landing.
