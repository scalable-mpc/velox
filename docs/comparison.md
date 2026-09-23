# Comparison, truncation and fixed-point multiplication in Velox

How Velox computes `[a < b]`, `max`, `min`, `ReLU`, `Trunc_d` and fixed-point
products over a Mersenne prime field, what the engine had to gain for it, why
the bitwise comparison opens nothing, and what the measurements say. Issue #5.

Code: [`planner/`](../planner) (the operations, [`planner/README.md`](../planner/README.md)
for its API), [`planner/src/primitives/carry_tree.rs`](../planner/src/primitives/carry_tree.rs)
(the bitwise less-than), [`planner/src/ops/drelu.rs`](../planner/src/ops/drelu.rs)
and [`planner/src/ops/fixed_point.rs`](../planner/src/ops/fixed_point.rs) (the two
pipelines), [`mpc/src/protocol/public_reconstruction/`](../mpc/src/protocol/public_reconstruction)
(the reveal), [`mpc/src/protocol/tuple_verification/reveal_check.rs`](../mpc/src/protocol/tuple_verification/reveal_check.rs),
[`mpc/src/protocol/multiplication/masked_mult.rs`](../mpc/src/protocol/multiplication/masked_mult.rs).
Circuits: [`docs/CIRCUIT_FORMAT.md`](CIRCUIT_FORMAT.md) (the `LT`, `RELU`, `MAX`,
`MIN`, `DRELU`, `TRUNC d`, `FMUL d` gates). Tests: `planner/tests/plaintext.rs`,
`apps/bristol_circuit` (unit), `testdata/circuits/comparison.arith` (network).
Bench: [`scripts/bench_ops.py`](../scripts/bench_ops.py).

References: **[LXY24]** F. Liu, X. Xie, Y. Yu, *Scalable Multi-Party Computation
Protocols for Machine Learning in the Honest-Majority Setting*, USENIX Security
2024. **[CdH10]** O. Catrina, S. de Hoogh, *Improved Primitives for Secure
Multiparty Integer Computation*, SCN 2010.

Contents

1. [The problem](#1-the-problem)
2. [Two facts about a Mersenne prime](#2-two-facts-about-a-mersenne-prime)
3. [DReLU, and everything built on it](#3-drelu-and-everything-built-on-it)
4. [The bitwise less-than: a carry tree that opens nothing](#4-the-bitwise-less-than-a-carry-tree-that-opens-nothing)
5. [Truncation and fixed-point multiplication](#5-truncation-and-fixed-point-multiplication)
6. [edaBits](#6-edabits)
7. [What the engine gained](#7-what-the-engine-gained)
8. [The Planner](#8-the-planner)
9. [Testing](#9-testing)
10. [Measurements](#10-measurements)
11. [Caveats](#11-caveats)
12. [Appendix A — cost formulas](#12-appendix-a--cost-formulas)
13. [Appendix B — notation](#13-appendix-b--notation)

---

## 1. The problem

Velox evaluates arithmetic circuits over a field: it can add and multiply
Shamir sharings, and nothing else. Comparison is not a polynomial of low
degree in the field, and neither is truncation — dropping the low `d` bits of
a value — so both need the bits of a secret, which a sharing does not expose.
Machine-learning inference needs both at every layer: `ReLU` and `max` are
comparisons, and fixed-point arithmetic truncates after every multiplication.

The standard answer is to blind the secret with a random value whose bits are
also held in shared form, open the blinded value, and do the bit-level work
on the public result against the shared bits of the mask. [LXY24] does this
cheaply over a *Mersenne* prime field `p = 2^ℓ − 1`; this implementation
follows it, with one substitution: [LXY24]'s bitwise less-than opens values it
should not (§4), and it is replaced by [CdH10]'s carry-out tree, specialised.

## 2. Two facts about a Mersenne prime

Everything below rests on two properties of `p = 2^ℓ − 1` (ℓ = 61 for
`m61base`, 31 for `m31base`), exposed through
[`fields::MersennePrimeField`](../fields/src/mersenne_prime.rs):

- **`p` in binary is all ones**, so for `b ∈ [0, p)`, `bits(p − b) = ¬bits(b)`:
  the bit sharings of `p − b` are `1 − [b_i]`, for free.
- **`p` is odd**, so `msb(v) = lsb(2v)` for a `v` read as a signed integer
  (`[0, (p−1)/2]` non-negative, the rest negative): doubling a negative value
  wraps around `p` once and flips the parity. Sign extraction is one doubling
  and a least-significant-bit test.

Both are used on *public* values only — the opened `y = 2v + r` and
`c = v + 2^{ℓ−2} + r` — which is why the trait exposes the canonical integer
and its bits and nothing about sharings.

The extension fields over a Mersenne prime (`m61`, `m31`) are *not* such
fields: their elements are not integers mod `p`. The operations in this
document run over `m61base` and `m31base` only. The Planner itself runs over
every field; it asks `ProtocolField::MERSENNE_BITS` at plan time and refuses
these operations by name elsewhere (§8).

## 3. DReLU, and everything built on it

`DReLU(v) = [v ≥ 0]` for a sharing `v` with `|v| < 2^{ℓ−1}` ([LXY24] Protocol
5.1, with the bitwise less-than of §4). With `r` an edaBit (§6):

1. **Reveal** `y = 2v + r`. Since `msb(v) = lsb(2v)` and `2v = y − r`,
   `lsb(2v) = y_0 ⊕ r_0 ⊕ [y < r]` — the last term is the borrow of the
   subtraction `y − r` over the integers, i.e. whether it wrapped around `p`.
2. **The carry tree** over the public bits of `y` and the shared bits of `r`
   gives `[c] = [y < r]` in `⌈log₂ ℓ⌉` multiplication rounds (§4).
3. **One multiplication** `[b]·[c]` with `[b] = y_0 ⊕ [r_0]` (affine in
   `[r_0]`, since `y_0` is public), then
   `DReLU = 1 − ([b] + [c] − 2[b][c])`.

Everything else is DReLU on a difference plus at most one multiplication:

| op | from `d = DReLU(a − b)` | extra |
|---|---|---|
| `[a < b]` | `1 − d` | — |
| `max(a, b)` | `d · (a − b) + b` | 1 mult |
| `min(a, b)` | `a − d · (a − b)` | 1 mult |
| `ReLU(a)` | `max(a, 0)` | 1 mult |
| `[a ≥ 0]` (DRELU gate) | `1 − [a < 0]` | — |

`a − b` must keep its sign in the top bit, hence the domain
`|a|, |b| < 2^{ℓ−2}`. The public-operand variants (`ComparePub`, `MaxPub`,
`MinPub`) are the same arithmetic with `b` a public field element.

## 4. The bitwise less-than: a carry tree that opens nothing

### What [LXY24] does, and why it is not used

[LXY24]'s `ΠBitwise-LT` compares a public `y` with shared bits `[r]` in one
round through a prefix-OR, `ΠPreOR`, built on `ΠMultPub`: products of shared
bits are masked and **opened**. Those opened products are functions of the
secret bits of `r`, which is the mask protecting `v`, so opening them leaks
exactly what the comparison exists to hide. A one-round prefix-OR is not
available without opening something; this implementation pays `⌈log₂ ℓ⌉`
rounds instead and opens nothing but `y`.

### [CdH10]'s carry out

`[y < r] = 1 − CarryOut(y, ¬r, c_in = 1)`: the carry out of the top bit of
`2^ℓ + y − r`. With generate/propagate pairs per bit,
`p_i = y_i ⊕ ¬r_i`, `g_i = y_i ∧ ¬r_i`, the carry is a fold of the associative
operator

```
(P_L, G_L) ∘ (P_H, G_H) = (P_L · P_H,  G_H + P_H · G_L)
```

over the bits, low to high. Folded as a balanced binary tree it takes
`⌈log₂ ℓ⌉` levels, each one multiplication round, and every value stays
shared.

### Two specialisations, because `y` is public

Code: [`CarryTree`](../planner/src/primitives/carry_tree.rs).

- **A leaf pair costs one multiplication, not two.** With `y_i` public,
  `p_i` is affine in `[r_i]` and `g_i = y_i (1 − p_i)` is affine in `p_i`. For
  two adjacent leaves, `P = p_L p_H` is one product and
  `G = y_H (1 − p_H) + y_L (p_H − P)` is then local.
- **The carry-in is folded into leaf 0.** `(0, 1) ∘ (p_0, g_0) = (0, g_0 + p_0)`,
  local. Every node containing bit 0 — the whole left spine, root included —
  then has `P = 0` for free, and costs one multiplication, `P_H · G_L`.

The multiplications per level, at the Planner's two field sizes:

| ℓ | levels | multiplications per level | total |
|---|---|---|---|
| 61 | 6 | 30, 29, 15, 7, 3, 1 | 85 |
| 31 | 5 | 15, 15, 7, 3, 1 | 41 |

Three kinds of node, by what they cost: the **spine** — the leftmost node of
every level, the one containing bit 0 — costs one product (`P` stays 0);
a **leaf pair** — every other level-1 node — one product; an **inner** node —
every other node from level 2 up — two. An odd slot is carried up unchanged.
The module docs of `carry_tree.rs` draw the tree at `k = 8`. The tree is exhaustively tested against integer `<` for every
width up to 8 bits and randomised at 31 and 61.

So `[a < b]` costs **8 rounds and 86 multiplications per element** at ℓ = 61
(reveal + 6 levels + the xor multiplication), **7 and 42** at ℓ = 31.

## 5. Truncation and fixed-point multiplication

### ΠTrunc ([LXY24] Protocol 3.1)

For a sharing `v` with `|v| < 2^{ℓ−2}` and an edaBit `r`, open
`c = v + 2^{ℓ−2} + r` — the offset moves `v` into `[0, (p−1)/2]`, where
[LXY24] Theorem 3.2 holds for any `r` — and compute, locally,

```
[Trunc_d(v)] = Trunc_d(c) − [Trunc_d(r)] + (1 − [r_msb]) · c_msb · (2^{ℓ−d} − 1) − 2^{ℓ−d−2}
```

`Trunc_d(c)` of the public `c` is a shift with the top `d` bits filled with
its MSB; `[Trunc_d(r)]` and `[r_msb]` are linear in the edaBit's bits; the
middle term corrects the one wrap-around case, detected from the public MSB
of `c`; the last undoes the offset (Corollary 3.3). The result is
`Trunc_d(v)` up to an additive error in `{0, ±1, ±2}`, and `d` must lie in
`1 ..= ℓ − 3`. **One round, no multiplication.** Code:
[`fixed_point::unmask`](../planner/src/ops/fixed_point.rs).

### ΠFixed-Mult in one round ([LXY24] Protocol 3.2)

`Trunc_d(x · y)` would naively be a multiplication round and then a
truncation round. [LXY24] merges them: the multiplication already ends by
opening a masked degree-`2t` product, so let the mask be the truncation's
`r + 2^{ℓ−2}` and open `c = x·y + r + 2^{ℓ−2}` directly — then the local
unmasking above. **One round, one multiplication.**

This is the one thing an application layer cannot do for itself: opening a
degree-`2t` product needs the engine's degree-`2t` zero sharings (below), and
the product has to enter tuple verification. So the engine gained a
primitive for it, `MaskedMultiply` (§7).

## 6. edaBits

A random `r` together with sharings of its `ℓ` bits — [LXY24]'s
`ΠSolvedBits` output `([r], [r]_B)`, an edaBit in Escudero et al.'s terms.
They are assembled locally from the engine's random bits: the engine delivers
a random bit as a sharing of `±1` (it computes `s / sqrt(s²)` for a random
`s`), a 0/1 bit is `[b] = (1 + [s]) / 2`, and `[r] = Σ 2^i [b_i]`. Nothing
communicates.

`r` ranges over `[0, 2^ℓ)`; the value `2^ℓ − 1 ≡ 0 (mod p)` occurs with
probability `2^{−ℓ}` and is accepted, as [LXY24] does.

Every Mersenne-only op consumes one edaBit per element — `ℓ` random bits. The
Planner reserves them per op-depth in fixed slices laid out from the
declaration (§8), for the same reason the engine reserves multiplication
masks per depth: op-depth `d` must draw the same material at every party
however the parties happen to schedule.

## 7. What the engine gained

The comparison and truncation logic lives entirely in the Planner; the engine
(`mpc`) gained three primitives and two changes to verification. Its
depth-wise structure is unchanged: an application names one batch per
depth, and the engine runs it.

### One public reconstruction

The linear multiplication, the random-bit squaring and the new reveal all
end the same way — opening a batch of sharings to everyone — so they now
share one module, [`public_reconstruction`](../mpc/src/protocol/public_reconstruction).
A batch is padded to whole chunks of `2t+1`; chunk `i` is read as the
coefficients of a degree-`2t` polynomial `Z_i`. At **L1** each party sends
party `p` its share of every `Z_i(α_p)`, from which `p` interpolates those
points; at **L2** each party broadcasts its points, and everyone recovers the
chunk from `2t+1` of them; a **hash** of the values then pins agreement.
`O(1)` field elements per value per party.

Two options: the **degree** of the input sharings (`t` for reveals and random
bits, `2t` for products — at degree `t` the L1 step checks the `t` points of
redundancy) and a **privacy term**. What party `p` learns at L1 is the
sharing polynomial of `Z_i(α_p)`, a public combination of the inputs'
polynomials. For a product `f_x · f_y` those are not random — their upper
coefficients are fixed by honest parties' shares of the factors — so a fresh
degree-`2t` sharing of zero is added to each L1 message; `t+1` per chunk
suffice.

### `Reveal`

`DepthInput::Reveal { depth, values }` opens degree-`t` sharings, without the
privacy term and without preprocessing, and hands the values to
`on_reveal_complete`. **Contract:** reveal only values blinded by a fresh
random sharing — the L1 argument above only holds when the revealed sharings'
polynomials are uniformly random. `MaskReveal` in the Planner is the form
that guarantees it; every Planner op reveals `v + r` or `2v + r` for a fresh
edaBit `r`.

### The reveal check

L2 has no redundancy: a corrupt party can shift a revealed value consistently
at every honest party. The verification phase therefore checks every reveal
of the run. With `c` the delinearization coin the tuple verification already
agrees on, each party folds

```
[Δ] = Σ_i c^i · ([v_i] − v_i)
```

over all reveals (ascending depth, operand order) and broadcasts its share;
at `2t+1` shares the degree-`t` consistency is checked and `Δ = 0` required,
or the run aborts. An honest run gives `Δ = 0`; a set of shifts `δ_i` fixed
before `c` was known survives only if `Σ c^i δ_i = 0`, probability at most
`#reveals / |F|`. `[Δ]` is opened unmasked: it is a combination of
fresh-blinded sharings, independent of every secret.

It runs **in parallel** with the tuple compression — both start the moment
the coin is known — and the output is unmasked once both flags are set, so
it adds no round and no preprocessing. A run without reveals sends nothing.

### `MaskedMultiply`

`DepthInput::MaskedMultiply { depth, x, y, mask }`: the linear
multiplication with the caller's `[mask]` in place of a pool mask. The engine
forms `[x·y + mask]`, opens it with the multiplication's configuration
(degree `2t`, privacy term on, its own zero sharings from the depth's
reservation), and hands the public `c` to `on_reveal_complete` without
unmasking. The tuple `(x, y, c − [mask])` enters tuple verification, so a
shifted `c` fails the tuple check like any wrong product, and the `c` need
no place in the reveal check. `FixedMul` is its one user.

### The delinearization depth

Verification's own multiplications are keyed at depths above the circuit's.
That base depth was the constant 5000; it is now
`APPLICATION_DEPTH_OFFSET + depth + 1` from the declared circuit, **rounded
up to even** — a compression level's two batches are told apart by depth
parity, a rule the constant satisfied silently — and computed at
construction, where the declaration is now read once, because a faster
party's coin share can arrive before this party starts.

## 8. The Planner

The application-facing layer: a crate between applications and the engine,
documented in full in [`planner/README.md`](../planner/README.md). In short:

- An application implements `PlannerApplication` — the engine hooks one level
  up — and each op-depth runs one vector `Op`: `Mul`, `Add`, `Reveal`,
  `MaskReveal`, `Compare`/`ComparePub`, `Max`/`MaxPub`, `Min`/`MinPub`,
  `Truncate`, `FixedMul`.
- **One op type per op-depth.** Each op is a fixed list of steps, each one
  engine batch; the Planner compiles the declared op-depths into engine
  rounds before preprocessing, which is what sizes the engine's per-depth
  reservations and the edaBit slices.
- **Every application goes through it.** `anonymous_broadcast`, `btx_setup`,
  `reveal_probe` and `bristol_circuit` are all `PlannerApplication`s. The
  Planner runs over every protocol field: `Mul`, `Add` and `Reveal` anywhere
  (BTX over BLS12-381, anonymous broadcast over the Fp4 extension), the rest
  over `m61base`/`m31base`, refused by name elsewhere when the plan is
  compiled.

The `.arith` circuit format expresses the operations as gates; a level's
gates are grouped by type into op groups, one op-depth each
([`CIRCUIT_FORMAT.md`](CIRCUIT_FORMAT.md)).

## 9. Testing

- **Plaintext engine.** Sharings are linear, so a plaintext value is a valid
  (degree-0) sharing: an engine where `Multiply` is the product, `Reveal` the
  identity and `MaskedMultiply` is `x·y + mask` drives the whole Planner.
  `planner/tests/plaintext.rs` checks every op against its integer reference
  at ℓ = 61 and 31 — the comparison family exhaustively on small values and
  randomised over the domain, truncation within ±2 — and a non-Mersenne field
  running `Mul`/`Add`/`Reveal` with the rest refused. Every application's tests
  drive it through the real `Planner` the same way.
- **Primitives.** The carry tree exhaustively to 8 bits and randomised at 31
  and 61; edaBits and `Trunc_d(r)` against integers; the public reconstruction
  in-process over a local Shamir sharing; the reveal-check fold.
- **Network.** 10 local parties: `comparison.arith` (every op gate) over
  `m61base` and `m31base`, outputs cross-checked against the reference;
  `reveal_probe`; the benchmark below, which checks every output of every
  run.

## 10. Measurements

Ten parties (`t = 3`) on one machine — an Intel i7-8565U laptop, 4 cores /
8 threads, 15 GB — over loopback TCP, compression factor 10. Each point is a
circuit of `k` gates of one type at one level (`scripts/bench_ops.py`
generates it), two input parties, every gate an output wire; inputs are
uniform in `[−2^13, 2^13)`, inside every op's domain at both ℓ, with
`TRUNC`/`FMUL` dropping `d = 8` bits. Phases are the syncer's latencies at the
`n − t`-th party to finish, each measured from the end of the previous one.
**144 runs, 3 per point; every output of every party that reported was
checked against the integer reference and matched** — exactly for
`MUL`/`LT`/`RELU`/`MAX`, within the ±2 of [LXY24] for `TRUNC`/`FMUL` (largest
error seen: 2).

**m61base** (ℓ = 61), median of 3 runs, milliseconds:

| op | rounds | k | tuples verified | preprocessing | online | verification | output |
|---|---|---|---|---|---|---|---|
| `MUL` | 1 | 1 | 9 | 476 | 5 | 16 | 90 |
| `MUL` | 1 | 64 | 72 | 484 | 6 | 32 | 163 |
| `MUL` | 1 | 1024 | 1,032 | 590 | 15 | 63 | 204 |
| `MUL` | 1 | 4096 | 4,104 | 895 | 37 | 84 | 693 |
| `LT` | 8 | 1 | 154 | 599 | 54 | 48 | 109 |
| `LT` | 8 | 64 | 9,416 | 986 | 107 | 80 | 110 |
| `LT` | 8 | 1024 | 150,536 | 5,891 | 589 | 258 | 226 |
| `LT` | 8 | 4096 | 602,120 | 21,544 | 2,184 | 741 | 672 |
| `RELU` | 9 | 1 | 155 | 656 | 106 | 49 | 118 |
| `RELU` | 9 | 64 | 9,480 | 1,083 | 115 | 81 | 122 |
| `RELU` | 9 | 1024 | 151,560 | 6,446 | 639 | 259 | 225 |
| `RELU` | 9 | 4096 | 606,216 | 21,112 | 2,226 | 696 | 610 |
| `MAX` | 9 | 1 | 155 | 595 | 51 | 47 | 105 |
| `MAX` | 9 | 64 | 9,480 | 983 | 106 | 77 | 109 |
| `MAX` | 9 | 1024 | 151,560 | 5,704 | 634 | 262 | 228 |
| `MAX` | 9 | 4096 | 606,216 | 21,210 | 2,046 | 675 | 633 |
| `TRUNC` | 1 | 1 | 68 | 709 | 8 | 41 | 133 |
| `TRUNC` | 1 | 64 | 3,912 | 1,059 | 10 | 82 | 150 |
| `TRUNC` | 1 | 1024 | 62,472 | 4,382 | 45 | 222 | 278 |
| `TRUNC` | 1 | 4096 | 249,864 | 13,878 | 92 | 353 | 621 |
| `FMUL` | 1 | 1 | 69 | 610 | 7 | 34 | 104 |
| `FMUL` | 1 | 64 | 3,976 | 859 | 10 | 73 | 112 |
| `FMUL` | 1 | 1024 | 63,496 | 4,154 | 32 | 160 | 222 |
| `FMUL` | 1 | 4096 | 253,960 | 13,999 | 69 | 378 | 700 |

**m31base** (ℓ = 31), median of 3 runs, milliseconds:

| op | rounds | k | tuples verified | preprocessing | online | verification | output |
|---|---|---|---|---|---|---|---|
| `MUL` | 1 | 1 | 9 | 630 | 6 | 24 | 111 |
| `MUL` | 1 | 64 | 72 | 624 | 9 | 35 | 125 |
| `MUL` | 1 | 1024 | 1,032 | 658 | 13 | 71 | 196 |
| `MUL` | 1 | 4096 | 4,104 | 845 | 30 | 75 | 518 |
| `LT` | 7 | 1 | 82 | 683 | 52 | 34 | 105 |
| `LT` | 7 | 64 | 4,680 | 772 | 64 | 69 | 109 |
| `LT` | 7 | 1024 | 74,760 | 2,850 | 291 | 159 | 189 |
| `LT` | 7 | 4096 | 299,016 | 9,319 | 927 | 310 | 512 |
| `RELU` | 8 | 1 | 83 | 593 | 52 | 34 | 106 |
| `RELU` | 8 | 64 | 4,744 | 807 | 76 | 73 | 118 |
| `RELU` | 8 | 1024 | 75,784 | 2,847 | 293 | 143 | 193 |
| `RELU` | 8 | 4096 | 303,112 | 9,409 | 979 | 325 | 508 |
| `MAX` | 8 | 1 | 83 | 616 | 49 | 39 | 112 |
| `MAX` | 8 | 64 | 4,744 | 776 | 76 | 65 | 110 |
| `MAX` | 8 | 1024 | 75,784 | 2,886 | 309 | 143 | 195 |
| `MAX` | 8 | 4096 | 303,112 | 9,227 | 951 | 325 | 455 |
| `TRUNC` | 1 | 1 | 40 | 595 | 9 | 44 | 121 |
| `TRUNC` | 1 | 64 | 1,992 | 778 | 5 | 72 | 151 |
| `TRUNC` | 1 | 1024 | 31,752 | 2,074 | 18 | 120 | 195 |
| `TRUNC` | 1 | 4096 | 126,984 | 6,151 | 50 | 246 | 553 |
| `FMUL` | 1 | 1 | 41 | 652 | 8 | 44 | 118 |
| `FMUL` | 1 | 64 | 2,056 | 797 | 7 | 80 | 132 |
| `FMUL` | 1 | 1024 | 32,776 | 2,014 | 23 | 126 | 188 |
| `FMUL` | 1 | 4096 | 131,080 | 6,261 | 41 | 198 | 493 |

`tuples verified` includes the random-bit squarings: every random bit is one
multiplication, so an edaBit adds ℓ tuples. `TRUNC` at `k = 4096`, ℓ = 61, is
`4096 · 61 + 8 = 249,864` tuples and no circuit multiplication at all; `LT`
adds its `86` per element to that.

What the numbers say:

- **Preprocessing dominates, and it is the random bits.** At `k = 4096`,
  ℓ = 61, `LT` spends 21.5 s in preprocessing against 2.2 s online; `TRUNC`,
  which multiplies nothing in the online phase, still spends 13.9 s. Each
  edaBit costs ℓ random bits, each a sharing, a squaring multiplication and a
  public reconstruction, and each of those multiplications then goes through
  verification. `MUL` at the same width preprocesses in 0.9 s. Anything that
  makes random bits cheaper — or reuses them — is where the next factor is.
- **Online, the comparison is its round count at small widths and its
  multiplications at large ones.** 8 rounds of `LT` take ~50 ms at `k = 1`
  (≈ 6 ms per local round), growing to 0.6 s at `k = 1024` and 2.2 s at
  `k = 4096` — 0.53 ms per comparison, 86 multiplications each. `TRUNC` and
  `FMUL` stay under 0.1 s online at every width: one round, no tree.
- **ℓ = 31 roughly halves everything that scales with ℓ:** half the random
  bits and a 41-multiplication tree instead of 85. `LT` at `k = 4096`
  preprocesses in 9.3 s and runs online in 0.9 s, one round fewer. (Over
  `m31base` the verification is `2^{−31}`-sound — §11.)
- **Verification and output are small and sublinear.** Tuple compression
  folds 600k tuples in ~0.7 s; the reveal check adds no round of its own.
- **The operations cost what the plan says.** Rounds per point are 1 (`MUL`,
  `TRUNC`, `FMUL`), 8 / 7 (`LT`), 9 / 8 (`RELU`, `MAX`) at ℓ = 61 / 31, read
  off the Planner's own log line in every run.

Two things the harness saw beyond the numbers.

- **Stragglers.** In three `MAX`, `k = 4096` runs one party had not finished
  its own verification when the harness stopped the run, 3 s after the syncer
  reported `n − t` parties done — at that width a party folds 606k tuples
  while nine others compete for eight hardware threads. Re-run six times with
  a 15 s grace period (`--grace`, now the default), five runs delivered the
  output to all ten parties (m61base: 26.9 s and 24.2 s total; m31base:
  10.7, 10.6, 10.6 s). They were stragglers, not stalls.
- **One stall.** The sixth re-run (m61base) stopped after every party had
  reconstructed the output wires and nine of ten had passed verification: the
  output-agreement BA exhausted its hybrid-model coins at every party and
  `on_output` never fired. That is the consensus-library behaviour in §11;
  across all 150 runs the coin exhaustion was logged in 10 and stopped 1.

## 11. Caveats

- **Verification soundness is bounded by the sharing field.** The
  delinearization coin — and so the tuple check and the reveal check — lives
  in `F`, not in an extension. Over `m61base` that is `~2^{−61}` per check,
  times the number of tuples or reveals; over `m31base` it is `~2^{−31}`,
  which is a benchmark configuration, not a secure one. (The ACSS proofs are
  lifted to an extension; the verification phase is not — see
  `ProtocolField::Ext`.)
- **The end-of-protocol agreement can run out of coins.** The
  output-agreement MVBA's binary BA runs on 15 deterministic hybrid-model
  coins and sometimes exhausts them (`Coins unavailable, abandoning BBA
  round 15`). Across the 150 benchmark runs this was logged in 10 and
  stopped one run — after every party had reconstructed the output but before
  `on_output`; one of four earlier `comparison.arith` runs stopped the same
  way. It is worth its own task: the BA should draw real coins, or more of
  them. The computation is complete
  and correct either way; the consensus library is untouched on this
  branch.
- **A level mixing op types is serialised.** The Planner takes one type per
  op-depth, so a circuit level with a `MUL` and an `LT` costs 1 + 8 rounds.
- **The comparison domain is `|v| < 2^{ℓ−2}`**, i.e. `2^59` / `2^29`, and
  truncation's error is `±2`, both inherited from [LXY24].
- **No WAN numbers yet.** The measurements are 10 parties on one machine;
  latency is dominated by computation and local networking, not by the round
  count. The AWS scripts in `benchmark/` do not yet drive `bristol_circuit`.

## 12. Appendix A — cost formulas

Per element, `L = ⌈log₂ ℓ⌉` tree levels, `M(ℓ)` the tree's multiplications
(85 at ℓ = 61, 41 at ℓ = 31):

| op | rounds | multiplications | edaBits (ℓ random bits each) | opened |
|---|---|---|---|---|
| `Mul` | 1 | 1 | 0 | masked product |
| `Add` | 0 | 0 | 0 | — |
| `Reveal` | 1 | 0 | 0 | the value |
| `MaskReveal` | 1 | 0 | 1 | `x + r` |
| `Truncate` | 1 | 0 | 1 | `x + 2^{ℓ−2} + r` |
| `FixedMul` | 1 | 1 (masked) | 1 | `x·y + 2^{ℓ−2} + r` |
| `Compare`, `ComparePub` | `L + 2` | `M(ℓ) + 1` | 1 | `2(a−b) + r` |
| `Max`, `Min`, `…Pub` | `L + 3` | `M(ℓ) + 2` | 1 | `2(a−b) + r` |

Engine preprocessing per multiplication: one random mask (none for a masked
one) and `(t+1)/(2t+1)` degree-`2t` zero sharings; per reveal, none; every
multiplication, masked or not, adds one tuple to verification.

## 13. Appendix B — notation

| | |
|---|---|
| `p = 2^ℓ − 1` | the Mersenne prime; ℓ = 61 (`m61base`) or 31 (`m31base`) |
| `[x]` | a degree-`t` Shamir sharing of `x` |
| `n`, `t` | parties and corruption threshold, `n ≥ 3t + 1` |
| signed reading | `x ≤ (p−1)/2` is `x`, larger is `x − p` |
| `Trunc_d(v)` | `v` shifted right by `d` keeping the sign: `⌊v / 2^d⌋` towards zero |
| edaBit | a random `[r]` with `[r_0] … [r_{ℓ−1}]` |
| op-depth | one Planner operation, one or more engine rounds |
| engine depth / round | one batch the engine runs: a multiply, a reveal or a masked multiply |
