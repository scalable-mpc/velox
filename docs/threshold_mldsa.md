# Threshold ML-DSA-44 (issue 19)

The `threshold_mldsa` application implements Quorus (Bienstock et al.,
"Quorus: Threshold ML-DSA", USENIX Security 2026, eprint 2025/1163) on the
Planner. The parties hold Shamir shares of an ML-DSA-44 secret key over the
ML-DSA field `F_q`, `q = 8380417 = 2^23 − 2^13 + 1`, and they make a
standard FIPS 204 signature. Each party checks the signature with RustCrypto
`ml-dsa` 0.1.1, an independent FIPS 204 implementation.

Only ML-DSA-44 is implemented. ML-DSA-65 and ML-DSA-87 are out of scope for
now.

## What one run does

Each run makes a new key, prepares `B` signing attempts (`--attempts`,
default 16) and signs one message (`--message`). The key exists only for the
run. The engine checks all multiplications at the end of the run, before it
opens the output wires.

| op-depth | op | width at `B = 16` | what it does |
|---|---|---|---|
| 1 | `Mul` | 30533 | `b0·b1` of each small-coefficient candidate |
| 2 | `Mul` | 30533 | the reject bit `b2·(b0 + b1 − b0·b1)` |
| 3 | `Reveal` | 30533 + 256 | the reject bits and the seed `ρ`; then `[s]`, `[e]`, `[e_w]`, `[t]`, `[y]`, `[w]` locally |
| 4 | `Reveal` | 1024 | `t`; then the public key, `tr` |
| 5 | `CanonicalMod { d: 44 }` | 16384 | `[w1]` of all attempts; then `[w0]` locally |
| 6 | `Reveal` | 16384 | `w1`; then `μ`, `c̃`, `c`, `[z]`, `[w̄0]` of each attempt locally |
| 7 | `LessThanPub` | 32768 | the bound tests of `z` and `w̄0` |
| 8 | `Mul` | 48 | the failure count of each attempt times 3 random sharings |
| 9 | `Reveal` | 48 | the verdicts |
| output | — | 1024 | `z` of the first attempt that passes |

Op-depths 1 to 4 are key generation (Quorus Protocol 5). Op-depths 1 to 3 and 5
are the preprocessing (Protocol 8). Op-depths 6 to 9 and the output are
signing (Protocols 4 and 5). Before op-depth 1, the Planner makes the
edaBits that `CanonicalMod` and `LessThanPub` use (49152 at `B = 16`).

## The code and the paper

### Small coefficients (Protocol 10)

The coefficients of `s`, `e` and `e_w` are uniform in `[−2, 2]`. A candidate
is `r = b0 + 2·b1 + 4·b2` for three random bits, and the parties keep it when
`r < 5`. Only the reject bit is opened. A kept `r` is uniform in `[0, 5)`, so
the opened bits tell nothing about the kept coefficients. The op-depth widths
must be known before preprocessing, so the run cannot draw more candidates.
Hoeffding's inequality gives the number of candidates: too few candidates
pass with a probability of at most `2^−40`, and then the run stops with an
error. All `s`, `e` and `e_w` of a run share one set of candidates, so `e_w`
needs no extra op-depths.

### `w1` for ML-DSA-44 (Protocol 8)

For ML-DSA-44, `(q − 1)/(2γ2) = 44` is not a power of 2. Every `w` in `F_q`
satisfies `HighBits(w) = ((−44·w + (q − 1)/2) mod q) mod 44`. The test
`rounding_over_all_of_f_q` in `apps/threshold_mldsa/src/mldsa.rs` checks this
for all `q` values of `w`.

Line 11 of Protocol 8 is not correct as printed. Functionality 7 defines
`F_BitLTC[B](x) = (B < x)`, so `b = [a < r]`, the wrap bit. With `(1 − b)` as
printed, the residue is wrong in all cases. With `b`, the residue is correct,
but the value is outside `[0, 44)` in about 48% of cases, so it needs a final
reduction. The Planner's `CanonicalMod { x, d }` (method A, see
`planner/src/ops/canonical_mod.rs`) has this reduction. It takes 13 rounds and
94 multiplications for each element at `d = 44`.

`[w0] = [w] − 2γ2·[w1]` is `LowBits(w)` modulo `q` for every `w`. This is
also true for the special case of `Decompose` where `r − r0 = q − 1`.

### The verdict (Protocols 3 and 4)

`LessThanPub` gives `[ok]` for each coefficient: `‖x‖ < B` is
`(x + B − 1) mod q < 2B − 1`, with `B = γ1 − β` for `z` and `B = γ2 − β` for
`w̄0` (as in Protocol 4).

Protocol 4 combines the `2048` results of an attempt with a batched OR
(Protocol 3). The code uses the failure count `f = Σ (1 − ok)` in its place.
`f ≤ 2048 < q`, so `f = 0` exactly when all coefficients pass. The parties
open `f·u` for 3 independent random sharings `u`. When `f ≠ 0`, each product
is uniform in `F_q \ {0}` unless its `u` is 0, so the products do not tell
the number of failures. A failing attempt passes only when all 3 `u` are 0:
the probability is `q^−3 < 2^−69` (decision D5 of the plan).

### Opening `z`

The paper opens `z` of a passing attempt with `F_Open`. The code makes it the
output wires. The engine opens output wires only after it checks all
multiplications of the run. So an adversary that adds errors to make a
failing attempt pass cannot see its `z`: the check finds the errors first.
When no attempt passes, the outputs are constant sharings of 0.

If `‖c·t0‖∞ ≥ γ2` or the hint has more than `ω` ones, the run gives no
signature (Protocol 5, line 7; decision D3).

## Costs and measurements

`scripts/bench_threshold_mldsa.py` runs the 10-party local fixture
(`--comp 10`) and reports the median of each phase:

- preprocessing: the engine's random sharings and random bits;
- circuit: all op-depths 1 to 9, with the edaBits before op-depth 1;
- signing: op-depths 6 to 9 only (the median over the 10 parties);
- verification: the end-of-run check of all multiplications and reveals;
- total: the run, until the outputs are open.

Measured on 2026-10-29 on one Apple M5 laptop (10 parties on one machine,
`--comp 10`, 3 runs per point, medians in milliseconds):

| B | random bits | edaBits | engine depths | preprocessing | circuit | signing | verification | total | runs with a signature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 72128 | 3072 | 32 | 1290 | 303 | 115 | 255 | 1901 | 0 of 3 |
| 4 | 285752 | 12288 | 32 | 4680 | 824 | 367 | 1501 | 7109 | 1 of 3 |
| 16 | 1140248 | 49152 | 32 | 26945 | 3274 | 1573 | 9724 | 39898 | 3 of 3 |

In each run, all 10 parties logged the same result. RustCrypto `ml-dsa`
accepted every signature. A run without a signature is a correct result:
no attempt passed the bound tests.

- The preprocessing phase is the largest cost. Most of the random bits are
  the bits of the edaBits (23 bits for each candidate).
- The number of engine depths does not change with `B`: the attempts run in
  parallel.
- In these 9 runs, 4 of the 19 attempts up to the first passing attempt
  passed (B = 1: 3 attempts, no pass; B = 4: 9 attempts, 1 pass; B = 16:
  7 attempts, 3 passes). The sample is small. FIPS 204 gives an expected
  4.25 attempts for
  ML-DSA-44 without the extra error `e_w` of Quorus.

Run the benchmark again with:

```bash
scripts/bench_threshold_mldsa.py run --reps 3 --attempts 1,4,16
scripts/bench_threshold_mldsa.py report
```
