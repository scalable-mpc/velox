# Comparison work — progress

Issue #5. Branch `issue-5-comparison-ops`. Plan in `docs/comparison-plan.md`.

Status legend: `todo` · `in progress` · `in review` · `done` · `blocked`.

| Task | Status | PR | Notes |
|---|---|---|---|
| T0 — preprocessing undercount | done | | Already fixed by `fb45d90`; verified on master `7395e86` (comp 10/64/256). Stale `TODO.md` note removed; budget audit table added to the plan. |
| T1 — `fields`: `MersennePrimeField` | done | | Both M61 and M31 base fields. Scope grew on request: `ProtocolField` for M31 base + Fp8 tower, `--field m31`/`m31base`; fixture runs clean over both. |
| T2 — `application`: `Reveal`, `MaskedMultiply`, `on_reveal_complete` | done | | Counts shape and `plan` charging included; engine stub arms until T3/T5b. Also fixed the stale `mpc` lib-test helper left by the BTX merge. |
| T3 — `mpc`: public reconstruction overhaul + reveal (E1) | done | | `public_reconstruction/` module (degree/privacy options, hash agreement, state on the depth object); `lin_mult` + `rand_bit` rewired; `apps/reveal_probe` e2e over m61/m61base/m31base; every fixture clean. |
| T4 — the Planner (`planner/`, offline) | done | | edaBits, carry tree, op vocabulary, layout, executors, `Planner: Application`; 16 tests incl. a plaintext engine running every op at ℓ = 61 and 31. Nothing on the network until T5b/T6. |
| T5 — `mpc`: reveal verification (E2), derived `delinearization_depth` (E3) | done | | `reveal_check.rs`: `[Δ]` folded with the delinearization coin the moment it is known, opened unmasked at `2t+1` with the degree-t check, in parallel with the tuple compression; the output waits for both flags. No new preprocessing, no extra round. `delinearization_depth` derived (even) at `Context::spawn`. `reveal_probe` over m61/m31base 10/10 pass; anonymous_broadcast skips the round. Fault injection deferred. |
| T5b — `mpc`: `MaskedMultiply` (E4) | done | | `masked_mult.rs`: caller's mask, zero sharings from `for_masked_depth`, `ReconConfig::MULTIPLICATION`, public `c` via `on_reveal_complete`, tuple `(x, y, c − [mask])` verified. `reveal_probe` gained a masked depth (one gate per dealer); 10/10 verify over m61 and m31base, 19 tuples through verification. |
| T6 — Bristol app on the Planner | done | | Gate set `SUB`, `LT`, `DRELU`, `RELU`, `MAX`, `MIN`, `TRUNC d`, `FMUL d`; levels hold `OpGroup`s by type; `BristolCircuit` is a `PlannerApplication` (m61base/m31base only). `comparison.arith` verified through the Planner on a plaintext engine and on the network over m61base and m31base; outputs cross-checked against the reference. |
| T7 — benchmark + `docs/comparison.md` | todo | | |

## Decisions

- 2026-09-14 — Replace Liu et al.'s `ΠBitwise-LT` with CdH `BitLTL` /
  `CarryOutL` (85 mults, 6 depths at ℓ=61). See plan.
- 2026-09-14 — Engine gains only `Reveal` (+ its verification); all op logic
  lives in an `ops` adapter crate between applications and the `Application`
  trait. `MersennePrimeField` bounds only the adapter.
- 2026-09-15 — FixedMul goes straight to v2: a `MaskedMultiply` engine
  primitive (ΠFixed-Mult in 1 round). The 2-round `Mul` + `Trunc` v1 is
  dropped.
- 2026-09-15 — Reveal robustness: deferred coin-weighted check in the
  verification phase (E2). Approved.
- 2026-09-16 — The ops layer is the **Planner**: a crate bridging
  application and engine, hooks of the same shape as `Application`
  returning `OpDepthInput` with ops Mul, Add, Compare, ComparePub, Max, Min,
  MaxPub, MinPub, Reveal, MaskReveal, Truncate, FixedMul (values, not wire
  ids); one `Ops` per op-depth; per-op-depth `OpProfile` declared up front;
  the Planner manages edaBits. Bristol moves to it in T6; other apps stay.
- 2026-09-18 — One operation per op-depth in the Planner (no mixing of kinds
  in a batch); an application needing two kinds side by side schedules two
  op-depths. Removes the contribution merging, per-op offsets and the
  twelve-field profile.
- 2026-09-16 — T3 is an overhaul, not an add-alongside: one
  `public_reconstruction` module with degree (t / 2t) and privacy (zero
  term) options, its state attached to each depth's state object; module
  named `public_reconstruction/`; `quad_mult` left alone (unreachable).

## Log

- 2026-09-22 — T6 implemented. `circuit`: `GateType` gains `Sub`, `Lt`, `DRelu`, `Relu`, `Max`, `Min`, `Trunc(d)`, `FMul(d)`; `Gate.inputs` is a list; `Depth.op_groups` (`OpGroup`) groups round-costing gates by type in file order; `evaluate_circuit` (Mersenne, signed). Parser takes `d` after the type token. `BristolCircuit` is a `PlannerApplication` only — the engine `Application` impl is gone, the binary takes m61base (default) or m31base; `DRELU = 1 − ComparePub(a, 0)` locally. Network runs of `comparison.arith` (10 parties, m61base and m31base) gave `[15, 1, 12, 11, −7, 1, −6, 2, 0]` against the reference `[15, 1, 12, 11, −7, 0, −4, 2, 0]` — the two truncations off by +1 and −2, inside Liu's ±2. Seen again on both runs: `binary_ba` "Coins unavailable, abandoning BBA round 15" in the consensus library after the outputs are out — pre-existing, not from this branch.
- 2026-09-21 — T5b implemented: `DepthInput::MaskedMultiply` runs (`multiplication/masked_mult.rs`, `ReconKind::MaskedMultiplication`); the engine stub arm is gone. `reveal_probe` now also masked-multiplies each dealer's first two inputs under its first blind and checks `c − mask = x·y` in `on_output`. Every Planner op can now run on the network; T6 is application work only.
- 2026-09-21 — T5 implemented: E2 as `tuple_verification/reveal_check.rs` + `ProtMsg::RevealCheckShare` (one field element per party, started at delinearization in parallel with the tuple compression, output gated on both — no round of its own); E3 as `delinearization_depth = APPLICATION_DEPTH_OFFSET + depth + 1` rounded to even, computed at construction — the declaration (`preprocessing_count`, `random_wires`) is now read once in `Context::spawn` and `init_rand_sh` uses the cached copies. Found on the way: compression levels distinguish their two batches by depth parity, so the delinearization depth must be even. Negative-path fault injection skipped on request (`--byzantine` remains unwired). `planner/README.md` written.
- 2026-09-14 — Branch created; plan and this file written. No code changes yet.
- 2026-09-15 — Rebased onto master after PR #14 (BTX setup) merged. E2 approved; FixedMul v2 chosen.
- 2026-09-15 — T0: undercount found already fixed (`fb45d90`); three fixture runs clean. `TODO.md` note dropped, budget audit recorded in the plan.
- 2026-09-18 — Plan vocabulary: `OpStep` (per-element step an op declares), `EngineRound` (a step placed at an engine depth), `OpDepthPlan`, `Plan` (`plan.rs`, was `layout.rs`); `Plan::compile`, `Planner::plan()`, `ops::steps_of`.
- 2026-09-18 — The `application` crate is folded into the Planner as `planner::api::engine` — the engine's API lives beside the application API it is built under; `mpc` and `velox` now depend on `planner`. `Kind` renamed `Type` throughout the Planner.
- 2026-09-18 — Planner reorganised into api/ (application, engine), planner core, layout, ops/ (one file per op on one template: `operands` / `on_step_complete` / `into_result`), primitives/; then simplified to one operation per op-depth, each type owning its schedule. 16 tests unchanged in coverage.
- 2026-09-16 — T4 implemented: `planner/` crate, re-exported from `velox`. Ops Mul/Add/Compare/ComparePub/Max/Min/MaxPub/MinPub/Reveal/MaskReveal/Truncate/FixedMul; rounds table verified by tests; comparison exhaustive on small values and randomised over the domain; Truncate/FixedMul within ±2.
- 2026-09-16 — T3 implemented: module + 7 offline tests, `lin_mult`/`rand_bit` on it, `Reveal` live, `reveal_probe` (10/10 parties verify) over three fields; anonymous_broadcast (comp 10/64, m31base), bristol and btx fixtures clean. `mpc` gains dev-deps `rand_core`/`rand_chacha`.
- 2026-09-15 — T2 implemented: two `DepthInput` variants with constructors, `on_reveal_complete` (default `Err`), `masked_gates_per_depth` + `plan` charging + `for_masked_depth`, stub dispatcher arms; 62 tests across application/mpc/apps pass; m61 fixture clean.
- 2026-09-15 — T1 implemented. `MersennePrimeField` for M61 + M31; `ProtocolField` for M31 base and Fp8 (Tonelli–Shanks sqrt); 50 field tests pass; `testdata/10` fixture completes over `m31` and `m31base`. Caveat recorded: tuple verification is 2^-31-sound over M31.
