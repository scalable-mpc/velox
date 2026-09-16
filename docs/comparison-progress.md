# Comparison work — progress

Issue #5. Branch `issue-5-comparison-ops`. Plan in `docs/comparison-plan.md`.

Status legend: `todo` · `in progress` · `in review` · `done` · `blocked`.

| Task | Status | PR | Notes |
|---|---|---|---|
| T0 — preprocessing undercount | done | | Already fixed by `fb45d90`; verified on master `7395e86` (comp 10/64/256). Stale `TODO.md` note removed; budget audit table added to the plan. |
| T1 — `fields`: `MersennePrimeField` | done | | Both M61 and M31 base fields. Scope grew on request: `ProtocolField` for M31 base + Fp8 tower, `--field m31`/`m31base`; fixture runs clean over both. |
| T2 — `application`: `Reveal` + `on_reveal_complete` | todo | | |
| T3 — `mpc`: reveal primitive (E1) | todo | | |
| T4 — `ops` crate (offline) | todo | | |
| T5 — `mpc`: reveal verification (E2), derived `delinearization_depth` (E3) | todo | | |
| T5b — `mpc`: `MaskedMultiply` (E4) | todo | | |
| T6 — Bristol app on `OpsApplication` | todo | | |
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

## Log

- 2026-09-14 — Branch created; plan and this file written. No code changes yet.
- 2026-09-15 — Rebased onto master after PR #14 (BTX setup) merged. E2 approved; FixedMul v2 chosen.
- 2026-09-15 — T0: undercount found already fixed (`fb45d90`); three fixture runs clean. `TODO.md` note dropped, budget audit recorded in the plan.
- 2026-09-15 — T1 implemented. `MersennePrimeField` for M61 + M31; `ProtocolField` for M31 base and Fp8 (Tonelli–Shanks sqrt); 50 field tests pass; `testdata/10` fixture completes over `m31` and `m31base`. Caveat recorded: tuple verification is 2^-31-sound over M31.
