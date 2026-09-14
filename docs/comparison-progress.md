# Comparison work — progress

Issue #5. Branch `issue-5-comparison-ops`. Plan in `docs/comparison-plan.md`.

Status legend: `todo` · `in progress` · `in review` · `done` · `blocked`.

| Task | Status | PR | Notes |
|---|---|---|---|
| T0 — preprocessing undercount fix | todo | | |
| T1 — `fields`: `MersennePrimeField` | todo | | |
| T2 — `application`: `Reveal` + `on_reveal_complete` | todo | | |
| T3 — `mpc`: reveal primitive (E1) | todo | | |
| T4 — `ops` crate (offline) | todo | | |
| T5 — `mpc`: reveal verification (E2), derived `delinearization_depth` (E3) | todo | | |
| T6 — Bristol app on `OpsApplication` | todo | | |
| T7 — benchmark + `docs/comparison.md` | todo | | |

## Decisions

- 2026-09-14 — Replace Liu et al.'s `ΠBitwise-LT` with CdH `BitLTL` /
  `CarryOutL` (85 mults, 6 depths at ℓ=61). See plan.
- 2026-09-14 — Engine gains only `Reveal` (+ its verification); all op logic
  lives in an `ops` adapter crate between applications and the `Application`
  trait. `MersennePrimeField` bounds only the adapter.
- 2026-09-14 — FixedMul v1 = `Mul` then `Trunc` (2 rounds); `MaskedMultiply`
  is a follow-up. *(pending confirmation)*
- 2026-09-14 — Reveal robustness: deferred coin-weighted check in the
  verification phase. *(pending confirmation)*

## Log

- 2026-09-14 — Branch created; plan and this file written. No code changes yet.
