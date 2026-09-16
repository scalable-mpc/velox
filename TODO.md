# TODO

## Rayon `par_iter` blocks a tokio worker inside the MPC `select!` loop

**Status:** Tiers 1-4 fixed (see "What was done" below).
Tracked as https://github.com/scalable-mpc/velox/issues/3.

### What's happening

Tokio and rayon each own an independent, uncoordinated thread pool sized to the
core count:

- `node/src/main.rs:12` — `#[tokio::main]` with no `worker_threads` override, so
  the multi-thread runtime gets `available_parallelism()` workers (N).
- No `ThreadPoolBuilder` anywhere in the workspace and no `RAYON_NUM_THREADS` in
  `benchmark/` or `scripts/`, so every `par_iter` runs on rayon's *global* pool,
  also N threads.

That is ~2N OS threads on N cores, arbitrated only by the Linux scheduler.

Every rayon call site sits inside an `async fn` that is `.await`ed directly on
the MPC context's `select!` loop task (`mpc/src/context.rs:441`). For example
`mpc/src/protocol/multiplication/quad_mult.rs:52` runs `into_par_iter().collect()`
between a state mutation and a `self.broadcast(...).await`.

`par_iter().collect()` called from a non-rayon thread goes through
`Registry::in_worker_cold`: it injects the job and then **blocks the calling
thread on a latch**. The caller does not become a rayon worker and does not
yield to tokio, so the future is never suspended — tokio cannot reuse that
worker thread for the duration of the job.

### Consequences

Network I/O is *not* stalled. The receive path lives on separate tasks
(`libnet-rs` `plaintcp/receiver.rs:38` spawns one task per peer connection;
`mpc/src/handlers/handler.rs` does a non-blocking `UnboundedSender::send` plus a
`Pong` ack), and the sub-protocol contexts (`acss_ab`, `sh2t`, `ctrbc`,
`avid_ab`, `fin_mvba`) are their own spawned tasks. Those keep running on the
remaining N-1 workers.

What stalls is the MPC state machine itself — `process_msg`,
`handle_acss_term_msg`, `handle_acs_output`, and the depth-advancing logic.
Messages queue instead of being consumed:

- `net_recv` is **unbounded** (`mpc/src/context.rs:203`). No backpressure, so a
  long rayon job shows up as inbox RSS growth rather than as a slowdown signal
  to peers. This compounds the peak-memory work in `mult_state` / `verf_state`.
- Sub-protocol output channels are bounded (10000; 500000 for `avss` at
  `mpc/src/context.rs:224`). A long enough stall blocks those producers and
  propagates backpressure into the ACSS/AVSS tasks.

Parallelism accounting, for reference:

- Nested `par_iter` does not multiply threads — e.g.
  `mpc/src/protocol/tuple_verification/compress_tup.rs:158` with inner
  `par_iter` at `:173-174`, and the nesting in `fields/src/poly.rs`, all feed the
  same global work-stealing deque. Thread count stays flat at N.
- Concurrent rayon calls from different tokio tasks do not each get N threads;
  they share the global injector. Aggregate rayon parallelism is capped at N.
- Oversubscription cost is bounded by the scheduler quantum (~1-4 ms), not by
  job length, so the network path sees jitter rather than serialization.

No `block_on` appears inside any rayon closure, so the classic rayon/tokio
deadlock is not present.

### Still open

1. **Backpressure.** `net_recv` is still unbounded (`mpc/src/context.rs:203`).
   Yielding means the loop keeps draining during compute, so the inbox grows
   more slowly, but nothing bounds it.

2. **Pool sizing.** Two uncoordinated N-thread pools on N cores; see below.

3. **Measurement.** None of this has been timed. The n=10 / 64-message fixture
   is far too small for the jobs to be long enough to matter (both master and
   this branch run it in ~1.8s). Instrumenting the converted sites and sweeping
   `RAYON_NUM_THREADS` would say which conversions actually pay.

### What was done

`fields::rayon_async` (`fields/src/par.rs`) is the house idiom: hand the job to
`rayon::spawn` and await a oneshot, so the future actually suspends and the
calling task's worker is free while the job runs at full rayon width. It
generalises the pattern `secret_sharing/avid_ab/src/rs.rs` already used for
erasure coding. `block_in_place` was considered and rejected — it keeps the job
on the tokio worker and only asks the runtime to compensate.

Converted (each takes ownership of its inputs via `std::mem::take` first, and
re-acquires the borrowed state after the await):

- `quad_mult.rs` — the matmul core and the L1 interpolation.
- `lin_mult.rs` — L1 and L2 interpolation.
- `rand_sh.rs` — the three-batch preprocessing combine (now one job), both
  secret-batch generators, and the per-message share deserialisation.
- `rand_bit.rs` — L1/L2 interpolation, the `sqrt`+`inv` batch, and the final
  bit derivation.
- `rand_mask.rs` — both Vandermonde combines.
- `compress_tup.rs` — the whole `ex_compr` block.
- `poly.rs` — `generate_evaluation_points{,_opt,_fft}`, which run on the
  ACSS/Sh2t dealer tasks.

Shape fixes, which yielding would only have hidden:

- `compress_tup.rs` second-set evaluation was a sequential `for` over the
  polynomials with a `par_iter` over the (small) point set inside it — parallel
  on the short axis, serial on the long one. Now one `powers_matrix * coeffs`
  GEMM, the idiom the same function already used for the first set. Pinned by
  `fields/tests/gemm_eval_equiv.rs`.
- `rand_bit.rs` debug path reconstructed the entire batch and then
  `truncate(100)` to log 100 values. It now truncates first.
- `poly.rs` `all_polys_positive` was a `par_iter().all()` over an `Option`
  discriminant check. Sequential now.

Once-guard fixed as a prerequisite: `verify_termination` in `rand_sh.rs` keyed
its once-ness on `rand_sharings_mult.len() > 0`, a payload field that is later
`split_off` and drained. Yielding there would have let a message processed
during the job re-enter and run preprocessing twice. It now uses a real
`combine_started: bool` on `RandSharings`, set *before* the job is dispatched.

The other converted sites were already safe: `quad_mult`, `lin_mult` and
`rand_bit` all set their `*_reconstruction_done` / `*_started` flag before the
work, and `verify_depth_mult_termination` cannot terminate a depth while its
reconstruction is in flight (it requires `reconstructed_len > 0`, which is what
the job is about to produce).

### Interpolation: closed form instead of Gaussian elimination

**Status:** done.

`inverse_vandermonde` was O(n^3) sequential Gaussian elimination with a `clone()`
per inner operation, reached from thirteen call sites. It is not redundant work -
the reconstruction sites build their point set from whichever `n-t` senders
arrive first, so the matrix genuinely differs each round and memoisation does not
apply. It was an Amdahl serial fraction: it does not shrink with cores, and it
dominates wall clock whenever the parallel remainder is small (`g < 2*n*N`, so
roughly `g < 3200` at `N=49`).

Two replacements, both in `fields/src/poly.rs`:

- `inverse_vandermonde_from_points` - the Lagrange closed form, O(n^2). Column
  `i` of the inverse is the `i`-th basis polynomial's coefficient vector; build
  `master(X) = prod_j (X - x_j)` once, synthetic-divide by `(X - x_i)` per
  column, scale by `1 / prod_{j != i}(x_i - x_j)` with all `n` inversions
  batched into one. Note *column*, not row - the transpose compiles and silently
  returns wrong answers.
- `lagrange_coefficients_at_zero` + `interpolate_at_zero` - for the sites that
  only ever read the secret. Six of the eight dynamic sites built the full
  `n x n` inverse, ran a matvec, and kept one number; `rand_mask` even had a
  comment saying it needed only row 0. Per reconstruction with `g` groups this
  goes from `2n^3` serial + `g*n^2` parallel to `n^2` serial + `g*n` parallel -
  an n-fold cut in the parallel half too, so unlike the closed-form inverse it
  keeps paying as `g` grows.

Constant-term sites: `quad_mult` (L1), `lin_mult` (L1), `rand_bit` (L1 and the
debug path), `rand_mask`, `common_coin`. Full-inverse sites, which need every
coefficient: `lin_mult` (L2), `rand_bit` (L2), `compress_tup` (first set and the
h polynomial), and the three internal uses in `poly.rs`.

`inverse_vandermonde` itself is retained, unused by protocol code, as the
reference the closed form is tested against
(`fields/tests/vandermonde_closed_form.rs`: bit-identical across n=1..13 and
four point-set shapes, plus `inv * V == I`, a round-trip through a known
polynomial, and the `x=0` edge case).

### Open question raised by the compress_tup rewrite

`x_poly_evals_ss` / `y_poly_evals_ss` were initialised as
`vec![vec![zero; x_vectors[0].len()]; x_vectors.len()]` and then *pushed* to, so
every row carries `d` leading zeros ahead of its `d` real evaluations. The zeros
are harmless to correctness — they contribute nothing to the downstream dot
products — but they double the inner-product length at every compression level.
The rewrite preserves them exactly, because dropping them changes
`x_vectors[0].len()` at the next level and therefore the compression recursion
itself. Worth deciding deliberately rather than as part of a perf change.

### Also noted

`num_threads: usize` on the ACSS/SH2T contexts
(`secret_sharing/acss_ab/src/context.rs:54`, `secret_sharing/sh2t/src/context.rs:54`,
both defaulted to `4`) is dead — it is set but never read, and never used to
build a rayon pool. Either wire it up as the pool size or delete it.

---

## `init_` / `handle_` / `verify_` once-only semantics

**Status:** audited, nothing changed. Design sketch in
`docs/once-gate-design.md` (documentation only, not compiled).

Intended contract:

- `handle_*` — folds an inbound message into per-key state. Runs many times, in
  any order. Should stay a monotone accumulator: insert, extend, count. No
  protocol effects, no `init_*` calls.
- `verify_*` — a predicate over accumulated state. Re-entered on every message
  that could have completed a threshold.
- `init_*` — performs the phase (heavy compute + network effects). Must run
  **exactly once per key** however many times `verify_*` fires.

Rayon is orthogonal to all of this: once-ness is a property of the tokio task's
control flow, not of the data-parallel work inside `init_*`, which should keep
running at full width.

### Audit findings

**1. `init_rand_sh` has no guard at all.** Called from the `SyncState::START`
arm (`mpc/src/context.rs:537`), which is driven by a network message. A
duplicated or replayed START re-runs the entire preprocessing phase. This is the
clearest violation of the contract and the one reachable from outside.

**2. Two `init_` calls are made from `handle_`, not `verify_`** —
`init_hash_broadcast` at `mpc/src/protocol/multiplication/quad_mult.rs:121` and
`mpc/src/protocol/multiplication/lin_mult.rs:293`.

**3. Once-guards are stored in payload fields, and the memory-reclaim work is
taking those fields away.** This is the structural defect.

- `rand_sh.rs` `verify_termination` — **fixed**. Its once-ness used to live in
  `rand_sharings_mult.len() > 0`, the length of a data vector that is itself
  `split_off` later in the same function. It now uses a dedicated
  `RandSharings::combine_started` flag, set before the combine is dispatched.
  This was a prerequisite for yielding that site to rayon.
- `mpc/src/protocol/multiplication/weak_mult.rs:148` — `depth_terminated`, a
  real flag, is the correct shape by comparison.
- Commit `f6eaa7d` introduced `std::mem::take` on exactly this class of field
  (e.g. `weak_mult.rs:180-186`).

Guard state and payload state are conflated, so reclaiming payload can silently
reopen a gate that has already fired. Any fix must keep guard state in a
structure that holds *only* keys, never payload.

### On using locks for this

A mutex alone does not give once-only semantics: two callers serialized by a
lock would each still run the phase. The flag *inside* the lock is what makes it
once-only; the lock only makes that check atomic when several owners can reach
it. Today exactly one tokio task owns each `Context` (`mpc/src/context.rs:254`)
and `&mut self` is already exclusive access, so a plain
`HashSet::insert` -> `bool` is the atomic test-and-set, with no lock needed.

If handler state is ever partitioned across tasks and a gate becomes shared, the
critical section must stay confined to the test-and-set itself. Holding a lock
across the `.await` inside `init_*` stalls or deadlocks the runtime, and holding
one across a `par_iter` serializes exactly the rayon work that is supposed to
run at full width.

### Phase inventory

| `init_` | key | current guard |
|---|---|---|
| `init_rand_sh` | run-once | **none** |
| `init_random_shared_bits_preparation` | run-once | `rand_sharings_mult.len() > 0` (payload-derived) |
| `init_rand_bit_reconstruction` | run-once | `depth_terminated` |
| `init_quadratic_multiplication_prot` | depth | `depth_terminated` |
| `init_linear_multiplication_prot` | depth | `depth_terminated` |
| `init_hash_broadcast` | depth | called from `handle_`; none |
| `init_compression_level` | level | `verify_level_termination` |
| `init_ex_compression_tuples` | depth | `verify_ex_mult_termination_verification` |
