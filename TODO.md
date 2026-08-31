# TODO

## Rayon `par_iter` blocks a tokio worker inside the MPC `select!` loop

**Status:** open — analysis done, not yet fixed.

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

### Proposed fix

1. Wrap the heavy `par_iter` sites in `tokio::task::block_in_place` so the
   runtime migrates the remaining tasks off that worker and keeps its full
   worker count during compute. Cheapest change; requires the multi-thread
   runtime (already the case).
2. Alternatively move the compute behind `spawn_blocking` + a `oneshot`, which
   also lets the `select!` loop keep draining while the job runs. More
   invasive — the closures currently borrow `&mut self` state.
3. Consider bounding `net_recv`, or sizing the rayon pool below the core count,
   so compute stalls surface as backpressure rather than unbounded inbox growth.

Primary sites to convert: `quad_mult.rs:52`, `quad_mult.rs:102`,
`lin_mult.rs:214`, `lin_mult.rs:276`, `compress_tup.rs:158`,
`rand_sh.rs:108`/`:127`/`:320`, `rand_bit.rs:124`/`:194`, `rand_mask.rs:61`/`:121`.

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

- `mpc/src/protocol/rand_sharings/rand_sh.rs:293` —
  `if self.rand_sharings_state.rand_sharings_mult.len() > 0 { return; }`.
  The once-ness of `verify_termination` lives in the length of a data vector
  that is itself `split_off` at `:344`.
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
