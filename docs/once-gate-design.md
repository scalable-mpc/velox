# Design note: structural once-only semantics for `init_*`

**Status: design only — not implemented, not wired into the crate.**

This is a reference sketch kept as documentation. It is deliberately *not* under
`mpc/src/`, so nothing here is compiled. See `TODO.md` for the surrounding
concurrency analysis and the audit findings that motivate it.

## Sketch

```rust
//! Structural once-only semantics for the protocol's `init_*` entry points.
//!
//! # Why this exists
//!
//! The engine's functions fall into three roles:
//!
//! - `handle_*` — fold an inbound message into per-key state. Runs many times,
//!   in any order, and must stay a *monotone accumulator*: insert, extend,
//!   count. No protocol effects, no `init_*` calls.
//! - `verify_*` — a predicate over accumulated state. Runs on every message
//!   that could have completed a threshold, so it is re-entered constantly.
//!   Decides whether a phase is ready to start.
//! - `init_*` — performs the phase: heavy compute plus network effects.
//!   Must run **exactly once per key**, however many times `verify_*` fires.
//!
//! Historically that last invariant was enforced by ad-hoc sentinels read out
//! of *payload* fields — `if rand_sharings_mult.len() > 0 { return }`,
//! `if mult_state.depth_terminated { return }`. Conflating guard state with
//! payload state is fragile in general, and actively unsound here: the memory
//! reclamation work (`std::mem::take` on per-depth buffers once they are last
//! read) can empty the very field a guard is testing, silently reopening a gate
//! that has already fired.
//!
//! [`OnceGate`] separates the two. It holds *only* keys — never payload — so no
//! amount of payload reclamation can reopen it. And because [`InitToken`] has
//! no public constructor and is not `Clone`, an `init_*` that takes one by
//! value cannot be called twice without minting a second token, which the gate
//! will not do. Double-init becomes a type error rather than a runtime check a
//! later refactor can quietly drop.
//!
//! # Relationship to rayon
//!
//! None, by construction. A claim is a single `HashSet` insert that completes
//! *before* the `init_*` body runs; no guard is held across the call. `init_*`
//! bodies keep using `par_iter` at full width exactly as before. The once-only
//! semantics are a property of the tokio task's control flow, not of the
//! data-parallel work underneath it.

use std::collections::HashSet;
use std::hash::Hash;
use std::marker::PhantomData;

/// One implementation per `init_*` entry point.
///
/// `Key` is what the phase is keyed by: `()` for phases that run once per
/// protocol run, `usize` for phases that run once per depth or level.
pub trait Phase: 'static {
    type Key: Eq + Hash + Clone + std::fmt::Debug;
    /// Used only in log lines when a duplicate claim is rejected.
    const NAME: &'static str;
}

/// Proof that a phase has been claimed for a particular key, and that this is
/// the only such proof in existence.
///
/// Deliberately not `Clone`, not `Copy`, and with a private field so it cannot
/// be constructed outside this module. The sole source is [`OnceGate::claim`].
/// An `init_*` that accepts one **by value** is therefore callable only as many
/// times as the gate has handed out tokens — that is, once per key.
#[must_use = "an InitToken authorises exactly one init_ call; dropping it consumes \
              the claim without running the phase, and the phase can never run again"]
pub struct InitToken<P: Phase> {
    key: P::Key,
    /// `fn() -> P` keeps the token covariant in `P` without implying it owns a
    /// `P`, so `Phase` markers need not be inhabited types.
    _phase: PhantomData<fn() -> P>,
}

impl<P: Phase> InitToken<P> {
    /// The key this token authorises. Lets an `init_*` recover its own depth
    /// or level from the token instead of taking it as a second, unchecked
    /// parameter that could disagree with the claim.
    pub fn key(&self) -> &P::Key {
        &self.key
    }
}

impl<P: Phase> std::fmt::Debug for InitToken<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InitToken<{}>({:?})", P::NAME, self.key)
    }
}

/// Records which keys of a phase have been claimed.
///
/// Holds guard state and nothing else. Never store payload here, and never
/// derive a claim from payload — that is the coupling this type exists to break.
pub struct OnceGate<P: Phase> {
    claimed: HashSet<P::Key>,
    _phase: PhantomData<fn() -> P>,
}

impl<P: Phase> OnceGate<P> {
    pub fn new() -> Self {
        Self { claimed: HashSet::new(), _phase: PhantomData }
    }

    /// Yields `Some(token)` the first time `key` is claimed and `None` on every
    /// later attempt.
    ///
    /// The returned token borrows nothing, so the `&mut self` borrow taken here
    /// ends at the call boundary. The idiomatic call site
    ///
    /// ```ignore
    /// if let Some(tok) = self.gates.rand_sh.claim(()) {
    ///     self.init_rand_sh(tok).await;
    /// }
    /// ```
    ///
    /// borrow-checks without any restructuring.
    pub fn claim(&mut self, key: P::Key) -> Option<InitToken<P>> {
        if self.claimed.insert(key.clone()) {
            Some(InitToken { key, _phase: PhantomData })
        } else {
            log::debug!("phase {} already initialised for key {:?}; skipping", P::NAME, key);
            None
        }
    }

    /// Whether `key` has been claimed. For diagnostics and for `verify_*`
    /// predicates that want to bail early; it is *not* a substitute for
    /// `claim`, which is the only operation that actually reserves the key.
    pub fn is_claimed(&self, key: &P::Key) -> bool {
        self.claimed.contains(key)
    }

    /// Number of keys claimed so far. Diagnostics only.
    pub fn claimed_count(&self) -> usize {
        self.claimed.len()
    }
}

impl<P: Phase> Default for OnceGate<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-owner variant of [`OnceGate`], for when `handle_*` state is eventually
/// partitioned across tasks and more than one owner can reach the same gate.
///
/// Semantics are identical — exactly one caller receives the token, everyone
/// else gets `None` — but the test-and-set is protected by a mutex so racing
/// tasks cannot both win.
///
/// # The invariant that keeps this safe
///
/// The lock is acquired and released **entirely inside `claim`**. The guard is
/// dropped before the function returns, so it is impossible for a caller to
/// hold it across an `.await` or across a `par_iter`. That is deliberate and
/// it is why `claim` returns a token rather than a guard: handing out a guard
/// would let a caller keep the lock for the duration of `init_*`, which would
/// both stall the runtime on the `.await` inside `init_*` and serialize the
/// rayon work underneath it.
///
/// Note that mutual exclusion alone would *not* give once-only semantics — two
/// callers serialized by a mutex would each still run the phase. The `insert`
/// returning `true` for exactly one caller is what makes it once-only; the lock
/// only ensures that check is atomic when there are several owners.
///
/// Deliberately uses `std::sync::Mutex`, not `tokio::sync::Mutex`: the critical
/// section is a single hash insert with no `.await` in it, so an async mutex
/// would add scheduling overhead for no benefit. Poisoning is recovered from
/// rather than propagated — a panic in a peer task must not make the gate
/// permanently unusable.
pub struct SharedOnceGate<P: Phase> {
    claimed: std::sync::Mutex<HashSet<P::Key>>,
    _phase: PhantomData<fn() -> P>,
}

impl<P: Phase> SharedOnceGate<P> {
    pub fn new() -> Self {
        Self { claimed: std::sync::Mutex::new(HashSet::new()), _phase: PhantomData }
    }

    /// Yields `Some(token)` to exactly one caller per key, across all threads.
    ///
    /// Takes `&self`, so this gate can sit behind an `Arc` and be shared by
    /// several owners without any of them needing `&mut`.
    pub fn claim(&self, key: P::Key) -> Option<InitToken<P>> {
        // Scoped so the guard is provably dropped before returning; nothing
        // that follows can await or fan out while holding it.
        let won = {
            let mut claimed = self.claimed.lock().unwrap_or_else(|e| e.into_inner());
            claimed.insert(key.clone())
        };
        if won {
            Some(InitToken { key, _phase: PhantomData })
        } else {
            log::debug!("phase {} already initialised for key {:?}; skipping", P::NAME, key);
            None
        }
    }

    pub fn is_claimed(&self, key: &P::Key) -> bool {
        self.claimed.lock().unwrap_or_else(|e| e.into_inner()).contains(key)
    }
}

impl<P: Phase> Default for SharedOnceGate<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Declares a phase marker plus its key type in one line.
macro_rules! declare_phases {
    ($($(#[$m:meta])* $name:ident => $key:ty),* $(,)?) => {
        $(
            $(#[$m])*
            pub struct $name;
            impl Phase for $name {
                type Key = $key;
                const NAME: &'static str = stringify!($name);
            }
        )*
    };
}

declare_phases! {
    /// `init_rand_sh` — preprocessing kickoff. Keyed by `()`: once per run.
    /// Reached from the `SyncState::START` network arm, so without a gate a
    /// duplicated or replayed START re-runs all of preprocessing.
    RandShPhase => (),
    /// `init_random_shared_bits_preparation` — once per run. Previously guarded
    /// by `rand_sharings_mult.len() > 0`, a payload-derived sentinel.
    RandBitPrepPhase => (),
    /// `init_rand_bit_reconstruction` — once per run.
    RandBitReconPhase => (),
    /// `init_quadratic_multiplication_prot` / `init_linear_multiplication_prot`
    /// — once per circuit depth. The two are mutually exclusive alternatives
    /// chosen by `choose_multiplication_protocol`, so they share one gate:
    /// claiming a depth for either bars the other.
    MultDepthPhase => usize,
    /// `init_hash_broadcast` — once per depth. Currently invoked from two
    /// `handle_*` paths; the gate makes the duplicate harmless while those call
    /// sites are moved behind `verify_*`.
    HashBroadcastPhase => usize,
    /// `init_compression_level` — once per verification level.
    CompressionLevelPhase => usize,
    /// `init_ex_compression_tuples` — once per verification depth.
    ExCompressionPhase => usize,
}

/// Every gate the engine owns, in one place.
///
/// Lives as a single `Context` field so the complete set of once-only phases is
/// auditable at a glance, and so a future refactor can hand `handle_*` a
/// borrow-split view of `Context` that simply does not include this struct —
/// making "handlers cannot start phases" structural too.
pub struct PhaseGates {
    pub rand_sh: OnceGate<RandShPhase>,
    pub rand_bit_prep: OnceGate<RandBitPrepPhase>,
    pub rand_bit_recon: OnceGate<RandBitReconPhase>,
    pub mult_depth: OnceGate<MultDepthPhase>,
    pub hash_broadcast: OnceGate<HashBroadcastPhase>,
    pub compression_level: OnceGate<CompressionLevelPhase>,
    pub ex_compression: OnceGate<ExCompressionPhase>,
}

impl PhaseGates {
    pub fn new() -> Self {
        Self {
            rand_sh: OnceGate::new(),
            rand_bit_prep: OnceGate::new(),
            rand_bit_recon: OnceGate::new(),
            mult_depth: OnceGate::new(),
            hash_broadcast: OnceGate::new(),
            compression_level: OnceGate::new(),
            ex_compression: OnceGate::new(),
        }
    }
}

impl Default for PhaseGates {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_once_phase_claims_exactly_once() {
        let mut gate: OnceGate<RandShPhase> = OnceGate::new();
        assert!(gate.claim(()).is_some(), "first claim must succeed");
        assert!(gate.claim(()).is_none(), "second claim must be refused");
        assert!(gate.claim(()).is_none(), "and stay refused");
    }

    #[test]
    fn keyed_phase_claims_once_per_key() {
        let mut gate: OnceGate<MultDepthPhase> = OnceGate::new();
        assert!(gate.claim(3).is_some());
        assert!(gate.claim(4).is_some(), "a different depth is independent");
        assert!(gate.claim(3).is_none(), "the same depth is refused");
        assert_eq!(gate.claimed_count(), 2);
    }

    #[test]
    fn token_carries_its_key() {
        let mut gate: OnceGate<CompressionLevelPhase> = OnceGate::new();
        let tok = gate.claim(7).expect("first claim");
        assert_eq!(*tok.key(), 7);
    }

    #[test]
    fn is_claimed_tracks_claim() {
        let mut gate: OnceGate<ExCompressionPhase> = OnceGate::new();
        assert!(!gate.is_claimed(&1));
        let _tok = gate.claim(1).expect("first claim");
        assert!(gate.is_claimed(&1));
    }

    /// The guarantee that matters: a gate holds keys only, so emptying protocol
    /// payload — which the memory-reclaim work does via `std::mem::take` — can
    /// never reopen a phase. This is the failure mode payload-derived sentinels
    /// such as `rand_sharings_mult.len() > 0` are exposed to.
    #[test]
    fn claim_survives_payload_reclamation() {
        let mut gate: OnceGate<RandBitPrepPhase> = OnceGate::new();
        let mut payload: Vec<u64> = vec![1, 2, 3];

        assert!(gate.claim(()).is_some());
        let _reclaimed = std::mem::take(&mut payload);
        assert!(payload.is_empty(), "payload reclaimed, as f6eaa7d does");

        assert!(gate.claim(()).is_none(), "gate must stay closed after reclamation");
    }
}
```
