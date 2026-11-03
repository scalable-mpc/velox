//! Running rayon jobs from async code without parking the calling task.
//!
//! `par_iter` called directly from an `async fn` does not yield. When the
//! caller is not itself a rayon worker, the job goes through
//! `Registry::in_worker_cold`, which injects it and then parks the calling
//! thread on a latch. The future is never suspended, so the tokio worker
//! driving that task is unavailable until the whole job finishes and the task's
//! inbox keeps growing behind it.
//!
//! That matters most in the MPC engine, where the entire protocol state machine
//! is one task (`mpc/src/context.rs`): a long `par_iter` stops it from draining
//! `net_recv` at all. Handing the job to `rayon::spawn` and awaiting a oneshot
//! suspends the future instead, leaving the worker free to drive other tasks
//! while the job runs at full rayon width.
//!
//! See `TODO.md` and https://github.com/scalable-mpc/velox/issues/3.

/// Run `job` on rayon's pool and await its result without blocking the caller.
///
/// The closure runs on a rayon worker, so `par_iter` inside it nests naturally
/// and gets the whole pool. Because the job outlives the caller's stack frame it
/// must own its inputs — take them out of the surrounding state (`std::mem::take`
/// or a clone) before the call and write the results back after the `.await`.
///
/// A panic inside `job` is re-raised here, matching what an inline `par_iter`
/// would have done.
pub async fn rayon_async<T, J>(job: J) -> T
where
    J: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn(move || {
        // Send failure only means the awaiting task went away; nothing to do.
        let _ = tx.send(job());
    });
    rx.await
        .expect("rayon job panicked or its worker was dropped before producing a result")
}

/// Deserialise a flat share buffer (see [`ser_be_flat`](crate::ser_be_flat))
/// on rayon without parking the caller.
///
/// ACSS and Sh2t run this when a dealer's sharing terminates, before they hand
/// the shares to the MPC engine: `n` dealers times the number of batches. Each
/// buffer is cheap per element but large, and the caller is the module's
/// message loop, which must keep draining its inbox meanwhile.
pub async fn deserialize_shares_async<F: crate::ProtocolField>(
    flat_shares: Vec<u8>,
) -> Vec<lambdaworks_math::field::element::FieldElement<F>> {
    use rayon::prelude::{ParallelIterator, ParallelSlice};
    rayon_async(move || {
        flat_shares.par_chunks(F::SER_BYTES)
            .map(|x| F::from_bytes_be(x).unwrap())
            .collect()
    }).await
}
