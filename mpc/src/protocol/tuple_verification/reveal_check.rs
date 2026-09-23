//! The reveal check: alongside the multiplication tuple verification, every
//! party confirms that the values the application took to be revealed are
//! the ones the sharings actually held. It starts the moment the
//! delinearization coin is known — the same moment the tuple compression
//! starts — and the output is unmasked once both have passed, so it costs
//! no round of its own.
//!
//! A reveal's L2 step has no redundancy, so a corrupt party could shift a
//! revealed value `v_i` by some `δ_i` consistently at every honest party. The
//! check folds every reveal of the run into one sharing with the
//! delinearization coin `c` already agreed on,
//!
//! ```text
//! [Δ] = Σ_i c^i · ([v_i] − v_i)
//! ```
//!
//! and opens it: `Δ = 0` when every opening was honest, and a shift fixed
//! before the coin was known survives only if `Σ c^i δ_i = 0`, probability at
//! most `#reveals / |K|`. The coin, and so the fold, lies in the statistical
//! extension `K`, like the rest of verification. No second coin: `c` is
//! uniform and is tossed by a party only once its circuit — every reveal
//! included — has terminated.
//!
//! `[Δ]` is opened unmasked, one share per party, through the same degree-`t`
//! check the final verification tuple uses. That leaks nothing: each `[v_i]`
//! is a value blinded by a fresh random sharing (the reveal's contract), so
//! `[Δ]`'s polynomial is a combination of uniformly random polynomials that
//! is independent of every secret. A run without reveals skips the round.

use planner::api::engine::Application;
use fields::{LargeFieldSer, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

use crate::{msg::ProtMsg, Context};

use super::{StatisticalElement, deser_statistical, open_statistical_sharings, ser_statistical};

impl<F: ProtocolField, A: Application<F>> Context<F, A> {
    /// The delinearization coin is known and this party's circuit is done:
    /// fold the reveals and broadcast this party's share of `[Δ]`, or count
    /// the check as passed if there were none.
    pub async fn init_reveal_check(&mut self) {
        // Every reveal of the run, in ascending depth order so all parties
        // fold the same sequence. Taken: this is the records' last reader.
        let mut revealed: Vec<_> = std::mem::take(&mut self.verf_state.revealed).into_iter().collect();
        revealed.sort_by_key(|(depth, _)| *depth);
        let (mut sharings, mut values) = (Vec::new(), Vec::new());
        for (_, (mut s, mut v)) in revealed {
            sharings.append(&mut s);
            values.append(&mut v);
        }
        if sharings.is_empty() && values.is_empty() {
            log::info!("Reveal check: nothing was revealed, nothing to check");
            self.verf_state.reveals_verified = true;
            self.try_finish_verification().await;
            return;
        }
        if sharings.len() != values.len() {
            log::error!(
                "Reveal check: {} sharings were opened but {} values came back; abandoning the protocol",
                sharings.len(), values.len()
            );
            return;
        }
        let Some(coin) = self
            .verf_state
            .ex_compr_state
            .get(&self.delinearization_depth)
            .and_then(|state| state.coin_output.clone())
        else {
            log::error!("Reveal check: the delinearization coin is missing; abandoning the protocol");
            return;
        };
        let delta = Self::fold_reveals(&sharings, &values, &coin);
        log::info!("Reveal check: folded {} revealed values, broadcasting the share of Δ", sharings.len());
        self.verf_state.reveal_check_sent = true;
        self.broadcast(ProtMsg::RevealCheckShare(ser_statistical::<F>(&delta))).await;
        self.try_complete_reveal_check().await;
    }

    /// `Σ c^i (s_i − v_i)`, over `K`: this party's share of `[Δ]`, in the
    /// order the reveals were recorded.
    pub(crate) fn fold_reveals(
        sharings: &[FieldElement<F>],
        values: &[FieldElement<F>],
        coin: &StatisticalElement<F>,
    ) -> StatisticalElement<F> {
        let mut weight = StatisticalElement::<F>::one();
        let mut delta = StatisticalElement::<F>::zero();
        for (share, value) in sharings.iter().zip(values.iter()) {
            delta = delta + &weight * F::from_statistical_coeffs(&[share - value]);
            weight = weight * coin;
        }
        delta
    }

    pub async fn handle_reveal_check_share(&mut self, share: LargeFieldSer, sender: usize) {
        if self.verf_state.reveal_check_done {
            return;
        }
        let Some(share) = deser_statistical::<F>(&share) else {
            log::warn!("Reveal check: undecodable share from party {}", sender);
            return;
        };
        let point = Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone());
        let shares = &mut self.verf_state.reveal_check_shares;
        if shares.0.contains(&point) {
            return;
        }
        shares.0.push(point);
        shares.1.push(share);
        log::info!("Reveal check: share {} of {} from party {}", shares.0.len(), 2 * self.num_faults + 1, sender);
        self.try_complete_reveal_check().await;
    }

    /// Both guards are load-bearing: this is reached from `init_reveal_check`
    /// and from every share, in either order. Only once this party has folded
    /// its own reveals — its circuit done, the coin known — and `2t+1` shares
    /// are in does it open `[Δ]`.
    async fn try_complete_reveal_check(&mut self) {
        if !self.verf_state.reveal_check_sent || self.verf_state.reveal_check_done {
            return;
        }
        let needed = 2 * self.num_faults + 1;
        if self.verf_state.reveal_check_shares.0.len() < needed {
            return;
        }
        self.verf_state.reveal_check_done = true;
        let (points, shares) = std::mem::take(&mut self.verf_state.reveal_check_shares);
        let points = points[..needed].to_vec();
        let shares = shares[..needed].to_vec();
        let Some(opened) = open_statistical_sharings::<F>(points, &[shares], self.num_faults) else {
            log::error!("Reveal check failed: the shares of Δ do not lie on a degree-t polynomial; abandoning the protocol");
            return;
        };
        let delta = &opened[0];
        if *delta != StatisticalElement::<F>::zero() {
            log::error!("Reveal check failed: Δ = {:?}, a revealed value was shifted; abandoning the protocol", delta);
            return;
        }
        log::info!("Reveal check passed: every revealed value matches its sharing");
        self.verf_state.reveals_verified = true;
        self.try_finish_verification().await;
    }

    /// The tuple verification and the reveal check run side by side and end
    /// in either order; the output is unmasked once both have passed. Reached
    /// from the last step of each, so the flags decide, once.
    pub async fn try_finish_verification(&mut self) {
        if !self.verf_state.tuples_verified || !self.verf_state.reveals_verified || self.verf_state.verification_finished {
            return;
        }
        self.verf_state.verification_finished = true;
        log::info!("Multiplication tuples and revealed values verified, proceeding to the output");
        self.terminate("verification".to_string(), vec![]).await;
        self.reconstruct_output().await;
    }
}

#[cfg(test)]
mod tests {
    use fields::{Mersenne31Field, ProtocolField};
    use lambdaworks_math::field::element::FieldElement;

    use super::StatisticalElement;

    type F = Mersenne31Field;
    type E = FieldElement<F>;
    type K = StatisticalElement<F>;

    /// The fold on plaintext sharings (degree 0), with a coin in Fp2: honest
    /// reveals fold to zero, a shifted one does not, and the weights are the
    /// coin's powers in order.
    #[test]
    fn honest_reveals_fold_to_zero_and_a_shift_does_not() {
        let coin: K = F::from_statistical_coeffs(&[E::from(7u64), E::from(2u64)]);
        let embed = |v: E| F::from_statistical_coeffs(&[v]);
        let values: Vec<E> = (1..=5u64).map(E::from).collect();
        let fold = |s: &[E], v: &[E]| crate::Context::<F, planner::DefaultApplication<F>>::fold_reveals(s, v, &coin);
        assert_eq!(fold(&values, &values), K::zero());
        assert_eq!(fold(&[], &[]), K::zero());

        let mut shifted = values.clone();
        shifted[3] = &shifted[3] + E::one();
        assert_eq!(fold(&values, &shifted), -(&coin * &coin * &coin));

        let mut shifted_first = values.clone();
        shifted_first[0] = &shifted_first[0] + E::from(3u64);
        assert_eq!(fold(&shifted_first, &values), embed(E::from(3u64)));
    }
}
