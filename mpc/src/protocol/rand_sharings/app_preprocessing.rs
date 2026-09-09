//! Per-depth reservation of the multiplication preprocessing an application's
//! circuit consumes.
//!
//! Every party's pool holds the same sharings in the same order — the ACS agrees
//! on the dealer set and the Vandermonde extraction is deterministic — so what
//! has to match across parties is which *slice* a given circuit depth consumes.
//! Two parties masking the same gate with different `r` do not reconstruct.
//!
//! Handing out slices in the order batches happen to be scheduled makes that
//! binding a function of local timing. It survives only while every party runs
//! its depths in the same order, which rules out running depths out of order and
//! rules out fast-forwarding past a depth whose reconstruction has already
//! arrived. So the layout is computed up front instead: the application declares
//! how many gates sit at each depth
//! ([`PreprocessingCounts::gates_per_depth`](application::PreprocessingCounts::gates_per_depth)),
//! and the table below turns that into a fixed offset per depth. Depth `d` then
//! reads the same slice at every party, whenever it happens to run.
//!
//! Slices are read, not drained, so re-running a depth — a replayed termination,
//! a fast-forward that revisits one — consumes exactly the same material.

use application::PreprocessingCounts;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// Where one depth's preprocessing sits in the reserved buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthReservation {
    pub rand_offset: usize,
    pub rand_len: usize,
    pub zero_offset: usize,
    pub zero_len: usize,
}

/// The application's share of the multiplication preprocessing, laid out by depth.
pub struct ApplicationPreprocessing<F: ProtocolField> {
    /// One entry per depth, index `depth - 1`.
    table: Vec<DepthReservation>,
    /// Masks, `rand_total()` of them, reserved at pool construction.
    rand: Vec<FieldElement<F>>,
    /// 2t-sharings of zero, `zero_total()` of them.
    zero: Vec<FieldElement<F>>,
    /// `2t + 1` — the group the linear multiplication protocol pads batches to.
    group: usize,
    /// `t + 1` — zero sharings the linear protocol draws per group.
    zero_per_group: usize,
}

impl<F: ProtocolField> ApplicationPreprocessing<F> {
    /// An empty table, for before `init_rand_sh` has run.
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            rand: Vec::new(),
            zero: Vec::new(),
            group: 1,
            zero_per_group: 1,
        }
    }

    /// Lay out one slice per declared depth. Buffers stay empty until
    /// [`fill`](Self::fill); this only fixes the offsets, which is all the
    /// preprocessing sizing needs.
    ///
    /// Each depth is charged what the linear multiplication protocol charges a
    /// batch of that many gates: the gate count padded up to a whole number of
    /// groups of `2t+1`, one mask per padded gate, and `t+1` zero sharings per
    /// group. Padding a depth that declares no gates costs nothing.
    pub fn plan(counts: &PreprocessingCounts, num_faults: usize) -> Self {
        let group = 2 * num_faults + 1;
        let zero_per_group = num_faults + 1;

        let mut table = Vec::with_capacity(counts.depth());
        let (mut rand_offset, mut zero_offset) = (0usize, 0usize);
        for gates in counts.gates_per_depth.iter() {
            let groups = gates.div_ceil(group);
            let reservation = DepthReservation {
                rand_offset,
                rand_len: groups * group,
                zero_offset,
                zero_len: groups * zero_per_group,
            };
            rand_offset += reservation.rand_len;
            zero_offset += reservation.zero_len;
            table.push(reservation);
        }

        Self {
            table,
            rand: Vec::new(),
            zero: Vec::new(),
            group,
            zero_per_group,
        }
    }

    /// Masks the whole circuit reserves.
    pub fn rand_total(&self) -> usize {
        self.table.last().map_or(0, |last| last.rand_offset + last.rand_len)
    }

    /// Zero sharings the whole circuit reserves.
    pub fn zero_total(&self) -> usize {
        self.table.last().map_or(0, |last| last.zero_offset + last.zero_len)
    }

    /// Number of depths the application declared.
    pub fn num_depths(&self) -> usize {
        self.table.len()
    }

    /// True once the reserved buffers have been filled from the pool.
    pub fn is_filled(&self) -> bool {
        self.rand.len() >= self.rand_total() && self.zero.len() >= self.zero_total()
    }

    /// Hand the reserved buffers their contents, taken off the front of the
    /// preprocessing pool once it exists.
    pub fn fill(&mut self, rand: Vec<FieldElement<F>>, zero: Vec<FieldElement<F>>) {
        self.rand = rand;
        self.zero = zero;
    }

    /// The preprocessing depth `depth` consumes for a batch of `gates` gates,
    /// counting depths from 1.
    ///
    /// A depth may run fewer gates than it declared — it then takes a prefix of
    /// its slice, which is the same prefix at every party — but never more.
    pub fn for_depth(
        &self,
        depth: usize,
        gates: usize,
    ) -> Result<(Vec<FieldElement<F>>, Vec<FieldElement<F>>), String> {
        let Some(reservation) = depth.checked_sub(1).and_then(|index| self.table.get(index)) else {
            return Err(format!(
                "depth {} is outside the {} depths the application declared",
                depth,
                self.table.len()
            ));
        };

        let groups = gates.div_ceil(self.group);
        let (rand_len, zero_len) = (groups * self.group, groups * self.zero_per_group);
        if rand_len > reservation.rand_len {
            return Err(format!(
                "depth {} ran {} gates but declared room for {}",
                depth,
                gates,
                reservation.rand_len
            ));
        }

        let rand_end = reservation.rand_offset + rand_len;
        let zero_end = reservation.zero_offset + zero_len;
        if rand_end > self.rand.len() || zero_end > self.zero.len() {
            return Err(format!(
                "depth {} needs masks [{}..{}) and zero sharings [{}..{}), but only {} and {} were \
                 reserved; preprocessing fell short",
                depth,
                reservation.rand_offset,
                rand_end,
                reservation.zero_offset,
                zero_end,
                self.rand.len(),
                self.zero.len()
            ));
        }

        // Read, not drained: re-running a depth must consume the same material.
        Ok((
            self.rand[reservation.rand_offset..rand_end].to_vec(),
            self.zero[reservation.zero_offset..zero_end].to_vec(),
        ))
    }
}

impl<F: ProtocolField> Default for ApplicationPreprocessing<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = fields::DefaultField;

    const NUM_FAULTS: usize = 3;
    /// 2t + 1
    const GROUP: usize = 7;

    fn planned(gates_per_depth: Vec<usize>) -> ApplicationPreprocessing<F> {
        let counts = PreprocessingCounts::new(gates_per_depth, 0, 1);
        ApplicationPreprocessing::plan(&counts, NUM_FAULTS)
    }

    fn filled(gates_per_depth: Vec<usize>) -> ApplicationPreprocessing<F> {
        let mut plan = planned(gates_per_depth);
        // Distinct values, so a wrong offset shows up as a wrong value.
        let rand = (0..plan.rand_total() as u64).map(FieldElement::<F>::from).collect();
        let zero = (0..plan.zero_total() as u64).map(FieldElement::<F>::from).collect();
        plan.fill(rand, zero);
        plan
    }

    /// Each depth is charged whole groups of 2t+1, and the slices are laid end to
    /// end in depth order.
    #[test]
    fn depths_are_charged_whole_groups() {
        let plan = planned(vec![1, GROUP, GROUP + 1]);

        assert_eq!(plan.table[0].rand_offset, 0);
        assert_eq!(plan.table[0].rand_len, GROUP, "1 gate still burns a group");
        assert_eq!(plan.table[1].rand_offset, GROUP);
        assert_eq!(plan.table[1].rand_len, GROUP);
        assert_eq!(plan.table[2].rand_offset, 2 * GROUP);
        assert_eq!(plan.table[2].rand_len, 2 * GROUP, "one gate over a group takes two");

        assert_eq!(plan.rand_total(), 4 * GROUP);
        assert_eq!(plan.zero_total(), 4 * (NUM_FAULTS + 1));
    }

    /// The whole point: a depth's slice is a function of the depth, not of the
    /// order the depths are asked for. Reading them backwards must give the same
    /// material as reading them forwards.
    #[test]
    fn slices_do_not_depend_on_request_order() {
        let plan = filled(vec![2, 5, 3]);

        let forwards: Vec<_> = (1..=3).map(|d| plan.for_depth(d, 2).unwrap()).collect();
        let mut backwards: Vec<_> = (1..=3).rev().map(|d| plan.for_depth(d, 2).unwrap()).collect();
        backwards.reverse();

        assert_eq!(forwards, backwards);
        // And the three depths really do get different material.
        assert_ne!(forwards[0], forwards[1]);
        assert_ne!(forwards[1], forwards[2]);
    }

    /// Re-running a depth — a replayed termination, or a fast-forward that
    /// revisits one — must consume exactly the same material, so the slices are
    /// read rather than drained.
    #[test]
    fn rerunning_a_depth_reuses_its_slice() {
        let plan = filled(vec![4, 4]);

        assert_eq!(plan.for_depth(2, 4).unwrap(), plan.for_depth(2, 4).unwrap());
    }

    /// A depth that runs fewer gates than it declared takes a prefix of its own
    /// slice, and does not encroach on the next depth's.
    #[test]
    fn a_short_depth_takes_a_prefix_of_its_own_slice() {
        let plan = filled(vec![GROUP * 2, GROUP]);

        let (short_rand, _) = plan.for_depth(1, 1).unwrap();
        let (full_rand, _) = plan.for_depth(1, GROUP * 2).unwrap();
        assert_eq!(short_rand.len(), GROUP);
        assert_eq!(short_rand, full_rand[..GROUP]);

        let (second_rand, _) = plan.for_depth(2, GROUP).unwrap();
        assert_eq!(second_rand[0], FieldElement::<F>::from(2 * GROUP as u64));
    }

    #[test]
    fn overrunning_a_declared_depth_is_refused() {
        let plan = filled(vec![1]);

        let error = plan.for_depth(1, GROUP + 1).unwrap_err();
        assert!(error.contains("declared room for"), "got {:?}", error);
    }

    #[test]
    fn an_undeclared_depth_is_refused() {
        let plan = filled(vec![1, 1]);

        let error = plan.for_depth(3, 1).unwrap_err();
        assert!(error.contains("outside the 2 depths"), "got {:?}", error);
    }

    /// A purely linear circuit declares no depths and reserves nothing.
    #[test]
    fn a_circuit_with_no_depths_reserves_nothing() {
        let plan = planned(Vec::new());

        assert_eq!(plan.rand_total(), 0);
        assert_eq!(plan.zero_total(), 0);
        assert_eq!(plan.num_depths(), 0);
    }
}
