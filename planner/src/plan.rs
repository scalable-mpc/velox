//! The plan: the application's declared op-depths compiled into engine
//! rounds.
//!
//! Every op-depth is one operation, and every operation has a fixed list of
//! steps (`ops::steps_of`), each one batch for the engine to run. The plan
//! places those steps at consecutive engine depths — one [`EngineRound`]
//! per step, sized to the declared width — and reads the engine's
//! preprocessing profile and random-wire needs straight off the result.
//! Computed once, before preprocessing, identically at every party from the
//! same declaration: that is what lets the engine reserve preprocessing per
//! depth in fixed slices.
//!
//! - [`EngineRound`] — one engine depth: an [`OpStep`] placed and sized.
//! - [`OpDepthPlan`] — one op-depth: its op's parameters and its rounds.
//! - [`Plan`] — every op-depth, end to end.

use anyhow::{bail, Result};

use crate::{
    api::{
        application::{OpParams, PlannerCounts},
        engine::{PreprocessingCounts, RandomWires},
    },
    ops::{self, EngineOperationType, OpStep},
};

/// One engine depth: a step of an op's schedule, placed at a depth and
/// sized to the op-depth's declared width. Becomes exactly one engine
/// batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineRound {
    pub step: OpStep,
    /// Engine depth, counting from 1.
    pub engine_depth: usize,
    /// Gates (or values opened) at the declared width.
    pub gates: usize,
}

/// One op-depth: the op's declared parameters, and the engine rounds it takes.
#[derive(Clone, Debug)]
pub struct OpDepthPlan {
    pub params: OpParams,
    pub rounds: Vec<EngineRound>,
}

impl OpDepthPlan {
    /// Engine rounds the op-depth takes.
    pub fn rounds(&self) -> usize {
        self.rounds.len()
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    op_depths: Vec<OpDepthPlan>,
    ell: usize,
    output: usize,
    app_rand_bits: usize,
    app_sharings: usize,
}

impl Plan {
    pub fn compile(counts: &PlannerCounts, ell: usize) -> Result<Self> {
        let mut op_depths = Vec::with_capacity(counts.depth());
        let mut engine_depth = 1;
        for (index, params) in counts.ops.iter().enumerate() {
            if params.elements == 0 {
                bail!("op-depth {} declares {:?} over no elements", index + 1, params.op_type);
            }
            let rounds = ops::steps_of(params.op_type, ell)
                .into_iter()
                .map(|step| {
                    let round = EngineRound { step, engine_depth, gates: step.num_engine_operands * params.elements };
                    engine_depth += 1;
                    round
                })
                .collect();
            op_depths.push(OpDepthPlan { params: *params, rounds });
        }
        Ok(Self { op_depths, ell, output: counts.output, app_rand_bits: counts.rand_bits, app_sharings: counts.sharings })
    }

    /// The plan of op-depth `op_depth`, counting from 1.
    pub fn op_depth(&self, op_depth: usize) -> Option<&OpDepthPlan> {
        op_depth.checked_sub(1).and_then(|i| self.op_depths.get(i))
    }

    pub fn op_depths(&self) -> &[OpDepthPlan] {
        &self.op_depths
    }

    pub fn ell(&self) -> usize {
        self.ell
    }

    /// Engine depths in all.
    pub fn engine_depths(&self) -> usize {
        self.op_depths.iter().map(|d| d.rounds.len()).sum()
    }

    /// edaBits each op-depth consumes, op-depth 1 first.
    pub fn edabits_per_depth(&self) -> Vec<usize> {
        self.op_depths.iter().map(|d| d.params.edabits()).collect()
    }

    pub fn edabits_total(&self) -> usize {
        self.edabits_per_depth().iter().sum()
    }

    /// Bits the Planner keeps for itself off the front of what the engine
    /// delivers: `ℓ` per edaBit.
    pub fn planner_bits(&self) -> usize {
        self.edabits_total() * self.ell
    }

    /// The engine's profile: plain and masked gates per engine depth, 0 at
    /// a reveal depth.
    pub fn preprocessing_counts(&self) -> PreprocessingCounts {
        let mut gates = Vec::with_capacity(self.engine_depths());
        let mut masked = Vec::with_capacity(self.engine_depths());
        for round in self.op_depths.iter().flat_map(|d| d.rounds.iter()) {
            let (plain, mask) = match round.step.engine_op {
                EngineOperationType::Multiply => (round.gates, 0),
                EngineOperationType::MaskedMultiply => (0, round.gates),
                EngineOperationType::Reveal => (0, 0),
            };
            gates.push(plain);
            masked.push(mask);
        }
        PreprocessingCounts::new(gates, self.output).with_masked_gates(masked)
    }

    /// The engine's random wires: the Planner's bits on top of the
    /// application's own.
    pub fn random_wires(&self) -> RandomWires {
        RandomWires::new(self.planner_bits() + self.app_rand_bits, self.app_sharings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::application::OpType;

    fn rounds(op_type: OpType, ell: usize) -> usize {
        Plan::compile(&PlannerCounts::new(vec![OpParams::new(op_type, 3)], 0), ell).unwrap().op_depth(1).unwrap().rounds()
    }

    #[test]
    fn rounds_per_type() {
        assert_eq!(rounds(OpType::Add, 61), 0);
        for op_type in [OpType::Mul, OpType::Reveal, OpType::MaskReveal, OpType::Truncate, OpType::FixedMul] {
            assert_eq!(rounds(op_type, 61), 1, "{op_type:?}");
        }
        for op_type in [OpType::Compare, OpType::ComparePub] {
            assert_eq!(rounds(op_type, 61), 8, "{op_type:?}");
            assert_eq!(rounds(op_type, 31), 7, "{op_type:?}");
        }
        for op_type in [OpType::Max, OpType::Min, OpType::MaxPub, OpType::MinPub] {
            assert_eq!(rounds(op_type, 61), 9, "{op_type:?}");
            assert_eq!(rounds(op_type, 31), 8, "{op_type:?}");
        }
    }

    #[test]
    fn a_comparison_depth_is_planned_level_by_level() {
        let plan = Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Max, 4)], 0), 61).unwrap();
        let op_depth = plan.op_depth(1).unwrap();
        let batches: Vec<EngineOperationType> = op_depth.rounds.iter().map(|r| r.step.engine_op).collect();
        assert_eq!(batches[0], EngineOperationType::Reveal);
        assert!(batches[1..].iter().all(|b| *b == EngineOperationType::Multiply));
        let gates: Vec<usize> = op_depth.rounds.iter().map(|r| r.gates).collect();
        assert_eq!(gates, vec![4, 30 * 4, 29 * 4, 15 * 4, 7 * 4, 3 * 4, 4, 4, 4]);
        let depths: Vec<usize> = op_depth.rounds.iter().map(|r| r.engine_depth).collect();
        assert_eq!(depths, (1..=9).collect::<Vec<_>>());
        let counts = plan.preprocessing_counts();
        assert_eq!(counts.gates_per_depth, vec![0, 120, 116, 60, 28, 12, 4, 4, 4]);
        assert_eq!(counts.masked_gates_per_depth, vec![0; 9]);
        assert_eq!(plan.random_wires(), RandomWires::new(4 * 61, 0));
    }

    #[test]
    fn op_depths_are_laid_end_to_end_and_app_wires_are_added() {
        let counts = PlannerCounts::new(
            vec![
                OpParams::new(OpType::Mul, 4),
                OpParams::new(OpType::Add, 1),
                OpParams::new(OpType::Truncate, 2),
                OpParams::new(OpType::FixedMul, 3),
            ],
            9,
        )
        .with_random_wires(5, 3);
        let plan = Plan::compile(&counts, 61).unwrap();
        assert_eq!(plan.op_depth(1).unwrap().rounds[0].engine_depth, 1);
        assert_eq!(plan.op_depth(2).unwrap().rounds(), 0);
        assert_eq!(plan.op_depth(3).unwrap().rounds[0].engine_depth, 2);
        assert_eq!(plan.op_depth(4).unwrap().rounds[0].engine_depth, 3);
        assert_eq!(plan.engine_depths(), 3);
        let pc = plan.preprocessing_counts();
        assert_eq!(pc.gates_per_depth, vec![4, 0, 0]);
        assert_eq!(pc.masked_gates_per_depth, vec![0, 0, 3]);
        assert_eq!(pc.output, 9);
        assert_eq!(plan.edabits_per_depth(), vec![0, 0, 2, 3]);
        assert_eq!(plan.random_wires(), RandomWires::new(5 * 61 + 5, 3));
        assert!(plan.op_depth(5).is_none());
        assert!(Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Mul, 0)], 0), 61).is_err());
    }
}
