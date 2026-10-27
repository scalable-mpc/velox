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
        application::{OpParams, OpType, PlannerCounts},
        engine::{PreprocessingCounts, RandomWires},
    },
    ops::{
        self, mod2m,
        uniform_mask::{self, MaskShape},
        EngineOperationType, OpStep,
    },
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
    /// The uniform-mask prefix, at engine depths `1..`: empty over a
    /// Mersenne field and when no op-depth uses edaBits.
    prefix: Vec<EngineRound>,
    op_depths: Vec<OpDepthPlan>,
    /// The bits of `p` over the Mersenne prime field `2^p_bits − 1`, `None`
    /// over any other protocol field.
    p_bits: Option<usize>,
    /// Bits of one edaBit: `p_bits` over a Mersenne field, the mask width over
    /// a prime `2^p_bits_0 − 2^p_bits_1 + 1`, `None` when the field has no
    /// uniform masks.
    mask_bits: Option<usize>,
    /// Over a prime `2^p_bits_0 − 2^p_bits_1 + 1` that is not Mersenne, when
    /// an op-depth uses edaBits: the shape of the prime and the candidates
    /// the prefix tests.
    mask_test: Option<(MaskShape, usize)>,
    output: usize,
    app_rand_bits: usize,
    app_sharings: usize,
}

impl Plan {
    /// `p_bits` is `F::MERSENNE_BITS` and `prime` is `F::PRIME_MODULUS`: a
    /// declaration with an op the field cannot run is refused here, before
    /// preprocessing.
    pub fn compile(counts: &PlannerCounts, p_bits: Option<usize>, prime: Option<u128>) -> Result<Self> {
        let shape = if p_bits.is_none() { prime.and_then(MaskShape::of_prime) } else { None };
        let mask_bits = p_bits.or(shape.map(|s| s.p_bits_0));
        let masks: usize = counts.ops.iter().map(|p| p.edabits()).sum();
        let mask_test = shape.filter(|_| masks > 0).map(|shape| (shape, shape.candidates_for(masks)));
        let mut engine_depth = 1;
        let mut place = |step: OpStep, elements: usize| {
            let round = EngineRound { step, engine_depth, gates: step.num_engine_operands * elements };
            engine_depth += 1;
            round
        };
        let prefix = match mask_test {
            Some((shape, candidates)) => uniform_mask::steps(shape).into_iter().map(|step| place(step, candidates)).collect(),
            None => Vec::new(),
        };
        let mut op_depths = Vec::with_capacity(counts.depth());
        for (index, params) in counts.ops.iter().enumerate() {
            if params.elements == 0 {
                bail!("op-depth {} declares {:?} over no elements", index + 1, params.op_type);
            }
            if let Err(e) = params.op_type.check_field(p_bits, mask_bits.is_some()) {
                bail!("op-depth {}: {}", index + 1, e);
            }
            if let (OpType::Mod2m { m }, Some(p_bits)) = (params.op_type, p_bits) {
                if let Err(e) = mod2m::check_m(m, p_bits) {
                    bail!("op-depth {}: {}", index + 1, e);
                }
            }
            // `m < mask_bits` also keeps the op's slices of the mask bits in
            // range.
            if let (OpType::CanonicalMod2m { m }, Some(bits)) = (params.op_type, mask_bits) {
                if m == 0 || m >= bits {
                    bail!("op-depth {}: CanonicalMod2m by {} bits is outside 1..={} for {}-bit masks", index + 1, m, bits - 1, bits);
                }
            }
            let rounds = ops::steps_of(params.op_type, mask_bits).into_iter().map(|step| place(step, params.elements)).collect();
            op_depths.push(OpDepthPlan { params: *params, rounds });
        }
        Ok(Self {
            prefix,
            op_depths,
            p_bits,
            mask_bits,
            mask_test,
            output: counts.output,
            app_rand_bits: counts.rand_bits,
            app_sharings: counts.sharings,
        })
    }

    /// The uniform-mask prefix's rounds, run before op-depth 1.
    pub fn prefix(&self) -> &[EngineRound] {
        &self.prefix
    }

    /// The prefix's shape and the number of candidates it tests, when the
    /// plan has a prefix.
    pub fn mask_test(&self) -> Option<(MaskShape, usize)> {
        self.mask_test
    }

    /// Bits of one edaBit, when the field has uniform masks.
    pub fn mask_bits(&self) -> Option<usize> {
        self.mask_bits
    }

    /// The plan of op-depth `op_depth`, counting from 1.
    pub fn op_depth(&self, op_depth: usize) -> Option<&OpDepthPlan> {
        op_depth.checked_sub(1).and_then(|i| self.op_depths.get(i))
    }

    pub fn op_depths(&self) -> &[OpDepthPlan] {
        &self.op_depths
    }

    pub fn p_bits(&self) -> Option<usize> {
        self.p_bits
    }

    /// Engine depths in all, the prefix's included.
    pub fn engine_depths(&self) -> usize {
        self.prefix.len() + self.op_depths.iter().map(|d| d.rounds.len()).sum::<usize>()
    }

    /// edaBits each op-depth consumes, op-depth 1 first.
    pub fn edabits_per_depth(&self) -> Vec<usize> {
        self.op_depths.iter().map(|d| d.params.edabits()).collect()
    }

    pub fn edabits_total(&self) -> usize {
        self.edabits_per_depth().iter().sum()
    }

    /// Bits the Planner keeps for itself off the front of what the engine
    /// delivers: the mask width per edaBit made. With a prefix, the Planner
    /// makes one edaBit per candidate, not one per edaBit used.
    pub fn planner_bits(&self) -> usize {
        let edabits_made = self.mask_test.map_or(self.edabits_total(), |(_, candidates)| candidates);
        edabits_made * self.mask_bits.unwrap_or(0)
    }

    /// The engine's profile: plain and masked gates per engine depth, 0 at
    /// a reveal depth.
    pub fn preprocessing_counts(&self) -> PreprocessingCounts {
        let mut gates = Vec::with_capacity(self.engine_depths());
        let mut masked = Vec::with_capacity(self.engine_depths());
        for round in self.prefix.iter().chain(self.op_depths.iter().flat_map(|d| d.rounds.iter())) {
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

    fn rounds(op_type: OpType, p_bits: usize) -> usize {
        Plan::compile(&PlannerCounts::new(vec![OpParams::new(op_type, 3)], 0), Some(p_bits), None).unwrap().op_depth(1).unwrap().rounds()
    }

    /// Over a non-Mersenne field the plan takes `Mul`, `Add` and `Reveal`,
    /// asks for no random bits of its own, and refuses the rest by name.
    #[test]
    fn a_non_mersenne_field_plans_only_the_field_generic_ops() {
        let counts = PlannerCounts::new(
            vec![OpParams::new(OpType::Mul, 4), OpParams::new(OpType::Add, 2), OpParams::new(OpType::Reveal, 3)],
            1,
        )
        .with_random_wires(7, 2);
        let plan = Plan::compile(&counts, None, None).unwrap();
        assert_eq!(plan.engine_depths(), 2);
        assert_eq!(plan.planner_bits(), 0);
        assert_eq!(plan.random_wires(), RandomWires::new(7, 2));
        for op_type in [OpType::Compare, OpType::MaxPub, OpType::Truncate, OpType::FixedMul, OpType::MaskReveal] {
            let err = Plan::compile(&PlannerCounts::new(vec![OpParams::new(op_type, 1)], 0), None, None).unwrap_err();
            assert!(err.to_string().contains("Mersenne"), "{op_type:?}: {err}");
        }
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
        // A reveal, then ⌈log₂ m⌉ tree levels.
        assert_eq!(rounds(OpType::Mod2m { m: 1 }, 127), 1);
        assert_eq!(rounds(OpType::Mod2m { m: 8 }, 127), 4);
        assert_eq!(rounds(OpType::Mod2m { m: 63 }, 127), 7);
    }

    /// A `Mod2m` whose `m` the field cannot mask is refused before
    /// preprocessing, naming the op-depth.
    #[test]
    fn mod2m_out_of_range_is_refused_at_compile() {
        let compile = |m, p_bits| Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Mod2m { m }, 1)], 0), Some(p_bits), None);
        assert!(compile(20, 61).is_ok());
        for (m, p_bits) in [(0, 61), (21, 61), (1, 31), (87, 127)] {
            let err = compile(m, p_bits).unwrap_err();
            assert!(err.to_string().starts_with("op-depth 1: Mod2m"), "m={m} p_bits={p_bits}: {err}");
        }
        let plan = compile(63, 127).unwrap();
        assert_eq!(plan.planner_bits(), 127, "one edaBit");
    }

    /// `CanonicalMod2m` takes `1 ≤ m < mask_bits` for the edaBit width
    /// `mask_bits`: 23 over the ML-DSA field, 61 over M61.
    #[test]
    fn canonical_mod2m_out_of_range_is_refused_at_compile() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let compile = |m, p_bits, prime| Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::CanonicalMod2m { m }, 1)], 0), p_bits, prime);
        assert_eq!(compile(4, None, Some(q)).unwrap().op_depth(1).unwrap().rounds(), 7);
        assert!(compile(22, None, Some(q)).is_ok());
        assert!(compile(60, Some(61), None).is_ok());
        for (m, p_bits, prime) in [(0, None, Some(q)), (23, None, Some(q)), (61, Some(61), None)] {
            let err = compile(m, p_bits, prime).unwrap_err();
            assert!(err.to_string().starts_with("op-depth 1: CanonicalMod2m"), "m={m}: {err}");
        }
        let err = compile(4, None, Some(101)).unwrap_err();
        assert!(err.to_string().contains("uniform masks"), "{err}");
    }

    #[test]
    fn a_comparison_depth_is_planned_level_by_level() {
        let plan = Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Max, 4)], 0), Some(61), None).unwrap();
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

    /// Over the ML-DSA field, `q = 2^23 − 2^13 + 1`, edaBits come from the
    /// uniform-mask prefix: its six rounds take engine depths 1..=6, sized to
    /// the candidates, and the op-depths follow.
    #[test]
    fn a_prime_with_uniform_masks_plans_the_prefix_first() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let counts = PlannerCounts::new(vec![OpParams::new(OpType::Mul, 2), OpParams::new(OpType::MaskReveal, 100)], 0)
            .with_random_wires(3, 0);
        let plan = Plan::compile(&counts, None, Some(q)).unwrap();
        let (shape, candidates) = plan.mask_test().unwrap();
        assert_eq!((shape.p_bits_0, shape.p_bits_1, candidates), (23, 13, 100 + 40 + 1));
        assert_eq!(plan.prefix().iter().map(|r| r.engine_depth).collect::<Vec<_>>(), (1..=6).collect::<Vec<_>>());
        assert_eq!(plan.prefix().iter().map(|r| r.gates).collect::<Vec<_>>(), vec![141, 282, 141, 141, 141, 141]);
        assert_eq!(plan.op_depth(1).unwrap().rounds[0].engine_depth, 7);
        assert_eq!(plan.op_depth(2).unwrap().rounds[0].engine_depth, 8);
        assert_eq!(plan.engine_depths(), 8);
        assert_eq!(plan.preprocessing_counts().gates_per_depth, vec![141, 282, 141, 141, 141, 0, 2, 0]);
        assert_eq!(plan.random_wires(), RandomWires::new(141 * 23 + 3, 0));
        // No prefix when nothing uses edaBits, and the Mersenne-only ops are
        // still refused.
        let plain = Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Mul, 2)], 0), None, Some(q)).unwrap();
        assert!(plain.prefix().is_empty() && plain.planner_bits() == 0);
        let err = Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Compare, 1)], 0), None, Some(q)).unwrap_err();
        assert!(err.to_string().contains("Mersenne"), "{err}");
        // A prime of another shape has no uniform masks.
        let err = Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::MaskReveal, 1)], 0), None, Some(101)).unwrap_err();
        assert!(err.to_string().contains("uniform masks"), "{err}");
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
        let plan = Plan::compile(&counts, Some(61), None).unwrap();
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
        assert!(Plan::compile(&PlannerCounts::new(vec![OpParams::new(OpType::Mul, 0)], 0), Some(61), None).is_err());
    }
}
