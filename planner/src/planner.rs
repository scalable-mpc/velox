//! The core: one op-depth at a time, one engine step at a time.
//!
//! The Planner knows nothing of the engine's hook signatures (that is
//! `api::engine`) or of any operation's arithmetic (that is `ops`). For each
//! op-depth the application schedules it does, in order:
//!
//! 1. [`schedule`](Planner::schedule): check the op against its declared parameters,
//!    draw the op-depth's edaBits, build the [`Operation`].
//! 2. [`next_batch`](Planner::next_batch): the op's operands for its next
//!    step, tagged with that round's engine depth.
//! 3. [`deliver`](Planner::deliver): the step's results back to the op.
//! 4. [`complete`](Planner::complete): the op's result for the application,
//!    once its steps are exhausted.

use anyhow::{bail, Result};
use fields::{MersennePrimeField, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

use crate::{
    api::application::{Op, OpResult, PlannerApplication, PlannerCounts},
    plan::Plan,
    ops::{self, EngineOperands, Operation},
    primitives::edabit::EdaBitPool,
};

/// One engine step's batch.
pub struct Batch<F: ProtocolField> {
    pub engine_depth: usize,
    pub operands: EngineOperands<F>,
}

/// The op-depth in flight.
struct DepthRun<F: ProtocolField + MersennePrimeField> {
    op_depth: usize,
    op: Box<dyn Operation<F>>,
    /// The next of the op's steps.
    step: usize,
    /// The step whose batch is out: its engine depth and result count.
    out: Option<(usize, usize)>,
}

pub struct Planner<F: ProtocolField + MersennePrimeField, A: PlannerApplication<F>> {
    pub(crate) app: A,
    plan: Plan,
    pool: EdaBitPool<F>,
    run: Option<DepthRun<F>>,
    /// Highest op-depth completed; op-depths run in order.
    completed: usize,
}

impl<F: ProtocolField + MersennePrimeField, A: PlannerApplication<F>> Planner<F, A> {
    /// Compile the application's declaration into the engine's plan.
    pub fn new(app: A) -> Result<Self> {
        let counts: PlannerCounts = app.preprocessing_count();
        let plan = Plan::compile(&counts, F::BITS)?;
        let pool = EdaBitPool::plan(&plan.edabits_per_depth());
        log::info!(
            "Planner: {} op-depths over {} engine depths, {} edaBits ({} random bits)",
            counts.depth(),
            plan.engine_depths(),
            plan.edabits_total(),
            plan.planner_bits()
        );
        Ok(Self { app, plan, pool, run: None, completed: 0 })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn app(&self) -> &A {
        &self.app
    }

    /// Fill the edaBit pool from the engine's random bits.
    pub(crate) fn fill_edabits(&mut self, signs: &[FieldElement<F>]) -> Result<()> {
        self.pool.fill(signs)
    }

    // -- 1. schedule ---------------------------------------------------------

    pub(crate) fn schedule(&mut self, depth: usize, op: Op<F>) -> Result<()> {
        if let Some(run) = &self.run {
            bail!("op-depth {} scheduled while op-depth {} is still running", depth, run.op_depth);
        }
        if depth != self.completed + 1 {
            bail!("op-depth {} scheduled after op-depth {}; op-depths run in order", depth, self.completed);
        }
        let Some(op_depth_plan) = self.plan.op_depth(depth) else {
            bail!("op-depth {} is outside the {} declared", depth, self.plan.op_depths().len());
        };
        op.validate(self.plan.ell())?;
        if !op_depth_plan.params.covers(&op.params()) {
            bail!("op-depth {} runs {:?} but declared {:?}", depth, op.params(), op_depth_plan.params);
        }
        let mut edabits = self.pool.for_depth(depth, op.len() * op.op_type().edabits_per_element())?.into_iter();
        let op = ops::build(op, &mut edabits)?;
        log::info!("Planner: op-depth {} started, {} engine rounds", depth, op_depth_plan.rounds());
        self.run = Some(DepthRun { op_depth: depth, op, step: 0, out: None });
        Ok(())
    }

    // -- 2. next batch --------------------------------------------------------

    /// The batch for the op's next step, or `None` when its steps are
    /// exhausted. A step with nothing in it — an op run over zero elements —
    /// is passed over, identically at every party.
    pub(crate) fn next_batch(&mut self) -> Result<Option<Batch<F>>> {
        let Some(run) = self.run.as_mut() else {
            bail!("no op-depth is running");
        };
        if run.out.is_some() {
            bail!("op-depth {} already has a step out", run.op_depth);
        }
        let op_depth_plan = self.plan.op_depth(run.op_depth).expect("scheduled against the plan");
        while run.step < op_depth_plan.rounds.len() {
            let round = &op_depth_plan.rounds[run.step];
            let operands = run.op.operands(run.step);
            if operands.batch_type() != round.step.engine_op {
                bail!("op-depth {} step {}: the op produced a {:?} batch for a {:?} round", run.op_depth, run.step, operands.batch_type(), round.step.engine_op);
            }
            if operands.is_empty() {
                run.step += 1;
                continue;
            }
            log::info!("Planner: op-depth {} step {} ({:?}) -> engine depth {}", run.op_depth, run.step, round.step.engine_op, round.engine_depth);
            run.out = Some((round.engine_depth, operands.len()));
            return Ok(Some(Batch { engine_depth: round.engine_depth, operands }));
        }
        Ok(None)
    }

    // -- 3. deliver -----------------------------------------------------------

    pub(crate) fn deliver(&mut self, engine_depth: usize, results: Vec<FieldElement<F>>) -> Result<()> {
        let Some(run) = self.run.as_mut() else {
            bail!("engine depth {} completed with no op-depth running", engine_depth);
        };
        let Some((expected_depth, expected_len)) = run.out.take() else {
            bail!("engine depth {} completed but no step was out", engine_depth);
        };
        if engine_depth != expected_depth {
            bail!("engine depth {} completed while depth {} was out", engine_depth, expected_depth);
        }
        if results.len() != expected_len {
            bail!("engine depth {} returned {} results for {} operands", engine_depth, results.len(), expected_len);
        }
        run.op.on_step_complete(run.step, results)?;
        run.step += 1;
        Ok(())
    }

    // -- 4. complete ----------------------------------------------------------

    /// The running op-depth's number and result, once `next_batch` has
    /// returned `None`.
    pub(crate) fn complete(&mut self) -> Result<(usize, OpResult<F>)> {
        let Some(run) = self.run.take() else {
            bail!("no op-depth to complete");
        };
        self.completed = run.op_depth;
        log::info!("Planner: op-depth {} complete", run.op_depth);
        Ok((run.op_depth, run.op.into_result()))
    }
}
