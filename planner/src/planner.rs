//! The Planner: what the engine hosts as its `Application`, and what hosts a
//! [`PlannerApplication`].
//!
//! It runs one op-depth at a time, one engine depth at a time. The engine
//! calls the hooks at the bottom of this file; every one of them ends in one
//! of two functions:
//!
//! - [`start_scheduled_op_depth`](Planner::start_scheduled_op_depth) takes
//!   what the application returned. For an `Op` it checks the op against its
//!   declared parameters, draws the op-depth's edaBits and builds the
//!   [`Operation`]; `Waiting` and `Done` pass straight to the engine.
//! - [`next_engine_depth_or_finish_op_depth`](Planner::next_engine_depth_or_finish_op_depth)
//!   hands the engine the running op's operands for its next step, at that
//!   step's engine depth. When the op has no steps left, it gives the op's
//!   result to the application and starts whatever the application
//!   schedules next.
//!
//! Between the two, `on_depth_complete` gives an engine depth's results back
//! to the op. The Planner knows no operation's arithmetic (that is `ops`) and
//! decides no engine protocol (that is the engine).

use anyhow::{bail, Result};
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

use crate::{
    api::{
        application::{OpDepthInput, PlannerApplication, PlannerCounts},
        engine::{Application, DepthInput, PreprocessingCounts, RandomWireShares, RandomWires},
    },
    ops::{self, EngineOperands, Operation},
    plan::Plan,
    primitives::edabit::EdaBitPool,
};

/// The op-depth currently being executed.
struct DepthRun<F: ProtocolField> {
    op_depth: usize,
    op: Box<dyn Operation<F>>,
    /// The next of the op's steps.
    step: usize,
    /// The step whose engine depth is out: that depth and its result count.
    out: Option<(usize, usize)>,
}

pub struct Planner<F: ProtocolField, A: PlannerApplication<F>> {
    pub(crate) app: A,
    execution_plan: Plan,
    edabit_pool: EdaBitPool<F>,
    current_engine_run: Option<DepthRun<F>>,
    /// Highest op-depth completed; op-depths run in order.
    completed: usize,
}

impl<F: ProtocolField, A: PlannerApplication<F>> Planner<F, A> {
    /// Compile the application's declaration into the engine's plan.
    pub fn new(app: A) -> Result<Self> {
        let counts: PlannerCounts = app.preprocessing_count();
        let plan = Plan::compile(&counts, F::MERSENNE_BITS)?;
        let pool = EdaBitPool::plan(&plan.edabits_per_depth());
        log::info!(
            "Planner: {} op-depths over {} engine depths, {} edaBits ({} random bits)",
            counts.depth(),
            plan.engine_depths(),
            plan.edabits_total(),
            plan.planner_bits()
        );
        Ok(Self { app, execution_plan: plan, edabit_pool: pool, current_engine_run: None, completed: 0 })
    }

    pub fn plan(&self) -> &Plan {
        &self.execution_plan
    }

    pub fn app(&self) -> &A {
        &self.app
    }

    /// Act on what a hook of the application returned: start the op-depth it
    /// scheduled and return that op-depth's first engine depth, or pass
    /// `Waiting` / `Done` through to the engine.
    async fn start_scheduled_op_depth(&mut self, app_input: Result<OpDepthInput<F>>) -> Result<DepthInput<F>> {
        let (depth, op) = match app_input? {
            OpDepthInput::Waiting => return Ok(DepthInput::Waiting),
            OpDepthInput::Done(outputs) => return Ok(DepthInput::Done(outputs)),
            OpDepthInput::Op { depth, op } => (depth, op),
        };
        if let Some(run) = &self.current_engine_run {
            bail!("op-depth {} scheduled while op-depth {} is still running", depth, run.op_depth);
        }
        if depth != self.completed + 1 {
            bail!("op-depth {} scheduled after op-depth {}; op-depths run in order", depth, self.completed);
        }
        let Some(op_depth_plan) = self.execution_plan.op_depth(depth) else {
            bail!("op-depth {} is outside the {} declared", depth, self.execution_plan.op_depths().len());
        };
        op.validate(self.execution_plan.ell())?;
        if !op_depth_plan.params.covers(&op.params()) {
            bail!("op-depth {} runs {:?} but declared {:?}", depth, op.params(), op_depth_plan.params);
        }
        let mut edabits = self.edabit_pool.for_depth(depth, op.len() * op.op_type().edabits_per_element())?.into_iter();
        let op = ops::build(op, &mut edabits)?;
        log::info!("Planner: op-depth {} started, {} engine rounds", depth, op_depth_plan.rounds());
        self.current_engine_run = Some(DepthRun { op_depth: depth, op, step: 0, out: None });
        self.next_engine_depth().await
    }

    /// The running op's operands for its next step, as the engine depth the
    /// plan put that step at. When the op has no steps left, hand its result
    /// to the application and start what the application schedules next.
    /// A step with no operands — an op run over zero elements — is passed
    /// over, identically at every party.
    async fn next_engine_depth(&mut self) -> Result<DepthInput<F>> {
        let Some(run) = self.current_engine_run.as_mut() else {
            bail!("no op-depth is running");
        };
        if run.out.is_some() {
            bail!("op-depth {} already has an engine depth out", run.op_depth);
        }
        // The op-depth's plan lists, step by step, the engine depth each of the
        // op's steps runs at and the kind of batch it must be.
        let op_depth_plan = self.execution_plan.op_depth(run.op_depth).expect("scheduled against the plan");
        while run.step < op_depth_plan.rounds.len() {
            let round = &op_depth_plan.rounds[run.step];
            let operands = run.op.operands(run.step);
            if operands.batch_type() != round.step.engine_op {
                bail!(
                    "op-depth {} step {}: the op produced a {:?} batch for a {:?} round",
                    run.op_depth, run.step, operands.batch_type(), round.step.engine_op
                );
            }
            if operands.is_empty() {
                run.step += 1;
                continue;
            }
            let depth = round.engine_depth;
            log::info!("Planner: op-depth {} step {} ({:?}) -> engine depth {}", run.op_depth, run.step, round.step.engine_op, depth);
            run.out = Some((depth, operands.len()));
            return match operands {
                EngineOperands::Reveal(values) => DepthInput::reveal(depth, values),
                EngineOperands::Multiply { x, y } => DepthInput::multiply(depth, x, y),
                EngineOperands::MaskedMultiply { x, y, mask } => DepthInput::masked_multiply(depth, x, y, mask),
            };
        }

        // Every step has run: the op-depth is finished.
        let run = self.current_engine_run.take().expect("checked above");
        self.completed = run.op_depth;
        log::info!("Planner: op-depth {} complete", run.op_depth);
        let app_input = self.app.on_depth_complete(run.op_depth, run.op.into_result()).await;
        Box::pin(self.start_scheduled_op_depth(app_input)).await
    }
}

#[async_trait]
impl<F: ProtocolField, A: PlannerApplication<F>> Application<F> for Planner<F, A> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        self.execution_plan.preprocessing_counts()
    }

    fn random_wires(&self) -> RandomWires {
        self.execution_plan.random_wires()
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        self.app.inputs().await
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        let app_input = self.app.input_sharing_termination(party, shares).await;
        self.start_scheduled_op_depth(app_input).await
    }

    /// The Planner's own random bits come off the front of what the engine
    /// delivers and fill the edaBit pool; the rest go to the application.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        let planner_bits = self.execution_plan.planner_bits();
        if wires.bits.len() < planner_bits {
            bail!("the Planner asked for {} random bits and got {}", planner_bits, wires.bits.len());
        }
        let mut bits = wires.bits;
        let app_bits = bits.split_off(planner_bits);
        self.edabit_pool.fill(&bits)?;
        drop(bits);
        let app_input = self.app.on_preprocessing_complete(RandomWireShares::new(app_bits, wires.sharings)).await;
        self.start_scheduled_op_depth(app_input).await
    }

    /// An engine depth finished — products of a multiply, or the public
    /// values of a reveal or masked multiply: they go to the running op's
    /// step, and the op moves on.
    async fn on_depth_complete(&mut self, engine_depth: usize, results: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        let Some(run) = self.current_engine_run.as_mut() else {
            bail!("engine depth {} completed with no op-depth running", engine_depth);
        };
        let Some((expected_depth, expected_len)) = run.out.take() else {
            bail!("engine depth {} completed but no engine depth was out", engine_depth);
        };
        if engine_depth != expected_depth {
            bail!("engine depth {} completed while depth {} was out", engine_depth, expected_depth);
        }
        if results.len() != expected_len {
            bail!("engine depth {} returned {} results for {} operands", engine_depth, results.len(), expected_len);
        }
        run.op.on_step_complete(run.step, results)?;
        run.step += 1;
        self.next_engine_depth().await
    }

    /// The engine reports reveals and masked multiplies here; to the Planner
    /// they are engine depths like any other.
    async fn on_reveal_complete(&mut self, engine_depth: usize, values: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        self.on_depth_complete(engine_depth, values).await
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        self.app.on_output(outputs).await
    }
}
