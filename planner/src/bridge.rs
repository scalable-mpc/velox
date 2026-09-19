//! The bridge: the Planner's interaction with the MPC engine.
//!
//! The engine calls the hooks below. Each one is a short translation:
//! the application's answer (`OpDepthInput`) is scheduled on the Planner
//! and the Planner's next [`Batch`] is returned as a `DepthInput`; a
//! completed engine depth is delivered to the Planner and the next batch
//! follows. When an op-depth's op_depth_plan is exhausted, the application's
//! `on_depth_complete` runs and its answer starts the cycle again. The
//! random bits the Planner reserved are taken off the front of what the
//! engine delivers, and the rest is passed through.

use anyhow::{bail, Result};
use crate::api::engine::{Application, DepthInput, PreprocessingCounts, RandomWireShares, RandomWires};
use async_trait::async_trait;
use fields::{MersennePrimeField, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

use crate::{
    api::application::{OpDepthInput, PlannerApplication},
    ops::EngineOperands,
    planner::{Batch, Planner},
};

impl<F: ProtocolField + MersennePrimeField, A: PlannerApplication<F>> Planner<F, A> {
    /// Act on what the application scheduled, and run the Planner until it
    /// has something for the engine.
    async fn drive(&mut self, input: Result<OpDepthInput<F>>) -> Result<DepthInput<F>> {
        match input? {
            OpDepthInput::Waiting => Ok(DepthInput::Waiting),
            OpDepthInput::Done(outputs) => Ok(DepthInput::Done(outputs)),
            OpDepthInput::Op { depth, op } => {
                self.schedule(depth, op)?;
                self.continue_run().await
            }
        }
    }

    /// The next batch of the running op-depth, or — if its rounds are done —
    /// the application's next move.
    async fn continue_run(&mut self) -> Result<DepthInput<F>> {
        match self.next_batch()? {
            Some(batch) => Ok(to_depth_input(batch)?),
            None => {
                let (op_depth, result) = self.complete()?;
                let next = self.app.on_depth_complete(op_depth, result).await;
                Box::pin(self.drive(next)).await
            }
        }
    }
}

fn to_depth_input<F: ProtocolField>(batch: Batch<F>) -> Result<DepthInput<F>> {
    let depth = batch.engine_depth;
    match batch.operands {
        EngineOperands::Reveal(values) => DepthInput::reveal(depth, values),
        EngineOperands::Multiply { x, y } => DepthInput::multiply(depth, x, y),
        EngineOperands::MaskedMultiply { x, y, mask } => DepthInput::masked_multiply(depth, x, y, mask),
    }
}

#[async_trait]
impl<F: ProtocolField + MersennePrimeField, A: PlannerApplication<F>> Application<F> for Planner<F, A> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        self.plan().preprocessing_counts()
    }

    fn random_wires(&self) -> RandomWires {
        self.plan().random_wires()
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        self.app.inputs().await
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        let next = self.app.input_sharing_termination(party, shares).await;
        self.drive(next).await
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        let mine = self.plan().planner_bits();
        if wires.bits.len() < mine {
            bail!("the Planner asked for {} random bits and got {}", mine, wires.bits.len());
        }
        let mut bits = wires.bits;
        let rest = bits.split_off(mine);
        self.fill_edabits(&bits)?;
        drop(bits);
        let next = self.app.on_preprocessing_complete(RandomWireShares::new(rest, wires.sharings)).await;
        self.drive(next).await
    }

    /// A multiply or masked-multiply depth finished: products in.
    async fn on_depth_complete(&mut self, depth: usize, results: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        self.deliver(depth, results)?;
        self.continue_run().await
    }

    /// A reveal or masked-multiply depth finished: public values in.
    async fn on_reveal_complete(&mut self, depth: usize, values: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        self.deliver(depth, values)?;
        self.continue_run().await
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        self.app.on_output(outputs).await
    }
}
