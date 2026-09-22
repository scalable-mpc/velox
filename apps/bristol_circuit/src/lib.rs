//! Evaluation of a Bristol-style arithmetic circuit on the MPC engine.
//!
//! This application reads a `.arith` file (see `docs/CIRCUIT_FORMAT.md`) and
//! evaluates whatever it describes, so a new computation is a new file rather
//! than a new application. The gate set is `ADD`, `SUB`, `MUL` and the op
//! gates `LT`, `DRELU`, `RELU`, `MAX`, `MIN`, `TRUNC d`, `FMUL d`.
//!
//! It is a [`PlannerApplication`]: the Planner hosts it and is the only thing
//! it talks to, so it runs over a Mersenne-prime field (`m61base`,
//! `m31base`), which is what the op gates need. Each op group of the circuit —
//! the gates of one type at one level — is one op-depth.
//!
//!   - `preprocessing_count` sizes preprocessing from the op groups, level by
//!     level. Linear gates cost neither a round nor a mask, which is why
//!     [`circuit`] levelises by round-costing gates alone.
//!   - `inputs` deals this party's slice of the input wires, positionally:
//!     the file's header says how many wires each input party supplies, and
//!     party `i` supplies the `i`-th block.
//!   - `input_sharing_termination` binds each dealer's shares to its wires and
//!     starts the circuit once every input dealer has terminated *and*
//!     preprocessing is in — the same two-flag pattern `AnonymousBroadcast`
//!     uses, since the two events race.
//!   - `on_depth_complete` writes an op-depth's results onto its group's
//!     output wires, evaluates the linear gates the level unblocks once its
//!     last group is in, and schedules the next group. The last group
//!     returns `Done`.
//!
//! Every gate of one type at a level goes into **one** op group, so a level
//! of one type costs one op however wide it is. A level mixing types runs its
//! groups one after another — the Planner takes one type per op-depth — so
//! a circuit author who cares about rounds keeps a level to one type.
//!
//! The format's `INNERP` gate is not supported yet; the parser rejects it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use async_trait::async_trait;
use velox::FieldElement;
use velox::{MersennePrimeField, ProtocolField};

pub mod parser;
pub use parser::{parse_circuit, parse_circuit_file};

use circuit::{Circuit, GateType, Wire};
use velox::{Op, OpDepthInput, OpParams, OpResult, OpType, PlannerApplication, PlannerCounts, RandomWireShares};

/// The op group in flight: where it sits in the circuit and where its results go.
struct Scheduled {
    level: usize,
    /// Index of the op group within its level.
    group: usize,
    gate_type: GateType,
    /// Output wires, in the order the results come back.
    outputs: Vec<Wire>,
}

pub struct BristolCircuit<F: ProtocolField + MersennePrimeField> {
    pub num_nodes: usize,
    pub my_id: usize,

    /// The circuit being evaluated, levelised by multiplicative depth.
    circuit: Circuit,

    /// Values this party secret-shares onto its input wires. Populated by
    /// [`BristolCircuit::with_inputs`]; random values pad a short list.
    my_inputs: Option<Vec<FieldElement<F>>>,
    /// Input sharings received from each dealer, keyed by dealer.
    input_wire_sharings: HashMap<usize, Vec<FieldElement<F>>>,
    /// Set once the input wires have been bound, so late dealers are ignored.
    inputs_assembled: bool,
    /// Set once the first op group has been scheduled, so the circuit starts
    /// exactly once.
    circuit_started: bool,
    /// Set once preprocessing has been handed over.
    preprocessing_done: bool,

    /// Sharing on each wire, indexed by wire number. `None` until written.
    wires: Vec<Option<FieldElement<F>>>,

    /// The op group whose results are awaited.
    scheduled: Option<Scheduled>,
    /// The next op group to schedule: `(level, index within the level)`.
    cursor: (usize, usize),
    /// Op groups completed. Groups are numbered from 1 in circuit order: the
    /// group number is the Planner's op-depth.
    completed: usize,
}

impl<F: ProtocolField + MersennePrimeField> BristolCircuit<F> {
    /// Builds an application over an already-parsed circuit.
    ///
    /// Fails if the circuit expects inputs from more parties than the
    /// deployment has: input wires are bound positionally, so a party index the
    /// deployment cannot produce would leave those wires permanently unwritten.
    pub fn new(num_nodes: usize, my_id: usize, circuit: Circuit) -> Result<Self> {
        if circuit.num_input_parties() > num_nodes {
            bail!(
                "circuit expects inputs from {} parties, but the deployment has {} nodes",
                circuit.num_input_parties(),
                num_nodes
            );
        }
        let num_wires = circuit.num_wires();
        Ok(Self {
            num_nodes,
            my_id,
            circuit,
            my_inputs: None,
            input_wire_sharings: HashMap::new(),
            inputs_assembled: false,
            circuit_started: false,
            preprocessing_done: false,
            wires: vec![None; num_wires],
            scheduled: None,
            cursor: (1, 0),
            completed: 0,
        })
    }

    /// Builds an application over the circuit in a `.arith` file.
    pub fn from_file<P: AsRef<Path>>(num_nodes: usize, my_id: usize, path: P) -> Result<Self> {
        let circuit = parse_circuit_file(path)?;
        log::info!(
            "BristolCircuit: parsed a circuit of {} gates ({} round-costing in {} op groups, {} linear) over {} \
             wires, multiplicative depth {}, {} inputs from {} parties, {} outputs",
            circuit.num_gates(),
            circuit.num_mult_gates(),
            circuit.num_op_groups(),
            circuit.num_add_gates(),
            circuit.num_wires(),
            circuit.multiplicative_depth(),
            circuit.num_inputs(),
            circuit.num_input_parties(),
            circuit.num_outputs(),
        );
        Self::new(num_nodes, my_id, circuit)
    }

    /// Supply the values this party feeds onto its input wires. Without this the
    /// party shares random field elements, which exercises the circuit but
    /// computes nothing meaningful.
    pub fn with_inputs(mut self, inputs: Vec<FieldElement<F>>) -> Self {
        self.my_inputs = Some(inputs);
        self
    }

    /// The circuit this application evaluates.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Number of input wires this party supplies — zero for a party the
    /// circuit's header does not name, which then deals nothing.
    pub fn inputs_per_party(&self) -> usize {
        self.circuit.inputs_of_party(self.my_id)
    }

    /// The secrets this party deals onto its input wires, padded with random
    /// values if the supplied list is short.
    pub fn generate_input_sharings(&self) -> Vec<FieldElement<F>> {
        let num_inputs = self.inputs_per_party();
        if num_inputs == 0 {
            return Vec::new();
        }
        let mut values = self.my_inputs.clone().unwrap_or_default();
        if values.len() < num_inputs {
            log::info!(
                "BristolCircuit: {} inputs supplied for {} input wires, padding with {} random values",
                values.len(),
                num_inputs,
                num_inputs - values.len()
            );
            values.extend((values.len()..num_inputs).map(|_| F::rand()));
        }
        values.truncate(num_inputs);
        values
    }

    /// The dealers whose input ACSS the circuit waits on: the parties the
    /// header gives a non-zero input count.
    ///
    /// Every one of them must terminate. Unlike anonymous broadcast — which
    /// takes any `k` of the sharings the honest majority produces — a circuit
    /// binds wires to dealers positionally, so a missing dealer is a missing
    /// wire and there is nothing to substitute for it.
    fn expected_dealers(&self) -> Vec<usize> {
        (0..self.circuit.num_input_parties())
            .filter(|party| self.circuit.inputs_of_party(*party) > 0)
            .collect()
    }

    /// Record a dealer's shares; `true` once the circuit may start.
    fn accept_input_sharing(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<bool> {
        if self.inputs_assembled {
            log::debug!("BristolCircuit: ignoring input sharing from party {}, input wires already bound", party);
            return Ok(false);
        }
        if self.circuit.inputs_of_party(party) == 0 {
            log::debug!(
                "BristolCircuit: party {} supplies no input wires in this circuit; ignoring its {} sharings",
                party,
                shares.len()
            );
            return Ok(false);
        }
        log::info!("BristolCircuit: input sharing from party {} terminated with {} sharings", party, shares.len());
        self.input_wire_sharings.insert(party, shares);
        self.try_assemble_input_wires()?;
        Ok(true)
    }

    /// Bind each dealer's shares to its block of input wires, once every
    /// expected dealer has terminated.
    fn try_assemble_input_wires(&mut self) -> Result<()> {
        if self.inputs_assembled {
            return Ok(());
        }
        let dealers = self.expected_dealers();
        if !dealers.iter().all(|party| self.input_wire_sharings.contains_key(party)) {
            return Ok(());
        }

        // Wires are numbered party by party in header order: party 0 owns the
        // first `n_0`, party 1 the next `n_1`, and so on.
        let mut wire = 0;
        for party in 0..self.circuit.num_input_parties() {
            let expected = self.circuit.inputs_of_party(party);
            if expected == 0 {
                continue;
            }
            let shares = &self.input_wire_sharings[&party];
            if shares.len() < expected {
                bail!("dealer {} owns {} input wires but dealt only {} sharings", party, expected, shares.len());
            }
            for share in shares[..expected].iter() {
                self.wires[wire] = Some(share.clone());
                wire += 1;
            }
        }

        log::info!("BristolCircuit: bound {} input wires from {} dealers", wire, dealers.len());
        self.inputs_assembled = true;
        // The per-dealer sharings have been copied onto the wires and have no
        // other reader. Dropping them is safe precisely because
        // `inputs_assembled` is now set: `accept_input_sharing` bails on that
        // flag before it would insert a late dealer back into this map, so an
        // emptied map can never be mistaken for "still waiting for dealers".
        self.input_wire_sharings.clear();
        Ok(())
    }

    /// Start the circuit as soon as both the input wires and the preprocessing
    /// material are in hand — the two arrive in either order. `true` when it
    /// starts now: level 0's linear gates are evaluated and the first op group
    /// is the caller's to schedule.
    fn try_start(&mut self) -> Result<bool> {
        if self.circuit_started || !self.inputs_assembled || !self.preprocessing_done {
            return Ok(false);
        }
        self.circuit_started = true;
        // Level 0 holds the linear gates that read only input wires; they are
        // evaluable before any op group has run.
        self.evaluate_linear_gates(0)?;
        Ok(true)
    }

    fn read_wire(&self, wire: Wire, gate_type: GateType, output: Wire, level: usize) -> Result<FieldElement<F>> {
        match self.wires[wire].clone() {
            Some(value) => Ok(value),
            None => bail!("{} gate writing wire {} at level {} reads unwritten wire {}", gate_type, output, level, wire),
        }
    }

    /// Evaluate the linear gates a level unblocks. They are held in topological
    /// order, so a single front-to-back pass never reads an unwritten wire.
    fn evaluate_linear_gates(&mut self, level: usize) -> Result<()> {
        let Some(depth) = self.circuit.level(level) else {
            return Ok(());
        };
        // The gate list is immutable for the run; cloning the level's linear
        // gates keeps the borrow off `self.wires`, which the loop writes.
        let gates = depth.linear_gates.clone();
        for gate in gates.iter() {
            let left = self.read_wire(gate.left(), gate.gate_type, gate.output, level)?;
            let right = self.read_wire(gate.right().expect("linear gates are binary"), gate.gate_type, gate.output, level)?;
            self.wires[gate.output] = Some(match gate.gate_type {
                GateType::Add => left + right,
                GateType::Sub => left - right,
                other => bail!("{} is not a linear gate", other),
            });
        }
        if !gates.is_empty() {
            log::info!("BristolCircuit: evaluated {} linear gates unblocked by level {}", gates.len(), level);
        }
        Ok(())
    }

    /// The operands of the next op group — its type, the left inputs and, for
    /// a binary type, the right inputs — or `None` once every group has run.
    /// Marks the group as scheduled.
    fn next_op_group(&mut self) -> Result<Option<(GateType, Vec<FieldElement<F>>, Vec<FieldElement<F>>)>> {
        if self.scheduled.is_some() {
            bail!("op group {} is still in flight", self.completed + 1);
        }
        let (level, index) = self.cursor;
        if level > self.circuit.multiplicative_depth() {
            return Ok(None);
        }
        let Some(group) = self.circuit.level(level).and_then(|depth| depth.op_groups.get(index)) else {
            bail!("no op group {} at level {}", index, level);
        };
        let group = group.clone();

        let mut x = Vec::with_capacity(group.len());
        let mut y = Vec::with_capacity(group.len());
        let mut outputs = Vec::with_capacity(group.len());
        for gate in group.gates.iter() {
            x.push(self.read_wire(gate.left(), gate.gate_type, gate.output, level)?);
            if let Some(right) = gate.right() {
                y.push(self.read_wire(right, gate.gate_type, gate.output, level)?);
            }
            outputs.push(gate.output);
        }

        log::info!(
            "BristolCircuit: scheduling op group {}/{} — level {}/{}, {} {} gates",
            self.completed + 1,
            self.circuit.num_op_groups(),
            level,
            self.circuit.multiplicative_depth(),
            outputs.len(),
            group.gate_type,
        );
        self.scheduled = Some(Scheduled { level, group: index, gate_type: group.gate_type, outputs });
        Ok(Some((group.gate_type, x, y)))
    }

    /// Write the scheduled group's results onto its output wires; if it was
    /// the last group of its level, evaluate the linear gates the level
    /// unblocks and move the cursor to the next level.
    fn apply_results(&mut self, results: Vec<FieldElement<F>>) -> Result<()> {
        let Some(scheduled) = self.scheduled.take() else {
            bail!("results arrived with no op group in flight");
        };
        if scheduled.outputs.len() != results.len() {
            bail!(
                "op group {} scheduled {} gates but {} results came back",
                self.completed + 1,
                scheduled.outputs.len(),
                results.len()
            );
        }
        for (wire, result) in scheduled.outputs.into_iter().zip(results.into_iter()) {
            self.wires[wire] = Some(result);
        }
        self.completed += 1;

        let groups_at_level = self.circuit.level(scheduled.level).map(|d| d.op_groups.len()).unwrap_or(0);
        if scheduled.group + 1 < groups_at_level {
            self.cursor = (scheduled.level, scheduled.group + 1);
        } else {
            self.evaluate_linear_gates(scheduled.level)?;
            self.cursor = (scheduled.level + 1, 0);
        }
        Ok(())
    }

    /// The type of the op group in flight.
    fn scheduled_type(&self) -> Option<GateType> {
        self.scheduled.as_ref().map(|s| s.gate_type)
    }

    /// A completion for op group `number` is the one awaited, or a replay.
    fn is_next_completion(&self, number: usize) -> bool {
        if number == self.completed + 1 && self.scheduled.is_some() {
            return true;
        }
        // The engine may replay a termination; op groups advance by one
        // and only forward, so anything else is a replay or out of order.
        log::warn!(
            "BristolCircuit: ignoring completion of op group {}; the circuit has completed {}",
            number,
            self.completed
        );
        false
    }

    /// Collect the circuit's output sharings, in the order the file lists them.
    fn finish(&mut self) -> Result<Vec<FieldElement<F>>> {
        let mut outputs = Vec::with_capacity(self.circuit.num_outputs());
        for wire in self.circuit.output_wires().iter() {
            match self.wires[*wire].clone() {
                Some(value) => outputs.push(value),
                None => bail!("output wire {} was never written", wire),
            }
        }
        log::info!(
            "BristolCircuit: circuit complete after {} op groups over {} levels, {} output wires",
            self.completed,
            self.circuit.multiplicative_depth(),
            outputs.len()
        );
        // The wire sharings are dead now that the outputs have been copied out;
        // a large circuit holds one field element per wire.
        self.wires = Vec::new();
        Ok(outputs)
    }

    // -- one op per op group --------------------------------------------------

    /// The Planner op a gate type runs as. `DRELU` is `1 − [a < 0]`, with
    /// the complement taken when the results come back.
    fn op_type_of(gate_type: GateType) -> OpType {
        match gate_type {
            GateType::Mul => OpType::Mul,
            GateType::Lt => OpType::Compare,
            GateType::DRelu => OpType::ComparePub,
            GateType::Relu => OpType::MaxPub,
            GateType::Max => OpType::Max,
            GateType::Min => OpType::Min,
            GateType::Trunc(_) => OpType::Truncate,
            GateType::FMul(_) => OpType::FixedMul,
            GateType::Add | GateType::Sub => unreachable!("linear gates are not grouped"),
        }
    }

    /// The next op group as a Planner op, or the circuit's outputs.
    fn schedule_op(&mut self) -> Result<OpDepthInput<F>> {
        let Some((gate_type, x, y)) = self.next_op_group()? else {
            return Ok(OpDepthInput::Done(self.finish()?));
        };
        let zeros = || vec![FieldElement::<F>::zero(); x.len()];
        let op = match gate_type {
            GateType::Mul => Op::Mul { x, y },
            GateType::Lt => Op::Compare { a: x, b: y },
            GateType::DRelu => Op::ComparePub { c: zeros(), a: x },
            GateType::Relu => Op::MaxPub { c: zeros(), a: x },
            GateType::Max => Op::Max { a: x, b: y },
            GateType::Min => Op::Min { a: x, b: y },
            GateType::Trunc(d) => Op::Truncate { x, d },
            GateType::FMul(d) => Op::FixedMul { x, y, d },
            GateType::Add | GateType::Sub => unreachable!("linear gates are not grouped"),
        };
        OpDepthInput::op(self.completed + 1, op)
    }
}

#[async_trait]
impl<F: ProtocolField + MersennePrimeField> PlannerApplication<F> for BristolCircuit<F> {
    fn preprocessing_count(&self) -> PlannerCounts {
        // One op-depth per op group, in circuit order: level by level, and within
        // a level in order of first appearance. The same file at every party
        // gives the same declaration.
        let ops: Vec<OpParams> = self
            .circuit
            .levels()
            .iter()
            .skip(1)
            .flat_map(|level| level.op_groups.iter())
            .map(|group| OpParams::new(Self::op_type_of(group.gate_type), group.len()))
            .collect();
        let counts = PlannerCounts::new(ops, self.circuit.num_outputs());
        log::info!(
            "BristolCircuit::preprocessing_count -> {} op-depths over {} levels, output={}",
            counts.depth(),
            self.circuit.multiplicative_depth(),
            counts.output
        );
        counts
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        self.generate_input_sharings()
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<OpDepthInput<F>> {
        self.accept_input_sharing(party, shares)?;
        if self.try_start()? { self.schedule_op() } else { Ok(OpDepthInput::Waiting) }
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        log::info!(
            "BristolCircuit: preprocessing complete — {} random bits and {} sharings passed through (unused); \
             circuit: {} op groups over {} levels",
            wires.bits.len(),
            wires.sharings.len(),
            self.circuit.num_op_groups(),
            self.circuit.multiplicative_depth(),
        );
        self.preprocessing_done = true;
        if self.try_start()? { self.schedule_op() } else { Ok(OpDepthInput::Waiting) }
    }

    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        log::info!("BristolCircuit: op-depth {} complete — {:?}", depth, result);
        if !self.is_next_completion(depth) {
            return Ok(OpDepthInput::Waiting);
        }
        let mut results = result.shares()?;
        if self.scheduled_type() == Some(GateType::DRelu) {
            // ComparePub gave [a < 0]; DReLU is its complement.
            let one = FieldElement::<F>::one();
            results = results.iter().map(|lt| &one - lt).collect();
        }
        self.apply_results(results)?;
        self.schedule_op()
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let signed: Vec<i128> = outputs.iter().map(circuit::to_signed::<F>).collect();
        log::info!("BristolCircuit: {} output wires reconstructed: {:?}", outputs.len(), signed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::{evaluate_circuit, from_signed, to_signed};
    use crate::{parse_circuit, parse_circuit_file};
    use velox::{fields::Mersenne61Field, Application, DepthInput, Planner, RandomWires};

    /// The tests exercise the application at one concrete field; the generic
    /// parameter is what the engine binds, not something the tests vary.
    type F = Mersenne61Field;

    const NUM_NODES: usize = 10;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/circuits")
            .join(name)
    }

    fn elem(value: i128) -> FieldElement<F> {
        from_signed::<F>(value)
    }

    fn signed_all(values: &[FieldElement<F>]) -> Vec<i128> {
        values.iter().map(to_signed::<F>).collect()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A stand-in engine under the Planner.
    ///
    /// It drives the Planner hosting the application through the same hook
    /// sequence `mpc::Context` does — preprocessing, then one batch per engine
    /// depth, then the output handover — but evaluates each batch in the clear:
    /// `Multiply` is the product, `Reveal` the identity, `MaskedMultiply` is
    /// `x·y + mask`, and the random bits are plaintext `±1`. Because linear
    /// operations on Shamir sharings are the same operations on the secrets,
    /// feeding secrets in where the engine would feed shares makes the outputs
    /// directly comparable to `evaluate_circuit`.
    struct Harness {
        planner: Planner<F, BristolCircuit<F>>,
        /// Engine depths run.
        depths: usize,
    }

    impl Harness {
        fn new(circuit: Circuit) -> Self {
            let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap();
            Self { planner: Planner::new(app).unwrap(), depths: 0 }
        }

        fn from_fixture(name: &str) -> Self {
            Self::new(parse_circuit_file(fixture(name)).unwrap())
        }

        fn circuit(&self) -> Circuit {
            self.planner.app().circuit().clone()
        }

        /// The engine hands over the random bits the Planner asked for.
        async fn deliver_preprocessing(&mut self) -> DepthInput<F> {
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
            let wires: RandomWires = self.planner.random_wires();
            let bits: Vec<FieldElement<F>> = (0..wires.bits)
                .map(|_| if rng.next() & 1 == 1 { FieldElement::<F>::one() } else { -FieldElement::<F>::one() })
                .collect();
            self.planner.on_preprocessing_complete(RandomWireShares::new(bits, Vec::new())).await.unwrap()
        }

        /// Hand the application every dealer's inputs, sliced out of the circuit's
        /// input-wire vector by the header's positional layout.
        async fn deliver_inputs(&mut self, inputs: &[FieldElement<F>]) -> DepthInput<F> {
            let circuit = self.circuit();
            let mut wire = 0;
            let mut last = DepthInput::Waiting;
            for party in 0..circuit.num_input_parties() {
                let count = circuit.inputs_of_party(party);
                let shares = inputs[wire..wire + count].to_vec();
                wire += count;
                last = self.planner.input_sharing_termination(party, shares).await.unwrap();
            }
            last
        }

        /// Run every engine depth the Planner schedules, and return the output
        /// sharings the application hands back at the end.
        async fn run(&mut self, mut next: DepthInput<F>) -> Vec<FieldElement<F>> {
            loop {
                next = match next {
                    DepthInput::Done(outputs) => return outputs,
                    DepthInput::Waiting => panic!("the application stalled with nothing scheduled"),
                    DepthInput::Multiply { depth, x, y } => {
                        self.depths += 1;
                        let products = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect();
                        self.planner.on_depth_complete(depth, products).await.unwrap()
                    }
                    DepthInput::Reveal { depth, values } => {
                        self.depths += 1;
                        self.planner.on_reveal_complete(depth, values).await.unwrap()
                    }
                    DepthInput::MaskedMultiply { depth, x, y, mask } => {
                        self.depths += 1;
                        let opened = x.iter().zip(y.iter()).zip(mask.iter()).map(|((a, b), m)| a * b + m).collect();
                        self.planner.on_reveal_complete(depth, opened).await.unwrap()
                    }
                };
            }
        }

        /// Preprocessing, inputs, every depth, outputs.
        async fn evaluate(&mut self, inputs: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
            let after_preprocessing = self.deliver_preprocessing().await;
            assert!(after_preprocessing.is_waiting(), "the circuit must wait for its input wires before starting");
            let started = self.deliver_inputs(inputs).await;
            self.run(started).await
        }
    }

    /// Every arithmetic fixture evaluated through the Planner must agree with
    /// the same circuit evaluated in the clear, and must cost exactly one
    /// engine depth per multiplicative level.
    #[tokio::test]
    async fn fixtures_match_cleartext_evaluation() {
        let fixtures = [
            "simple_add.arith",
            "simple_mul.arith",
            "multiply_three.arith",
            "polynomial_eval.arith",
            "inner_product_2.arith",
        ];

        for name in fixtures {
            let circuit = parse_circuit_file(fixture(name)).unwrap();
            let inputs: Vec<FieldElement<F>> = (1..=circuit.num_inputs() as i128).map(|value| elem(value * 7 - 20)).collect();
            let expected = evaluate_circuit::<F>(&circuit, &inputs).unwrap();

            let mut harness = Harness::from_fixture(name);
            let outputs = harness.evaluate(&inputs).await;

            assert_eq!(outputs, expected, "{}: outputs disagree with cleartext evaluation", name);
            assert_eq!(harness.depths, circuit.multiplicative_depth(), "{}: one engine depth per level", name);
        }
    }

    /// The comparison fixture agrees with the cleartext reference: exactly on
    /// every wire but the truncations, which the protocol computes to within
    /// ±2; and it costs the rounds the op table says.
    #[tokio::test]
    async fn comparison_fixture_matches_cleartext_evaluation() {
        let circuit = parse_circuit_file(fixture("comparison.arith")).unwrap();
        // The first case is the fixture's input files: a = 3, b = 5 from
        // party 0; c = 11, d = −7 from party 1.
        let cases: Vec<Vec<i128>> = vec![
            vec![3, 5, 11, -7],
            vec![-9, 20, -9, 20],
            vec![1 << 40, -(1 << 40), 12345, -54321],
        ];
        // Outputs: w4 w5 w7 w8 w9 w10 w11 w13 w14; w10 and w11 are truncations.
        let truncated = [5usize, 6];
        for values in cases {
            let inputs: Vec<FieldElement<F>> = values.iter().map(|v| elem(*v)).collect();
            let expected = signed_all(&evaluate_circuit::<F>(&circuit, &inputs).unwrap());
            let mut harness = Harness::new(circuit.clone());
            let got = signed_all(&harness.evaluate(&inputs).await);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                if truncated.contains(&i) {
                    assert!((g - e).abs() <= 2, "inputs {values:?}: output {i}: got {g}, expected {e} ± 2");
                } else {
                    assert_eq!(g, e, "inputs {values:?}: output {i}");
                }
            }
            // MUL 1 + LT 8 + RELU 9 + MAX 9 + MIN 9 + FMUL 1 + DRELU 8, then TRUNC 1.
            assert_eq!(harness.depths, 1 + 8 + 9 + 9 + 9 + 1 + 8 + 1);
        }
        let mut harness = Harness::new(circuit);
        let got = signed_all(&harness.evaluate(&[elem(3), elem(5), elem(11), elem(-7)]).await);
        assert_eq!(&got[..5], &[15, 1, 12, 11, -7]);
        assert!((got[5] - 0).abs() <= 2 && (got[6] - (-4)).abs() <= 2, "{got:?}");
        assert_eq!(&got[7..], &[2, 0]);
    }

    /// `f(x) = a·x² + b·x + c` in two rounds, not four: level 1 groups `x²` and
    /// `b·x` together, level 2 runs `a·x²` and folds in both additions.
    #[tokio::test]
    async fn polynomial_eval_runs_two_rounds() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        // a = 3, b = 5, c = 7 from party 0; x = 11 from party 1.
        let inputs = vec![elem(3), elem(5), elem(7), elem(11)];

        let mut harness = Harness::new(circuit);
        let outputs = harness.evaluate(&inputs).await;

        assert_eq!(outputs, vec![elem(3 * 121 + 5 * 11 + 7)]);
        assert_eq!(harness.depths, 2);
    }

    /// Independent gates at the same level share one op group, however many
    /// there are — the grouping is where the protocol's efficiency comes from.
    #[tokio::test]
    async fn independent_gates_share_one_round() {
        // w4 = w0·w1, w5 = w2·w3, w6 = w4 + w5 — an inner product spelled out.
        let text = "\
3 7
1
4
1
6

2 1 0 1 4 MUL
2 1 2 3 5 MUL
2 1 4 5 6 ADD
";
        let circuit = parse_circuit(text, "inline").unwrap();
        assert_eq!(circuit.multiplicative_depth(), 1);

        let inputs = vec![elem(2), elem(3), elem(5), elem(7)];
        let mut harness = Harness::new(circuit);
        let outputs = harness.evaluate(&inputs).await;

        // 2·3 + 5·7 = 6 + 35 = 41.
        assert_eq!(outputs, vec![elem(41)]);
        assert_eq!(harness.depths, 1, "two gates, one round");
    }

    /// A level mixing types runs one op group per type, one after another.
    #[tokio::test]
    async fn mixed_level_runs_one_op_group_per_type() {
        // w4 = w0·w1 and w5 = [w2 < w3] at level 1: a MUL group and an LT group.
        let text = "\
2 6
1
4
2
4 5

2 1 0 1 4 MUL
2 1 2 3 5 LT
";
        let circuit = parse_circuit(text, "inline").unwrap();
        assert_eq!((circuit.multiplicative_depth(), circuit.num_op_groups()), (1, 2));
        let counts = PlannerApplication::preprocessing_count(&BristolCircuit::<F>::new(NUM_NODES, 0, circuit.clone()).unwrap());
        assert_eq!(counts.ops, vec![OpParams::new(OpType::Mul, 1), OpParams::new(OpType::Compare, 1)]);

        let mut harness = Harness::new(circuit);
        let outputs = harness.evaluate(&[elem(2), elem(3), elem(-5), elem(7)]).await;
        assert_eq!(signed_all(&outputs), vec![6, 1]);
        assert_eq!(harness.depths, 1 + 8, "the MUL's round, then the LT's eight");
    }

    /// `INNERP` is part of the format but not runnable yet, so a circuit using
    /// it never reaches the application.
    #[test]
    fn innerp_circuits_do_not_load() {
        assert!(parse_circuit_file(fixture("inner_product_4_innerp.arith")).is_err());
    }

    /// A circuit of only linear gates costs no round at all: the application
    /// hands back outputs the moment its inputs and preprocessing are in.
    #[tokio::test]
    async fn purely_linear_circuit_returns_outputs_without_a_round() {
        let mut harness = Harness::from_fixture("simple_add.arith");
        let after_preprocessing = harness.deliver_preprocessing().await;
        assert!(after_preprocessing.is_waiting());

        let started = harness.deliver_inputs(&[elem(4), elem(9)]).await;
        match started {
            DepthInput::Done(outputs) => assert_eq!(outputs, vec![elem(13)]),
            _ => panic!("a linear circuit must finish without a round"),
        }
    }

    /// The two starting conditions race, so the circuit must start on whichever
    /// arrives second — here inputs first, preprocessing second.
    #[tokio::test]
    async fn circuit_starts_when_preprocessing_arrives_after_the_inputs() {
        let mut harness = Harness::from_fixture("simple_mul.arith");

        let after_inputs = harness.deliver_inputs(&[elem(6), elem(7)]).await;
        assert!(after_inputs.is_waiting(), "no preprocessing yet, so nothing can be scheduled");

        let started = harness.deliver_preprocessing().await;
        let outputs = harness.run(started).await;

        assert_eq!(outputs, vec![elem(42)]);
    }

    /// The Planner declaration: one op-depth per op group, typed and sized, in
    /// circuit order.
    #[test]
    fn planner_counts_follow_the_op_groups() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap();
        let counts = app.preprocessing_count();
        assert_eq!(counts.ops, vec![OpParams::new(OpType::Mul, 2), OpParams::new(OpType::Mul, 1)]);
        assert_eq!(counts.output, 1);
        assert_eq!((counts.rand_bits, counts.sharings), (0, 0), "the circuit reads no random wires itself");

        let circuit = parse_circuit_file(fixture("comparison.arith")).unwrap();
        let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap();
        let counts = app.preprocessing_count();
        let ops: Vec<(OpType, usize)> = counts.ops.iter().map(|p| (p.op_type, p.elements)).collect();
        assert_eq!(
            ops,
            vec![
                (OpType::Mul, 1),
                (OpType::Compare, 2),
                (OpType::MaxPub, 1),
                (OpType::Max, 1),
                (OpType::Min, 1),
                (OpType::FixedMul, 1),
                (OpType::ComparePub, 1),
                (OpType::Truncate, 1),
            ]
        );
        assert_eq!(counts.output, 9);
    }

    /// Input wires are bound positionally, so each party deals exactly the block
    /// of wires the header gives it and parties the header does not name deal
    /// nothing.
    #[tokio::test]
    async fn input_sharing_is_positional() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();

        let mut party_zero = BristolCircuit::<F>::new(NUM_NODES, 0, circuit.clone())
            .unwrap()
            .with_inputs(vec![elem(3), elem(5), elem(7)]);
        let mut party_one = BristolCircuit::<F>::new(NUM_NODES, 1, circuit.clone())
            .unwrap()
            .with_inputs(vec![elem(11)]);
        let mut party_four = BristolCircuit::<F>::new(NUM_NODES, 4, circuit).unwrap();

        assert_eq!(party_zero.inputs_per_party(), 3);
        assert_eq!(party_one.inputs_per_party(), 1);
        assert_eq!(party_four.inputs_per_party(), 0);

        assert_eq!(party_zero.inputs().await, vec![elem(3), elem(5), elem(7)]);
        assert_eq!(party_one.inputs().await, vec![elem(11)]);
        assert!(party_four.inputs().await.is_empty(), "a party the circuit gives no input wires deals nothing");
    }

    /// A short input list is padded rather than rejected, so a party can run the
    /// circuit without a complete input file.
    #[test]
    fn short_input_lists_are_padded() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap().with_inputs(vec![elem(3)]);

        let sharings = app.generate_input_sharings();

        assert_eq!(sharings.len(), 3);
        assert_eq!(sharings[0], elem(3));
    }

    /// Drive the application's own hooks, below the Planner: preprocessing and
    /// inputs in, the first op out.
    async fn start_bare(name: &str, inputs: &[FieldElement<F>]) -> (BristolCircuit<F>, Op<F>) {
        let circuit = parse_circuit_file(fixture(name)).unwrap();
        let mut app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit.clone()).unwrap();
        assert!(app.on_preprocessing_complete(RandomWireShares::empty()).await.unwrap().is_waiting());
        let mut wire = 0;
        let mut last = OpDepthInput::Waiting;
        for party in 0..circuit.num_input_parties() {
            let count = circuit.inputs_of_party(party);
            last = app.input_sharing_termination(party, inputs[wire..wire + count].to_vec()).await.unwrap();
            wire += count;
        }
        let OpDepthInput::Op { depth, op } = last else { panic!("op-depth 1 must be scheduled") };
        assert_eq!(depth, 1, "the application names the op-depth");
        (app, op)
    }

    /// A replayed op-depth completion must not re-run the group or advance the
    /// circuit past it.
    #[tokio::test]
    async fn replayed_completion_is_ignored() {
        let (mut app, op) = start_bare("multiply_three.arith", &[elem(2), elem(3), elem(5)]).await;
        let Op::Mul { x, y } = op else { panic!("level 1 is a MUL group") };
        let products: Vec<FieldElement<F>> = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect();

        let level_two = app.on_depth_complete(1, OpResult::Shares(products.clone())).await.unwrap();
        assert!(!level_two.is_waiting(), "op-depth 2 follows op-depth 1");

        let replay = app.on_depth_complete(1, OpResult::Shares(products)).await.unwrap();
        assert!(replay.is_waiting(), "a replayed op-depth 1 must schedule nothing");
    }

    /// An application error reaches the Planner as an `Err` rather than as a
    /// `Waiting` the engine would sit on forever.
    #[tokio::test]
    async fn a_broken_result_vector_is_reported_not_swallowed() {
        let (mut app, _) = start_bare("multiply_three.arith", &[elem(2), elem(3), elem(5)]).await;

        // Op-depth 1 scheduled one gate; hand back two results.
        let error = app
            .on_depth_complete(1, OpResult::Shares(vec![elem(1), elem(2)]))
            .await
            .expect_err("a mismatched result count must be an error")
            .to_string();
        assert!(error.contains("scheduled 1 gates but 2 results"), "got {:?}", error);

        // A public result where shares were expected is an error too.
        let (mut app, _) = start_bare("multiply_three.arith", &[elem(2), elem(3), elem(5)]).await;
        assert!(app.on_depth_complete(1, OpResult::Public(vec![elem(6)])).await.is_err());
    }

    /// A circuit naming more input parties than the deployment has can never
    /// have all its input wires written, so it is rejected at construction
    /// rather than stalling at run time.
    #[test]
    fn circuit_needing_more_parties_than_the_deployment_is_rejected() {
        let text = "\
1 3
2
1 1
1
2

2 1 0 1 2 MUL
";
        let circuit = parse_circuit(text, "inline").unwrap();
        let error = match BristolCircuit::<F>::new(1, 0, circuit) {
            Ok(_) => panic!("a circuit needing 2 input parties must not build on a 1-node deployment"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("expects inputs from 2 parties"), "got {:?}", error);
    }
}
