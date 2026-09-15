//! Evaluation of a Bristol-style arithmetic circuit on the MPC engine.
//!
//! This is the second concrete [`Application`], alongside
//! [`AnonymousBroadcast`](crate::AnonymousBroadcast). Where that one has its
//! circuit compiled in, this one reads a `.arith` file (see
//! `docs/CIRCUIT_FORMAT.md`) and evaluates whatever it describes, so a new
//! computation is a new file rather than a new `Application` impl.
//!
//! It implements the [`Application`] trait with no engine changes. The mapping is:
//!
//!   - [`Application::preprocessing_count`] sizes preprocessing from the gate
//!     count and the **multiplicative** depth. Linear gates cost neither a round
//!     nor a mask, which is why [`crate::circuit`] levelises by multiplication
//!     gates alone.
//!   - [`Application::inputs`] deals this party's slice of the input wires,
//!     positionally: the file's header says how many wires each input party
//!     supplies, and party `i` supplies the `i`-th block.
//!   - [`Application::input_sharing_termination`] binds each dealer's shares to
//!     its wires and starts the circuit once every input dealer has terminated
//!     *and* preprocessing is in — the same two-flag pattern
//!     `AnonymousBroadcast::try_start_circuit` uses, since the two events race.
//!   - [`Application::on_depth_complete`] writes a level's results onto their
//!     output wires, evaluates the linear gates that level unblocks, and
//!     schedules the next level's batch. The last level returns
//!     [`DepthInput::Done`] instead.
//!
//! One property is worth stating because it is where the performance is: every
//! `MUL` at a level goes into **one** batch, so a level costs one round however
//! wide it is. That is what makes levelising by multiplicative depth matter — a
//! level's width is free, its count is not.
//!
//! The format's `INNERP` gate is not supported yet; the parser rejects it. See
//! `docs/CIRCUIT_FORMAT.md` for what it would take.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use async_trait::async_trait;
use velox::ProtocolField;
use velox::FieldElement;

pub mod parser;
pub use parser::{parse_circuit, parse_circuit_file};

use circuit::{Circuit, Wire};
use velox::{Application, DepthInput, PreprocessingCounts, RandomWireShares};

pub struct BristolCircuit<F: ProtocolField> {
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
    /// Set once level 1 has been scheduled, so the circuit starts exactly once.
    circuit_started: bool,
    /// Set once preprocessing has been handed over.
    preprocessing_done: bool,

    /// Sharing on each wire, indexed by wire number. `None` until written.
    wires: Vec<Option<FieldElement<F>>>,

    /// Output wires of the level currently in flight, in the order the engine
    /// returns their results.
    scheduled_outputs: Vec<Wire>,
    /// Highest level whose results have been folded in. Levels run `1..=depth`.
    levels_completed: usize,
}

impl<F: ProtocolField> BristolCircuit<F> {
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
            scheduled_outputs: Vec::new(),
            levels_completed: 0,
        })
    }

    /// Builds an application over the circuit in a `.arith` file.
    pub fn from_file<P: AsRef<Path>>(num_nodes: usize, my_id: usize, path: P) -> Result<Self> {
        let circuit = parse_circuit_file(path)?;
        log::info!(
            "BristolCircuit: parsed a circuit of {} gates ({} multiplications, {} linear) over {} \
             wires, multiplicative depth {}, {} inputs from {} parties, {} outputs",
            circuit.num_gates(),
            circuit.num_mul_gates(),
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

    /// Bind each dealer's shares to its block of input wires, once every
    /// expected dealer has terminated.
    fn try_assemble_input_wires(&mut self) -> Result<()> {
        if self.inputs_assembled {
            return Ok(());
        }
        let dealers = self.expected_dealers();
        if !dealers
            .iter()
            .all(|party| self.input_wire_sharings.contains_key(party))
        {
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
                bail!(
                    "dealer {} owns {} input wires but dealt only {} sharings",
                    party,
                    expected,
                    shares.len()
                );
            }
            for share in shares[..expected].iter() {
                self.wires[wire] = Some(share.clone());
                wire += 1;
            }
        }

        log::info!(
            "BristolCircuit: bound {} input wires from {} dealers",
            wire,
            dealers.len()
        );
        self.inputs_assembled = true;
        // The per-dealer sharings have been copied onto the wires and have no
        // other reader. Dropping them is safe precisely because
        // `inputs_assembled` is now set: `input_sharing_termination` bails on
        // that flag before it would insert a late dealer back into this map, so
        // an emptied map can never be mistaken for "still waiting for dealers".
        self.input_wire_sharings.clear();
        Ok(())
    }

    /// Start the circuit as soon as both the input wires and the preprocessing
    /// material are in hand — the two arrive in either order.
    fn try_start_circuit(&mut self) -> Result<DepthInput<F>> {
        if self.circuit_started || !self.inputs_assembled || !self.preprocessing_done {
            return Ok(DepthInput::Waiting);
        }
        self.circuit_started = true;
        // Level 0 holds the linear gates that read only input wires; they are
        // evaluable before any multiplication has run.
        self.evaluate_linear_gates(0)?;
        self.schedule_level(1)
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
            let (Some(left), Some(right)) = (
                self.wires[gate.input_left].clone(),
                self.wires[gate.input_right].clone(),
            ) else {
                bail!(
                    "linear gate writing wire {} at level {} reads an unwritten wire",
                    gate.output,
                    level
                );
            };
            self.wires[gate.output] = Some(left + right);
        }
        if !gates.is_empty() {
            log::info!(
                "BristolCircuit: evaluated {} linear gates unblocked by level {}",
                gates.len(),
                level
            );
        }
        Ok(())
    }

    /// Build the multiplication batch for `level`, or — once the last level is
    /// done — hand back the circuit's output sharings.
    fn schedule_level(&mut self, level: usize) -> Result<DepthInput<F>> {
        if level > self.circuit.multiplicative_depth() {
            return self.finish();
        }
        let Some(depth) = self.circuit.level(level) else {
            bail!("no level {} in the circuit", level);
        };

        // Every multiplication gate at this level goes into one batch, so the
        // level costs a single round however wide it is.
        let mut x = Vec::with_capacity(depth.mult_gates.len());
        let mut y = Vec::with_capacity(depth.mult_gates.len());
        let mut mul_outputs = Vec::with_capacity(depth.mult_gates.len());

        for gate in depth.mult_gates.iter() {
            let (Some(left), Some(right)) = (
                self.wires[gate.input_left].clone(),
                self.wires[gate.input_right].clone(),
            ) else {
                bail!(
                    "MUL gate writing wire {} at level {} reads an unwritten wire",
                    gate.output,
                    level
                );
            };
            x.push(left);
            y.push(right);
            mul_outputs.push(gate.output);
        }

        log::info!(
            "BristolCircuit: scheduling level {}/{} — {} multiplication gates in one batch",
            level,
            self.circuit.multiplicative_depth(),
            mul_outputs.len()
        );

        // Results come back in the order the gates were pushed.
        self.scheduled_outputs = mul_outputs;

        DepthInput::multiply(level, x, y)
    }

    /// Write a level's multiplication results onto their output wires.
    fn apply_mult_results(&mut self, level: usize, results: Vec<FieldElement<F>>) -> Result<()> {
        let outputs = std::mem::take(&mut self.scheduled_outputs);
        if outputs.len() != results.len() {
            bail!(
                "level {} scheduled {} gates but {} results came back",
                level,
                outputs.len(),
                results.len()
            );
        }
        for (wire, result) in outputs.into_iter().zip(results.into_iter()) {
            self.wires[wire] = Some(result);
        }
        Ok(())
    }

    /// Collect the circuit's output sharings, in the order the file lists them.
    fn finish(&mut self) -> Result<DepthInput<F>> {
        let mut outputs = Vec::with_capacity(self.circuit.num_outputs());
        for wire in self.circuit.output_wires().iter() {
            match self.wires[*wire].clone() {
                Some(value) => outputs.push(value),
                None => bail!("output wire {} was never written", wire),
            }
        }
        log::info!(
            "BristolCircuit: circuit complete after {} multiplication levels, {} output wires",
            self.circuit.multiplicative_depth(),
            outputs.len()
        );
        // The wire sharings are dead now that the outputs have been copied out;
        // a large circuit holds one field element per wire.
        self.wires = Vec::new();
        Ok(DepthInput::Done(outputs))
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for BristolCircuit<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        // One multiplication, and so one mask, per MUL gate — reported level by
        // level so the engine can reserve each level a fixed slice of the
        // preprocessing pool. Level 0 is skipped: it holds only linear gates,
        // which cost neither a round nor a mask.
        //
        // Every party parses the same circuit file and so declares the same
        // profile, which is what makes the reservation agree across parties.
        let gates_per_depth: Vec<usize> = self
            .circuit
            .levels()
            .iter()
            .skip(1)
            .map(|level| level.num_mul_gates())
            .collect();

        // No random wires: the circuit has no gate that consumes one, so
        // `random_wires` is left at its default. Comparison gates would be the
        // first (see issue #5), and would declare their solved-bit demand there.
        let counts = PreprocessingCounts::new(gates_per_depth, self.circuit.num_outputs());
        log::info!(
            "BristolCircuit::preprocessing_count -> mult_gates={}, depth={}, output={}",
            counts.mult_gates(),
            counts.depth(),
            counts.output
        );
        counts
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        self.generate_input_sharings()
    }

    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        if self.inputs_assembled {
            log::debug!(
                "BristolCircuit: ignoring input sharing from party {}, input wires already bound",
                party
            );
            return Ok(DepthInput::Waiting);
        }
        if self.circuit.inputs_of_party(party) == 0 {
            log::debug!(
                "BristolCircuit: party {} supplies no input wires in this circuit; ignoring its {} \
                 sharings",
                party,
                shares.len()
            );
            return Ok(DepthInput::Waiting);
        }
        log::info!(
            "BristolCircuit: input sharing from party {} terminated with {} sharings",
            party,
            shares.len()
        );
        self.input_wire_sharings.insert(party, shares);
        self.try_assemble_input_wires()?;
        self.try_start_circuit()
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        log::info!(
            "BristolCircuit: preprocessing complete — {} random bits (unused); circuit: {} \
             multiplications over {} levels",
            wires.bits.len(),
            self.circuit.num_mul_gates(),
            self.circuit.multiplicative_depth(),
        );
        self.preprocessing_done = true;
        self.try_start_circuit()
    }

    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!(
            "BristolCircuit: multiplication at level {} complete — {} results",
            depth,
            results.len(),
        );
        // The engine may replay a level's termination; levels advance by one and
        // only forward, so anything else is a replay or an out-of-order delivery.
        if depth != self.levels_completed + 1 {
            log::warn!(
                "BristolCircuit: ignoring completion of level {}; the circuit is at level {}",
                depth,
                self.levels_completed
            );
            return Ok(DepthInput::Waiting);
        }
        self.apply_mult_results(depth, results)?;
        self.levels_completed = depth;
        self.evaluate_linear_gates(depth)?;
        self.schedule_level(depth + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::evaluate_circuit;
    use crate::{parse_circuit, parse_circuit_file};

    /// The tests exercise the application at one concrete field; the generic
    /// parameter is what the engine binds, not something the tests vary.
    type F = velox::fields::DefaultField;

    const NUM_NODES: usize = 10;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/circuits")
            .join(name)
    }

    fn elem(value: u64) -> FieldElement<F> {
        FieldElement::<F>::from(value)
    }

    /// A stand-in engine.
    ///
    /// It drives the application through the same hook sequence `mpc::Context`
    /// does — preprocessing, then one multiplication batch per level, then the
    /// output handover — but evaluates each batch in the clear rather than
    /// running the protocol. Because linear operations on Shamir sharings are
    /// the same operations on the secrets, feeding secrets in where the engine
    /// would feed shares makes the outputs directly comparable to
    /// `evaluate_circuit`.
    struct Harness {
        app: BristolCircuit<F>,
        /// Batch sizes the application scheduled, level by level.
        batches: Vec<usize>,
    }

    impl Harness {
        fn new(circuit: Circuit) -> Self {
            let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap();
            Self {
                app,
                batches: Vec::new(),
            }
        }

        fn from_fixture(name: &str) -> Self {
            Self::new(parse_circuit_file(fixture(name)).unwrap())
        }

        /// The engine hands over random bits only; the multiplication masks stay
        /// in its own pool and are drawn per batch.
        async fn deliver_preprocessing(&mut self) -> DepthInput<F> {
            self.app.on_preprocessing_complete(RandomWireShares::empty()).await.unwrap()
        }

        /// Hand the application every dealer's inputs, sliced out of the circuit's
        /// input-wire vector by the header's positional layout.
        async fn deliver_inputs(&mut self, inputs: &[FieldElement<F>]) -> DepthInput<F> {
            let circuit = self.app.circuit().clone();
            let mut wire = 0;
            let mut last = DepthInput::Waiting;
            for party in 0..circuit.num_input_parties() {
                let count = circuit.inputs_of_party(party);
                let shares = inputs[wire..wire + count].to_vec();
                wire += count;
                last = self
                    .app
                    .input_sharing_termination(party, shares)
                    .await
                    .unwrap();
            }
            last
        }

        /// Run every level the application schedules, and return the output
        /// sharings it hands back at the end. Each batch names its own depth,
        /// which is what the engine reads it back as.
        async fn run(&mut self, mut depth_input: DepthInput<F>) -> Vec<FieldElement<F>> {
            loop {
                match depth_input {
                    DepthInput::Done(outputs) => return outputs,
                    DepthInput::Waiting => panic!("the application stalled with nothing scheduled"),
                    DepthInput::Multiply { depth, x, y } => {
                        self.batches.push(x.len());
                        let results: Vec<FieldElement<F>> =
                            x.into_iter().zip(y.into_iter()).map(|(x, y)| x * y).collect();
                        depth_input = self.app.on_depth_complete(depth, results).await.unwrap();
                    }
                }
            }
        }

        /// Preprocessing, inputs, every level, outputs.
        async fn evaluate(&mut self, inputs: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
            let after_preprocessing = self.deliver_preprocessing().await;
            assert!(
                after_preprocessing.is_waiting(),
                "the circuit must wait for its input wires before starting"
            );
            let started = self.deliver_inputs(inputs).await;
            self.run(started).await
        }
    }

    /// Every fixture evaluated through the application's hooks must agree with
    /// the same circuit evaluated in the clear, and must cost exactly one
    /// multiplication batch per multiplicative level.
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
            let inputs: Vec<FieldElement<F>> = (1..=circuit.num_inputs() as u64)
                .map(|value| elem(value * 7 + 3))
                .collect();
            let expected = evaluate_circuit::<F>(&circuit, &inputs).unwrap();

            let mut harness = Harness::from_fixture(name);
            let outputs = harness.evaluate(&inputs).await;

            assert_eq!(outputs, expected, "{}: outputs disagree with cleartext evaluation", name);
            assert_eq!(
                harness.batches.len(),
                circuit.multiplicative_depth(),
                "{}: one multiplication batch per multiplicative level",
                name
            );
        }
    }

    /// `f(x) = a·x² + b·x + c` in two rounds, not four: level 1 batches `x²` and
    /// `b·x` together, level 2 runs `a·x²` and folds in both additions. This is
    /// the scheduling half of the levelisation fix — the parser-side half is in
    /// `circuit::parser::tests::polynomial_eval_is_two_deep_not_four`.
    #[tokio::test]
    async fn polynomial_eval_runs_two_batches() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        // a = 3, b = 5, c = 7 from party 0; x = 11 from party 1.
        let inputs = vec![elem(3), elem(5), elem(7), elem(11)];

        let mut harness = Harness::new(circuit);
        let outputs = harness.evaluate(&inputs).await;

        assert_eq!(outputs, vec![elem(3 * 121 + 5 * 11 + 7)]);
        assert_eq!(harness.batches, vec![2, 1], "two gates in the first round, one in the second");
    }

    /// Independent gates at the same level share one round, however many there
    /// are — the batching is where the protocol's efficiency comes from.
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
        let expected = evaluate_circuit::<F>(&circuit, &inputs).unwrap();

        let mut harness = Harness::new(circuit);
        let outputs = harness.evaluate(&inputs).await;

        // 2·3 + 5·7 = 6 + 35 = 41.
        assert_eq!(outputs, vec![elem(41)]);
        assert_eq!(outputs, expected);
        assert_eq!(harness.batches, vec![2], "two gates, one round");
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
            _ => panic!("a linear circuit must finish without a multiplication round"),
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

    /// Preprocessing is sized off the multiplication gates and the multiplicative
    /// depth. For `polynomial_eval` that is 3 multiplications over 2 levels — not
    /// the 4 depths the unported parser reported.
    #[test]
    fn preprocessing_counts_follow_multiplicative_depth() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit).unwrap();

        let counts = app.preprocessing_count();

        assert_eq!(counts.depth(), 2, "multiplicative depth, not gate depth");
        assert_eq!(counts.mult_gates(), 3, "one mask per multiplication gate");
        assert_eq!(
            counts.gates_per_depth,
            vec![2, 1],
            "two gates at level 1, one at level 2 — the engine reserves each its own slice"
        );
        assert_eq!(counts.output, 1);
        assert_eq!(app.random_wires().bits, 0, "no gate consumes a random bit yet");
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
        assert!(
            party_four.inputs().await.is_empty(),
            "a party the circuit gives no input wires deals nothing"
        );
    }

    /// A short input list is padded rather than rejected, so a party can run the
    /// circuit without a complete input file.
    #[test]
    fn short_input_lists_are_padded() {
        let circuit = parse_circuit_file(fixture("polynomial_eval.arith")).unwrap();
        let app = BristolCircuit::<F>::new(NUM_NODES, 0, circuit)
            .unwrap()
            .with_inputs(vec![elem(3)]);

        let sharings = app.generate_input_sharings();

        assert_eq!(sharings.len(), 3);
        assert_eq!(sharings[0], elem(3));
    }

    /// A replayed level termination must not re-run the level or advance the
    /// circuit past it.
    #[tokio::test]
    async fn replayed_level_termination_is_ignored() {
        let mut harness = Harness::from_fixture("multiply_three.arith");
        let after_preprocessing = harness.deliver_preprocessing().await;
        assert!(after_preprocessing.is_waiting());

        let started = harness.deliver_inputs(&[elem(2), elem(3), elem(5)]).await;
        let DepthInput::Multiply { depth, x, y } = started else {
            panic!("level 1 must be scheduled");
        };
        assert_eq!(depth, 1, "the application names the depth, not the engine");
        let results: Vec<FieldElement<F>> =
            x.into_iter().zip(y.into_iter()).map(|(x, y)| x * y).collect();

        let level_two = harness
            .app
            .on_depth_complete(1, results.clone())
            .await
            .unwrap();
        assert!(!level_two.is_waiting(), "level 2 follows level 1");

        let replay = harness.app.on_depth_complete(1, results).await.unwrap();
        assert!(replay.is_waiting(), "a replayed level 1 must schedule nothing");
    }

    /// An application error reaches the engine as an `Err` rather than as a
    /// `Waiting` the engine would sit on forever. Under the old API both
    /// arrived as an empty `DepthInput` and the protocol simply hung.
    #[tokio::test]
    async fn a_broken_result_batch_is_reported_not_swallowed() {
        let mut harness = Harness::from_fixture("multiply_three.arith");
        let _ = harness.deliver_preprocessing().await;
        let started = harness.deliver_inputs(&[elem(2), elem(3), elem(5)]).await;
        assert!(!started.is_waiting());

        // Level 1 scheduled one gate; hand back two results.
        let error = harness
            .app
            .on_depth_complete(1, vec![elem(1), elem(2)])
            .await
            .expect_err("a mismatched result count must be an error")
            .to_string();
        assert!(error.contains("scheduled 1 gates but 2 results"), "got {:?}", error);
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
