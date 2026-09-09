//! Anonymous broadcast via a butterfly mixing network.
//!
//! This is the Velox port of the anonymous-broadcast application: the same
//! circuit the engine used to hard-code in `mpc/src/protocol/online_phase`
//! (`init_mixing` / `init_butterfly_mixing_level` /
//! `verify_mixing_level_termination`), lifted out of the engine and expressed
//! against the [`Application`] trait.
//!
//! Circuit:
//!   - `k` input wires are shuffled through `log_k²` depths of butterfly switches.
//!   - At each depth, wires at distance `switch_index = 2^((log_k - depth mod log_k) mod log_k)`
//!     are paired, giving `k/2` switches.
//!   - Each switch multiplies `(w1 − w2)` by a random bit, then computes
//!       `out1 = (w1 + w2 + product) / 2`,  `out2 = (w1 + w2 − product) / 2`.
//!   - The outputs of a depth are the inputs of the next; after the last depth
//!     the wires are handed back for output reconstruction.
//!
//! Differences from the Talos version of this application:
//!   - **No network routing.** Velox has no network-routing module, so the
//!     transpose-routing preprocessing and the `F_SELECT` plumbing are gone.
//!   - **No packing / SIMD mode.** A Velox sharing carries a single secret, so
//!     one wire is one sharing and the batch/`log_batch` bookkeeping disappears.
//!   - **No preprocessing bookkeeping.** The engine draws the masks each depth's
//!     batch consumes; this application only asks for random bits, which it does
//!     consume itself, one per switch.

use std::collections::{HashMap, VecDeque};

use anyhow::{bail, Result};
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

use crate::{Application, DepthInput, PreprocessingCounts};

pub struct AnonymousBroadcast<F: ProtocolField> {
    pub num_nodes: usize,
    pub num_faults: usize,
    pub my_id: usize,

    /// Number of wires (anonymity set size). Must be a power of two.
    k_value: usize,
    /// log2(k_value)
    log_k: usize,
    /// Total circuit depth = log_k²
    max_depth: usize,
    /// Circuit depth currently in flight (1-indexed; 0 before the first depth).
    current_depth: usize,

    /// Values this party secret-shares into the mixing network. Populated by
    /// [`AnonymousBroadcast::with_inputs`]; random values are generated when unset.
    my_inputs: Option<Vec<FieldElement<F>>>,
    /// Input sharings received from each dealer, keyed by dealer.
    input_wire_sharings: HashMap<usize, Vec<FieldElement<F>>>,
    /// Dealers whose input sharings make up the circuit's input wires. `None`
    /// until the engine reports the agreed-upon set (see
    /// [`AnonymousBroadcast::set_input_party_set`]), in which case every party
    /// is expected to deal.
    input_party_set: Option<Vec<usize>>,
    /// Set once the input wires have been assembled, so late dealers are ignored.
    inputs_assembled: bool,
    /// Set once depth 1 has been scheduled, so the circuit starts exactly once.
    circuit_started: bool,

    /// Random bit sharings, consumed one per wire pair per depth.
    rand_bits: VecDeque<FieldElement<F>>,
    /// Set once preprocessing has been handed over.
    preprocessing_done: bool,

    /// Wire sharings per depth: `wire_sharings[d]` are the `k` wires entering depth `d`.
    wire_sharings: HashMap<usize, Vec<FieldElement<F>>>,
    /// Wire-pair sums `(w1 + w2)` saved per depth, needed once the
    /// multiplication result for that depth arrives to finish the switch.
    wire_pair_sums: HashMap<usize, Vec<FieldElement<F>>>,

    /// Precomputed multiplicative inverse of 2 in the field.
    two_inverse: FieldElement<F>,
}

impl<F: ProtocolField> AnonymousBroadcast<F> {
    pub fn new(num_nodes: usize, num_faults: usize, my_id: usize, k_value: usize) -> Self {
        assert!(k_value.is_power_of_two(), "k_value must be a power of two");
        assert!(k_value >= 2, "k_value must be at least 2");
        let log_k = k_value.trailing_zeros() as usize;
        Self {
            num_nodes,
            num_faults,
            my_id,
            k_value,
            log_k,
            max_depth: log_k * log_k,
            current_depth: 0,
            my_inputs: None,
            input_wire_sharings: HashMap::new(),
            input_party_set: None,
            inputs_assembled: false,
            circuit_started: false,
            rand_bits: VecDeque::new(),
            preprocessing_done: false,
            wire_sharings: HashMap::new(),
            wire_pair_sums: HashMap::new(),
            two_inverse: FieldElement::<F>::from(2u64).inv().unwrap(),
        }
    }

    /// Supply the values this party feeds into the mixing network. Without this
    /// the application shares random field elements, which is enough to exercise
    /// the circuit but carries no message.
    pub fn with_inputs(mut self, inputs: Vec<FieldElement<F>>) -> Self {
        self.my_inputs = Some(inputs);
        // party set with inputs
        let parties: Vec<usize> = (0..self.num_nodes).into_iter().collect();
        self.set_input_party_set(parties);
        self
    }

    /// Tell the application which dealers' input sharings form the circuit's
    /// input wires. Every party must be given the *same* set — the wires are
    /// assembled in the order given here, so a disagreement would leave parties
    /// mixing different wire orders. Velox agrees on this set through the ACS
    /// instance that gates preprocessing; until the engine passes it on, the
    /// application waits for all `num_nodes` dealers.
    pub fn set_input_party_set(&mut self, parties: Vec<usize>) {
        self.input_party_set = Some(parties);
    }

    /// Number of input values this party deals. The circuit needs `k` wires and
    /// only `n - t` dealers are guaranteed to terminate, so each party deals
    /// `k / (n - t) + 1` values — the same sizing the engine uses today.
    pub fn inputs_per_party(&self) -> usize {
        (self.k_value / (self.num_nodes - self.num_faults)) + 1
    }

    /// The secrets this party deals, padded with random values if the supplied
    /// list is short.
    pub fn generate_input_sharings(&self) -> Vec<FieldElement<F>> {
        let num_inputs = self.inputs_per_party();
        let mut values = self.my_inputs.clone().unwrap_or_default();
        if values.len() < num_inputs {
            log::info!(
                "AnonymousBroadcast: {} inputs supplied, padding with {} random values",
                values.len(),
                num_inputs - values.len()
            );
            values.extend((values.len()..num_inputs).map(|_| F::rand()));
        }
        values.truncate(num_inputs);
        values
    }

    /// Assemble the circuit's input wires once every expected dealer's input
    /// ACSS has terminated. Wires are laid out dealer by dealer in the order of
    /// the agreed party set and truncated to `k`.
    fn try_assemble_input_wires(&mut self) -> Result<()> {
        if self.inputs_assembled {
            return Ok(());
        }
        let parties: Vec<usize> = match self.input_party_set.as_ref() {
            Some(parties) => parties.clone(),
            None => (0..self.num_nodes).collect(),
        };
        if !parties
            .iter()
            .all(|party| self.input_wire_sharings.contains_key(party))
        {
            return Ok(());
        }

        let mut input_sharings: Vec<FieldElement<F>> = Vec::new();
        for party in parties.iter() {
            input_sharings.extend(self.input_wire_sharings[party].iter().cloned());
        }
        if input_sharings.len() < self.k_value {
            bail!(
                "not enough input sharings for mixing: need at least {}, got {}",
                self.k_value,
                input_sharings.len()
            );
        }
        input_sharings.truncate(self.k_value);

        log::info!(
            "AnonymousBroadcast: assembled {} input wires from {} dealers",
            input_sharings.len(),
            parties.len()
        );
        self.wire_sharings.insert(1, input_sharings);
        self.inputs_assembled = true;
        // The per-dealer sharings have been concatenated into wire vector 1 and
        // have no other reader. Dropping them here is safe precisely because
        // `inputs_assembled` is now set: `input_sharing_termination` bails on
        // that flag before it would insert a late dealer back into this map, so
        // an emptied map can never be mistaken for "still waiting for dealers".
        self.input_wire_sharings.clear();
        Ok(())
    }

    /// Schedule depth 1 as soon as both the input wires and the preprocessing
    /// material are in hand — the two arrive in either order.
    fn try_start_circuit(&mut self) -> Result<DepthInput<F>> {
        if self.circuit_started || !self.inputs_assembled || !self.preprocessing_done {
            return Ok(DepthInput::Waiting);
        }
        self.circuit_started = true;
        self.init_butterfly_level(1)
    }

    /// Compute the butterfly pairing for `depth` and return the multiplication
    /// batch it needs: wire differences × random bits.
    ///
    /// Mirrors the engine's `init_butterfly_mixing_level`.
    fn init_butterfly_level(&mut self, depth: usize) -> Result<DepthInput<F>> {
        let Some(wires) = self.wire_sharings.get(&depth) else {
            log::warn!(
                "AnonymousBroadcast: wire sharings for depth {} not available yet",
                depth
            );
            return Ok(DepthInput::Waiting);
        };

        let log_switch_index = ((self.log_k - (depth % self.log_k)) % self.log_k) as u32;
        let switch_index = usize::pow(2, log_switch_index);

        // Butterfly pairing read straight off the wire vector. A bitmap answers
        // "is this wire still unpaired" in one byte per wire, where the previous
        // version copied every wire into a `HashMap` and drained it with `remove`
        // to answer the same question.
        let mut paired = vec![false; wires.len()];
        let mut sums = Vec::new();
        let mut diffs = Vec::new();
        for i in 0..self.k_value {
            let j = i + switch_index;
            if j < wires.len() && !paired[i] && !paired[j] {
                paired[i] = true;
                paired[j] = true;
                let w1 = &wires[i];
                let w2 = &wires[j];
                sums.push(w1.clone() + w2.clone());
                diffs.push(w1.clone() - w2.clone());
            }
        }

        log::info!(
            "AnonymousBroadcast: depth {} — {} wire pairs (switch_index={}), {} wires left unpaired",
            depth,
            diffs.len(),
            switch_index,
            paired.iter().filter(|is_paired| !**is_paired).count()
        );

        let num_switches = diffs.len();
        if self.rand_bits.len() < num_switches {
            bail!(
                "depth {} needs {} random bits, {} left",
                depth,
                num_switches,
                self.rand_bits.len()
            );
        }
        let bits: Vec<FieldElement<F>> = (0..num_switches)
            .map(|_| self.rand_bits.pop_front().unwrap())
            .collect();

        self.current_depth = depth;
        self.wire_pair_sums.insert(depth, sums);

        // This depth's input wires are dead: they have been folded into the
        // `sums` recorded above and the `diffs` handed to multiplication, and
        // every remaining reader works off those two.
        //
        // The entry is emptied, not removed: `handle_mult_results` dedupes a
        // replayed depth termination by testing `wire_sharings.contains_key`,
        // so removing it would let a replay re-run the depth.
        if let Some(consumed_wires) = self.wire_sharings.get_mut(&depth) {
            *consumed_wires = Vec::new();
        }

        DepthInput::multiply(depth, diffs, bits)
    }

    /// Apply the butterfly switch to a depth's multiplication results and
    /// schedule whatever comes next: the following depth, or — at the last
    /// depth — the output sharings.
    ///
    /// Mirrors the engine's `verify_mixing_level_termination`.
    fn handle_mult_results(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        if self.wire_sharings.contains_key(&(depth + 1)) {
            // Already processed — the engine may replay a depth's termination.
            return Ok(DepthInput::Waiting);
        }
        let Some(sums) = self.wire_pair_sums.remove(&depth) else {
            bail!("no wire pair sums recorded for depth {}", depth);
        };
        if sums.len() != results.len() {
            bail!(
                "depth {} returned {} multiplication results for {} wire pairs",
                depth,
                results.len(),
                sums.len()
            );
        }

        let two_inverse = &self.two_inverse;
        let next_depth_wires: Vec<FieldElement<F>> = sums
            .into_iter()
            .zip(results.into_iter())
            .flat_map(|(sum, product)| {
                let wire1 = (sum.clone() + product.clone()) * two_inverse.clone();
                let wire2 = (sum - product) * two_inverse.clone();
                [wire1, wire2]
            })
            .collect();

        if depth >= self.max_depth {
            log::info!(
                "AnonymousBroadcast: mixing circuit complete after {} depths, {} output wires",
                self.max_depth,
                next_depth_wires.len()
            );
            // Record the key so a replayed termination of the last depth is
            // deduped, but not the payload: no depth reads wire vector
            // `max_depth + 1`, so the stored copy would be write-only. The wires
            // themselves are moved into the output below.
            self.wire_sharings.insert(depth + 1, Vec::new());
            return Ok(DepthInput::Done(next_depth_wires));
        }

        self.wire_sharings.insert(depth + 1, next_depth_wires);
        self.init_butterfly_level(depth + 1)
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for AnonymousBroadcast<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        // Every depth pairs the k wires into k/2 switches, one multiplication and
        // one random bit each — the same shape at every depth, and the same at
        // every party, which is what lets the engine reserve each depth a fixed
        // slice of the preprocessing pool.
        let switches_per_depth = self.k_value / 2;
        let counts = PreprocessingCounts::new(
            vec![switches_per_depth; self.max_depth],
            switches_per_depth * self.max_depth,
            self.k_value,
        );
        log::info!(
            "AnonymousBroadcast::preprocessing_count -> mult_gates={}, depth={}, rand_bits={}, output={}",
            counts.mult_gates(),
            counts.depth(),
            counts.rand_bits,
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
                "AnonymousBroadcast: ignoring input sharing from party {}, wires already assembled",
                party
            );
            return Ok(DepthInput::Waiting);
        }
        log::info!(
            "AnonymousBroadcast: input sharing from party {} terminated with {} sharings",
            party,
            shares.len()
        );
        self.input_wire_sharings.insert(party, shares);
        self.try_assemble_input_wires()?;
        self.try_start_circuit()
    }

    async fn on_preprocessing_complete(
        &mut self,
        rand_bit_sharings: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!(
            "AnonymousBroadcast: preprocessing complete — {} random bits; circuit: k={}, log_k={}, max_depth={}",
            rand_bit_sharings.len(),
            self.k_value,
            self.log_k,
            self.max_depth,
        );
        self.rand_bits.extend(rand_bit_sharings);
        self.preprocessing_done = true;
        self.try_start_circuit()
    }

    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!(
            "AnonymousBroadcast: multiplication at depth {} complete — {} results",
            depth,
            results.len(),
        );
        self.handle_mult_results(depth, results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests exercise the application at one concrete field; the generic
    /// parameter is what the engine binds, not something the tests vary.
    type F = fields::DefaultField;

    fn app(k: usize) -> AnonymousBroadcast<F> {
        AnonymousBroadcast::new(9, 2, 0, k)
    }

    /// The depth and operands of a scheduled batch, or a panic naming what came
    /// instead.
    fn expect_multiply(
        depth_input: DepthInput<F>,
    ) -> (usize, Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        match depth_input {
            DepthInput::Multiply { depth, x, y } => (depth, x, y),
            DepthInput::Waiting => panic!("expected a multiplication batch, got Waiting"),
            DepthInput::Done(_) => panic!("expected a multiplication batch, got Done"),
        }
    }

    #[test]
    fn circuit_dimensions() {
        let app = app(8);
        assert_eq!(app.k_value, 8);
        assert_eq!(app.log_k, 3);
        assert_eq!(app.max_depth, 9);
        assert_eq!(app.current_depth, 0);
    }

    #[test]
    fn preprocessing_counts_cover_every_switch() {
        let app = app(16);
        // k=16, log_k=4, max_depth=16, 8 switches per depth.
        let counts = app.preprocessing_count();
        assert_eq!(counts.gates_per_depth, vec![8; 16], "8 switches at each of 16 depths");
        assert_eq!(counts.mult_gates(), 8 * 16);
        assert_eq!(counts.rand_bits, 8 * 16);
        assert!(counts.output >= 16);
        assert_eq!(counts.depth(), 16);
    }

    /// With every random bit equal to one, each switch is the identity:
    ///   product = (w1 − w2) · 1,  out1 = w1,  out2 = w2.
    /// Depth 1 of a k=4 circuit pairs (0,2) and (1,3), so the wires come back
    /// as [w0, w2, w1, w3].
    #[test]
    fn butterfly_switch_is_identity_on_one_bits() {
        let mut app = app(4);
        // Truncate the circuit to a single depth, so depth 1's switch output is
        // handed straight back instead of being folded into depth 2 — which
        // consumes it, the wire vector being freed as soon as a depth starts.
        app.max_depth = 1;
        let wires: Vec<FieldElement<F>> =
            (1..=4u64).map(|x| FieldElement::<F>::from(x * 10)).collect();
        app.wire_sharings.insert(1, wires.clone());
        app.rand_bits = (0..16).map(|_| FieldElement::<F>::one()).collect();

        let (depth, diffs, bits) = expect_multiply(app.init_butterfly_level(1).unwrap());
        assert_eq!(depth, 1, "the application names the depth its batch belongs to");
        assert_eq!(diffs.len(), 2);
        assert_eq!(bits.len(), 2);

        // bit = 1, so the multiplication output equals the wire differences and
        // every switch is the identity. Depth 1 of a k=4 circuit pairs (0,2) and
        // (1,3), so the wires come back as [w0, w2, w1, w3].
        let next = match app.handle_mult_results(1, diffs).unwrap() {
            DepthInput::Done(next) => next,
            other => panic!("expected the truncated circuit to finish, got {:?}", other),
        };

        assert_eq!(next[0], wires[0]);
        assert_eq!(next[1], wires[2]);
        assert_eq!(next[2], wires[1]);
        assert_eq!(next[3], wires[3]);
    }

    #[test]
    fn last_depth_returns_output_sharings() {
        let mut app = app(4);
        // k=4, log_k=2, max_depth=4.
        app.current_depth = app.max_depth;
        app.wire_pair_sums.insert(
            app.max_depth,
            vec![FieldElement::<F>::from(2u64), FieldElement::<F>::from(4u64)],
        );

        let depth_input = app
            .handle_mult_results(
                app.max_depth,
                vec![FieldElement::<F>::zero(), FieldElement::<F>::zero()],
            )
            .unwrap();

        match depth_input {
            DepthInput::Done(outputs) => assert_eq!(outputs.len(), 4),
            _ => panic!("the last depth must return output sharings"),
        }
    }

    /// A short random-bit pool is an error, not a silent stall: the engine can
    /// report it rather than hanging while every party waits for a depth that
    /// will never be scheduled.
    #[test]
    fn a_short_random_bit_pool_is_reported() {
        let mut app = app(4);
        app.wire_sharings
            .insert(1, (1..=4u64).map(|x| FieldElement::<F>::from(x)).collect());
        app.rand_bits = VecDeque::new();

        let error = app.init_butterfly_level(1).unwrap_err().to_string();
        assert!(error.contains("random bits"), "got {:?}", error);
    }
}
