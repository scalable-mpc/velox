//! Anonymous broadcast via a butterfly mixing network.
//!
//! This is the Velox port of the anonymous-broadcast application: the same
//! circuit the engine currently hard-codes in `mpc/src/protocol/online_phase`
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
//!     transpose-routing preprocessing, the `F_SELECT` plumbing and the routing
//!     hooks are gone. [`Application::on_network_routing_complete`] is
//!     implemented only because the trait requires it, and never fires.
//!   - **No packing / SIMD mode.** A Velox sharing carries a single secret, so
//!     there is no first-half/second-half packing: one wire is one sharing, the
//!     second-half half of every `(first_half, second_half)` pair stays empty,
//!     and the batch/`log_batch` bookkeeping disappears with it.

use std::collections::{HashMap, VecDeque};

use async_trait::async_trait;
use fields::rand_field_element;

use crate::{
    Application, DepthInput, MultInput, Multiplication, NetworkRoutingPreprocessing,
    PreprocessingCounts, SmallField,
};

pub struct AnonymousBroadcast {
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
    my_inputs: Option<Vec<SmallField>>,
    /// Input sharings received from each dealer, keyed by dealer.
    input_wire_sharings: HashMap<usize, Vec<SmallField>>,
    /// Dealers whose input sharings make up the circuit's input wires. `None`
    /// until the engine reports the agreed-upon set (see
    /// [`AnonymousBroadcast::set_input_party_set`]), in which case every party
    /// is expected to deal.
    input_party_set: Option<Vec<usize>>,
    /// Set once the input wires have been assembled, so late dealers are ignored.
    inputs_assembled: bool,
    /// Set once depth 1 has been scheduled, so the circuit starts exactly once.
    circuit_started: bool,

    /// Random sharings and random zero sharings the multiplication protocol
    /// consumes, split per depth.
    random_sharings: HashMap<usize, Vec<SmallField>>,
    zero_sharings: HashMap<usize, Vec<SmallField>>,
    /// Random bit sharings, consumed one per wire pair per depth.
    rand_bits: VecDeque<SmallField>,
    /// Set once preprocessing has been handed over.
    preprocessing_done: bool,

    /// Wire sharings per depth: `wire_sharings[d]` are the `k` wires entering depth `d`.
    wire_sharings: HashMap<usize, Vec<SmallField>>,
    /// Wire-pair sums `(w1 + w2)` saved per depth, needed once the
    /// multiplication result for that depth arrives to finish the switch.
    wire_pair_sums: HashMap<usize, Vec<SmallField>>,

    /// Precomputed multiplicative inverse of 2 in the field.
    two_inverse: SmallField,
}

impl AnonymousBroadcast {
    pub fn new(num_nodes: usize, num_faults: usize, my_id: usize, k_value: usize) -> Self {
        assert!(k_value.is_power_of_two(), "k_value must be a power of two");
        assert!(k_value >= 2, "k_value must be at least 2");
        let log_k = k_value.trailing_zeros() as usize;
        AnonymousBroadcast {
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
            random_sharings: HashMap::new(),
            zero_sharings: HashMap::new(),
            rand_bits: VecDeque::new(),
            preprocessing_done: false,
            wire_sharings: HashMap::new(),
            wire_pair_sums: HashMap::new(),
            two_inverse: SmallField::from(2u64).inv().unwrap(),
        }
    }

    /// Supply the values this party feeds into the mixing network. Without this
    /// the application shares random field elements, which is enough to exercise
    /// the circuit but carries no message.
    pub fn with_inputs(mut self, inputs: Vec<SmallField>) -> Self {
        self.my_inputs = Some(inputs);
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

    /// One sharing per input value: Velox sharings are unpacked, so every inner
    /// `Vec` holds exactly one secret.
    pub fn generate_input_sharings(&self) -> Vec<Vec<SmallField>> {
        let num_inputs = self.inputs_per_party();
        let mut values = match self.my_inputs.as_ref() {
            Some(inputs) => inputs.clone(),
            None => Vec::new(),
        };
        if values.len() < num_inputs {
            log::info!(
                "AnonymousBroadcast: {} inputs supplied, padding with {} random values",
                values.len(),
                num_inputs - values.len()
            );
            values.extend((values.len()..num_inputs).map(|_| rand_field_element()));
        }
        values.truncate(num_inputs);
        values.into_iter().map(|value| vec![value]).collect()
    }

    /// Assemble the circuit's input wires once every expected dealer's input
    /// ACSS has terminated. Wires are laid out dealer by dealer in the order of
    /// the agreed party set and truncated to `k`.
    fn try_assemble_input_wires(&mut self) {
        if self.inputs_assembled {
            return;
        }
        let parties: Vec<usize> = match self.input_party_set.as_ref() {
            Some(parties) => parties.clone(),
            None => (0..self.num_nodes).collect(),
        };
        if !parties
            .iter()
            .all(|party| self.input_wire_sharings.contains_key(party))
        {
            return;
        }

        let mut input_sharings: Vec<SmallField> = Vec::new();
        for party in parties.iter() {
            input_sharings.extend(self.input_wire_sharings[party].iter().cloned());
        }
        if input_sharings.len() < self.k_value {
            log::error!(
                "AnonymousBroadcast: not enough input sharings for mixing. Expected at least {}, got {}",
                self.k_value,
                input_sharings.len()
            );
            return;
        }
        input_sharings.truncate(self.k_value);

        log::info!(
            "AnonymousBroadcast: assembled {} input wires from {} dealers",
            input_sharings.len(),
            parties.len()
        );
        self.wire_sharings.insert(1, input_sharings);
        self.inputs_assembled = true;
    }

    /// Schedule depth 1 as soon as both the input wires and the preprocessing
    /// material are in hand — the two arrive in either order.
    async fn try_start_circuit(&mut self) -> DepthInput {
        if self.circuit_started || !self.inputs_assembled || !self.preprocessing_done {
            return DepthInput::empty();
        }
        self.circuit_started = true;
        self.init_butterfly_level(1).await
    }

    /// Compute the butterfly pairing for `depth` and return the multiplication
    /// batch it needs: wire differences × random bits.
    ///
    /// Mirrors the engine's `init_butterfly_mixing_level`.
    async fn init_butterfly_level(&mut self, depth: usize) -> DepthInput {
        let Some(wires) = self.wire_sharings.get(&depth) else {
            log::warn!(
                "AnonymousBroadcast: wire sharings for depth {} not available yet",
                depth
            );
            return DepthInput::empty();
        };

        let mut wire_index_map: HashMap<usize, SmallField> = wires
            .iter()
            .enumerate()
            .map(|(i, wire)| (i, wire.clone()))
            .collect();

        let log_switch_index = ((self.log_k - (depth % self.log_k)) % self.log_k) as u32;
        let switch_index = usize::pow(2, log_switch_index);

        let mut sums = Vec::new();
        let mut diffs = Vec::new();
        for i in 0..self.k_value {
            if wire_index_map.contains_key(&i) && wire_index_map.contains_key(&(i + switch_index)) {
                let w1 = wire_index_map.remove(&i).unwrap();
                let w2 = wire_index_map.remove(&(i + switch_index)).unwrap();
                sums.push(w1.clone() + w2.clone());
                diffs.push(w1 - w2);
            }
        }

        log::info!(
            "AnonymousBroadcast: depth {} — {} wire pairs (switch_index={}), {} wires left unpaired",
            depth,
            diffs.len(),
            switch_index,
            wire_index_map.len()
        );

        // Check every piece of preprocessing this depth consumes before touching
        // any of it, so a short pool cannot leave the state half-consumed.
        let num_switches = diffs.len();
        if self.rand_bits.len() < num_switches {
            log::warn!(
                "AnonymousBroadcast: cannot start depth {} — need {} random bits, have {}",
                depth,
                num_switches,
                self.rand_bits.len()
            );
            return DepthInput::empty();
        }
        let (Some(rand_sharings), Some(zero_sharings)) = (
            self.random_sharings.get(&depth),
            self.zero_sharings.get(&depth),
        ) else {
            log::warn!(
                "AnonymousBroadcast: cannot start depth {} — no random/zero sharings allocated for it",
                depth
            );
            return DepthInput::empty();
        };
        // The multiplication protocol consumes one random sharing per gate and,
        // in its linear variant, roughly one zero sharing per two gates.
        if rand_sharings.len() < num_switches || zero_sharings.len() < num_switches / 2 {
            log::warn!(
                "AnonymousBroadcast: cannot start depth {} — {} gates need {} random and {} zero sharings, have {} and {}",
                depth,
                num_switches,
                num_switches,
                num_switches / 2,
                rand_sharings.len(),
                zero_sharings.len()
            );
            return DepthInput::empty();
        }

        let bits: Vec<SmallField> = (0..num_switches)
            .map(|_| self.rand_bits.pop_front().unwrap())
            .collect();

        // Velox sharings are unpacked: only the first-half representation is used.
        let mult_input = match MultInput::new((diffs, Vec::new()), (bits, Vec::new()), depth) {
            Ok(mult_input) => mult_input,
            Err(err) => {
                log::warn!(
                    "AnonymousBroadcast: cannot build multiplication at depth {}: {:?}",
                    depth,
                    err
                );
                return DepthInput::empty();
            }
        };

        self.current_depth = depth;
        self.wire_pair_sums.insert(depth, sums);

        let rand_sharings = self.random_sharings.remove(&depth).unwrap();
        let zero_sharings = self.zero_sharings.remove(&depth).unwrap();
        DepthInput::from_mult(Multiplication::new(mult_input, rand_sharings, zero_sharings))
    }

    /// Apply the butterfly switch to a depth's multiplication results and
    /// schedule whatever comes next: the following depth, or — at the last
    /// depth — the output sharings.
    ///
    /// Mirrors the engine's `verify_mixing_level_termination`.
    async fn handle_mult_results(&mut self, depth: usize, results: Vec<SmallField>) -> DepthInput {
        if self.wire_sharings.contains_key(&(depth + 1)) {
            // Already processed — the engine may replay a depth's termination.
            return DepthInput::empty();
        }
        let Some(sums) = self.wire_pair_sums.remove(&depth) else {
            log::warn!(
                "AnonymousBroadcast: no wire pair sums recorded for depth {}",
                depth
            );
            return DepthInput::empty();
        };
        if sums.len() != results.len() {
            log::error!(
                "AnonymousBroadcast: multiplication results ({}) and wire pairs ({}) do not match at depth {}",
                results.len(),
                sums.len(),
                depth
            );
            return DepthInput::empty();
        }

        let two_inverse = &self.two_inverse;
        let next_depth_wires: Vec<SmallField> = sums
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
            self.wire_sharings.insert(depth + 1, next_depth_wires.clone());
            // The last depth's wires are the circuit's outputs: hand them back
            // for output reconstruction rather than scheduling another depth.
            return DepthInput::from_mult(Multiplication::from_output(
                (next_depth_wires, Vec::new()),
                depth,
            ));
        }

        self.wire_sharings.insert(depth + 1, next_depth_wires);
        self.init_butterfly_level(depth + 1).await
    }
}

#[async_trait]
impl Application for AnonymousBroadcast {
    fn preprocessing_count(&mut self) -> PreprocessingCounts {
        // Round each raw count up to the next multiple of (t + 1): preprocessing
        // extracts t + 1 random sharings out of every group of party sharings it
        // combines through the Vandermonde matrix, so a partial group would
        // otherwise be lost to floor division downstream.
        let group = self.num_faults + 1;
        let round_up = |x: usize| x.div_ceil(group) * group;

        // k/2 switches per depth, one multiplication and one random bit each.
        let total_switches = round_up((self.k_value / 2) * self.max_depth);
        let output_count = round_up(self.k_value);

        let counts = PreprocessingCounts::new_w_nr(
            total_switches,
            self.max_depth,
            total_switches,
            output_count,
        );
        log::info!(
            "AnonymousBroadcast::preprocessing_count -> simd_mult={}, depth={}, rand_bits={}, output={}",
            counts.simd_mult,
            counts.depth,
            counts.rand_bits,
            counts.output
        );
        counts
    }

    async fn inputs(&mut self) -> Vec<Vec<SmallField>> {
        self.generate_input_sharings()
    }

    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<SmallField>,
    ) -> DepthInput {
        if self.inputs_assembled {
            log::debug!(
                "AnonymousBroadcast: ignoring input sharing from party {}, wires already assembled",
                party
            );
            return DepthInput::empty();
        }
        log::info!(
            "AnonymousBroadcast: input sharing from party {} terminated with {} sharings",
            party,
            shares.len()
        );
        self.input_wire_sharings.insert(party, shares);
        self.try_assemble_input_wires();
        self.try_start_circuit().await
    }

    async fn on_preprocessing_complete(
        &mut self,
        rand_sharings_mult: Vec<SmallField>,
        rand_sharings_3t: Vec<SmallField>,
        rand_bit_sharings: Vec<SmallField>,
        network_routing_preprocessing: Option<NetworkRoutingPreprocessing>,
    ) -> DepthInput {
        log::info!(
            "AnonymousBroadcast: preprocessing complete — {} random sharings, {} zero sharings, \
             {} random bits; circuit: k={}, log_k={}, max_depth={}",
            rand_sharings_mult.len(),
            rand_sharings_3t.len(),
            rand_bit_sharings.len(),
            self.k_value,
            self.log_k,
            self.max_depth,
        );
        if network_routing_preprocessing.is_some() {
            log::warn!(
                "AnonymousBroadcast: network routing preprocessing received but this application does not route; ignoring it"
            );
        }

        self.rand_bits.extend(rand_bit_sharings);

        // Split the multiplication preprocessing evenly across the depths, keyed
        // by depth (1-indexed) so each depth picks up its own batch.
        let distribute = |pool: Vec<SmallField>, target: &mut HashMap<usize, Vec<SmallField>>| {
            let chunk_size = pool.len() / self.max_depth;
            if chunk_size == 0 {
                log::error!(
                    "AnonymousBroadcast: {} preprocessed sharings cannot cover {} depths",
                    pool.len(),
                    self.max_depth
                );
                return;
            }
            for (index, chunk) in pool.chunks(chunk_size).take(self.max_depth).enumerate() {
                target.insert(index + 1, chunk.to_vec());
            }
        };
        let mut random_sharings = std::mem::take(&mut self.random_sharings);
        let mut zero_sharings = std::mem::take(&mut self.zero_sharings);
        distribute(rand_sharings_mult, &mut random_sharings);
        distribute(rand_sharings_3t, &mut zero_sharings);
        self.random_sharings = random_sharings;
        self.zero_sharings = zero_sharings;

        self.preprocessing_done = true;
        self.try_start_circuit().await
    }

    async fn on_multiplication_complete(
        &mut self,
        depth: usize,
        results: (Vec<SmallField>, Vec<SmallField>),
    ) -> DepthInput {
        log::info!(
            "AnonymousBroadcast: multiplication at depth {} complete — {} results",
            depth,
            results.0.len(),
        );
        // Velox sharings are unpacked, so only the first-half results carry values.
        self.handle_mult_results(depth, results.0).await
    }

    async fn on_network_routing_complete(
        &mut self,
        depth: usize,
        _results: (Vec<SmallField>, Vec<SmallField>),
    ) -> DepthInput {
        // Velox has no network-routing module; this application never schedules
        // routing, so this hook should not fire.
        log::warn!(
            "AnonymousBroadcast: unexpected network routing completion at depth {}; ignoring",
            depth
        );
        DepthInput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(k: usize) -> AnonymousBroadcast {
        AnonymousBroadcast::new(9, 2, 0, k)
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
        let mut app = app(16);
        // k=16, log_k=4, max_depth=16, 8 switches per depth.
        let counts = app.preprocessing_count();
        assert!(counts.simd_mult >= 8 * 16);
        assert!(counts.rand_bits >= 8 * 16);
        assert!(counts.output >= 16);
        assert_eq!(counts.depth, 16);
        assert!(counts.net_route.is_none());
    }

    /// With every random bit equal to one, each switch is the identity:
    ///   product = (w1 − w2) · 1,  out1 = w1,  out2 = w2.
    /// Depth 1 of a k=4 circuit pairs (0,2) and (1,3), so the wires come back
    /// as [w0, w2, w1, w3].
    #[tokio::test]
    async fn butterfly_switch_is_identity_on_one_bits() {
        let mut app = app(4);
        let wires: Vec<SmallField> = (1..=4u64).map(|x| SmallField::from(x * 10)).collect();
        app.wire_sharings.insert(1, wires.clone());
        app.rand_bits = (0..16).map(|_| SmallField::one()).collect();
        app.random_sharings
            .insert(1, vec![SmallField::zero(); 8]);
        app.zero_sharings.insert(1, vec![SmallField::zero(); 8]);

        let depth_input = app.init_butterfly_level(1).await;
        let mult = depth_input.mult().expect("depth 1 must schedule a multiplication");
        assert_eq!(mult.input.x.0.len(), 2);
        assert!(mult.input.x.1.is_empty(), "velox sharings are unpacked");

        // bit = 1, so the multiplication output equals the wire differences.
        let products = mult.input.x.0.clone();
        // Depth 2 has no preprocessing allocated in this test, so the returned
        // DepthInput is empty — the wires it computed are what we check.
        let _ = app.handle_mult_results(1, products).await;

        let next = app.wire_sharings.get(&2).expect("depth 2 wires");
        assert_eq!(next[0], wires[0]);
        assert_eq!(next[1], wires[2]);
        assert_eq!(next[2], wires[1]);
        assert_eq!(next[3], wires[3]);
    }

    #[tokio::test]
    async fn last_depth_returns_output_sharings() {
        let mut app = app(4);
        // k=4, log_k=2, max_depth=4.
        app.current_depth = app.max_depth;
        app.wire_pair_sums
            .insert(app.max_depth, vec![SmallField::from(2u64), SmallField::from(4u64)]);

        let depth_input = app
            .handle_mult_results(app.max_depth, vec![SmallField::zero(), SmallField::zero()])
            .await;
        let mult = depth_input.mult().expect("last depth must return outputs");
        let output = mult.output.expect("output sharings");
        assert_eq!(output.0.len(), 4);
        assert!(output.1.is_empty());
    }
}
