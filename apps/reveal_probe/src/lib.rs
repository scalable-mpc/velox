//! An end-to-end check of the engine's public reveal.
//!
//! Every party deals `k` inputs. The circuit is three depths:
//!
//! 1. `Multiply` the first two input wires — one gate, so the run also has a
//!    verified tuple and the multiplication path runs alongside the reveal.
//! 2. `Reveal` every input wire blinded by a random sharing, `[x_i] + [r_i]`,
//!    with the `[r_i]` drawn as random wires. This is the only thing in the
//!    workspace that reveals.
//! 3. `Done` with the blinds `[r_i]` and the product as output wires.
//!
//! When the output is reconstructed — verified and agreed, which is when
//! `on_output` fires — each party checks its own block: `revealed_i − r_i`
//! must be the input it dealt, and dealer 0 checks the product. The result is
//! one log line, `reveal_probe: N reveals verified`, which the fixture greps
//! for; a mismatch is an error naming the wire.

use std::collections::HashMap;

use anyhow::{bail, Result};
use async_trait::async_trait;
use velox::{Application, DepthInput, FieldElement, PreprocessingCounts, ProtocolField, RandomWireShares, RandomWires};

pub struct RevealProbe<F: ProtocolField> {
    num_nodes: usize,
    my_id: usize,
    /// Inputs per party.
    k: usize,

    /// What this party dealt, kept to check against.
    my_inputs: Vec<FieldElement<F>>,
    /// Input sharings by dealer.
    input_wires: HashMap<usize, Vec<FieldElement<F>>>,
    /// The blinds, one per input wire across all dealers.
    blinds: Option<Vec<FieldElement<F>>>,
    /// The public values the reveal returned.
    revealed: Option<Vec<FieldElement<F>>>,
    /// The depth-1 product, kept until it goes out as the last output wire.
    product: Option<FieldElement<F>>,
    circuit_started: bool,
}

impl<F: ProtocolField> RevealProbe<F> {
    pub fn new(num_nodes: usize, my_id: usize, k: usize) -> Result<Self> {
        if k < 2 {
            bail!("the probe multiplies the first two inputs, so it needs at least 2 per party");
        }
        Ok(Self {
            num_nodes,
            my_id,
            k,
            my_inputs: Vec::new(),
            input_wires: HashMap::new(),
            blinds: None,
            revealed: None,
            product: None,
            circuit_started: false,
        })
    }

    /// Input wires across all dealers, in dealer order.
    fn wires(&self) -> Vec<FieldElement<F>> {
        (0..self.num_nodes)
            .flat_map(|dealer| self.input_wires[&dealer].iter().cloned())
            .collect()
    }

    fn try_start(&mut self) -> Result<DepthInput<F>> {
        if self.circuit_started || self.blinds.is_none() || self.input_wires.len() < self.num_nodes {
            return Ok(DepthInput::Waiting);
        }
        self.circuit_started = true;
        let wires = self.wires();
        log::info!("reveal_probe: all {} dealers in; multiplying wires 0 and 1", self.num_nodes);
        DepthInput::multiply(1, vec![wires[0].clone()], vec![wires[1].clone()])
    }

    /// The check, as a function of what the run produced. `outputs` are the
    /// `n·k` blinds followed by the product.
    pub fn check(
        my_id: usize,
        k: usize,
        my_inputs: &[FieldElement<F>],
        revealed: &[FieldElement<F>],
        outputs: &[FieldElement<F>],
    ) -> Result<usize> {
        let total = revealed.len();
        if outputs.len() != total + 1 {
            bail!("expected {} output wires ({} blinds and the product), got {}", total + 1, total, outputs.len());
        }
        let blinds = &outputs[..total];
        let product = &outputs[total];
        let mut verified = 0;
        for (j, x) in my_inputs.iter().enumerate() {
            let wire = my_id * k + j;
            let unblinded = &revealed[wire] - &blinds[wire];
            if unblinded != *x {
                bail!("wire {} (party {}, input {}): revealed − blind does not match the input dealt", wire, my_id, j);
            }
            verified += 1;
        }
        if my_id == 0 && *product != &my_inputs[0] * &my_inputs[1] {
            bail!("the product wire does not match x_0 · x_1");
        }
        Ok(verified)
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for RevealProbe<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        // Depth 1 multiplies one gate; depth 2 reveals, which consumes nothing.
        PreprocessingCounts::new(vec![1, 0], self.num_nodes * self.k + 1)
    }

    fn random_wires(&self) -> RandomWires {
        RandomWires::new(0, self.num_nodes * self.k)
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        self.my_inputs = (0..self.k).map(|_| F::rand()).collect();
        self.my_inputs.clone()
    }

    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        if self.circuit_started {
            return Ok(DepthInput::Waiting);
        }
        if shares.len() != self.k {
            bail!("dealer {} shared {} inputs, expected {}", party, shares.len(), self.k);
        }
        self.input_wires.insert(party, shares);
        self.try_start()
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        if wires.sharings.len() != self.num_nodes * self.k {
            bail!("asked for {} random sharings, got {}", self.num_nodes * self.k, wires.sharings.len());
        }
        self.blinds = Some(wires.sharings);
        self.try_start()
    }

    async fn on_depth_complete(&mut self, depth: usize, results: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        if depth != 1 || results.len() != 1 {
            bail!("unexpected multiplication completion: depth {}, {} results", depth, results.len());
        }
        // Keep the product as the last output wire; reveal the blinded inputs.
        self.product = results.into_iter().next();
        let blinds = self.blinds.as_ref().unwrap();
        let blinded: Vec<FieldElement<F>> = self.wires().iter().zip(blinds.iter()).map(|(w, r)| w + r).collect();
        log::info!("reveal_probe: revealing {} blinded input wires", blinded.len());
        DepthInput::reveal(2, blinded)
    }

    async fn on_reveal_complete(&mut self, depth: usize, values: Vec<FieldElement<F>>) -> Result<DepthInput<F>> {
        if depth != 2 || values.len() != self.num_nodes * self.k {
            bail!("unexpected reveal completion: depth {}, {} values", depth, values.len());
        }
        self.revealed = Some(values);
        let Some(product) = self.product.take() else {
            bail!("the reveal completed before the multiplication did");
        };
        let mut outputs = self.blinds.clone().unwrap();
        outputs.push(product);
        log::info!("reveal_probe: reveal complete, reconstructing {} blinds and the product", outputs.len() - 1);
        Ok(DepthInput::Done(outputs))
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let Some(revealed) = self.revealed.as_ref() else {
            bail!("output reconstructed before the reveal completed");
        };
        match Self::check(self.my_id, self.k, &self.my_inputs, revealed, &outputs) {
            Ok(n) => {
                log::info!("reveal_probe: {} reveals verified", n);
                Ok(())
            }
            Err(err) => {
                log::error!("reveal_probe: FAILED: {:#}", err);
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The application against a plaintext engine: shares are the values
    //! themselves, multiply is the product, reveal is the identity.

    use super::*;
    use velox::fields::Mersenne61Field;

    type F = Mersenne61Field;

    fn elem(x: u64) -> FieldElement<F> {
        FieldElement::<F>::from(x)
    }

    #[tokio::test]
    async fn runs_the_three_depths_and_verifies() {
        let (n, k) = (3, 2);
        let mut app = RevealProbe::<F>::new(n, 1, k).unwrap();
        let mine = app.inputs().await;
        assert_eq!(mine.len(), k);
        let blinds: Vec<_> = (0..n * k).map(|i| elem(100 + i as u64)).collect();

        assert!(app.on_preprocessing_complete(RandomWireShares::new(Vec::new(), blinds.clone())).await.unwrap().is_waiting());
        // Dealers 0 and 2 supply fixed values, dealer 1 is this party.
        assert!(app.input_sharing_termination(0, vec![elem(3), elem(5)]).await.unwrap().is_waiting());
        assert!(app.input_sharing_termination(2, vec![elem(7), elem(9)]).await.unwrap().is_waiting());
        let DepthInput::Multiply { depth, x, y } = app.input_sharing_termination(1, mine.clone()).await.unwrap() else {
            panic!("all dealers in: the multiplication should start");
        };
        assert_eq!((depth, x.len()), (1, 1));

        let product = &x[0] * &y[0];
        let DepthInput::Reveal { depth, values } = app.on_depth_complete(1, vec![product.clone()]).await.unwrap() else {
            panic!("the product should trigger the reveal");
        };
        assert_eq!((depth, values.len()), (2, n * k));

        let DepthInput::Done(outputs) = app.on_reveal_complete(2, values).await.unwrap() else {
            panic!("the reveal should finish the circuit");
        };
        assert_eq!(outputs.len(), n * k + 1);
        assert_eq!(outputs[n * k], product);
        app.on_output(outputs).await.unwrap();
    }

    #[test]
    fn check_catches_a_shifted_reveal() {
        let (k, my_id) = (2, 0);
        let mine = vec![elem(3), elem(5)];
        let blinds: Vec<_> = (0..4).map(|i| elem(10 + i)).collect();
        let mut revealed: Vec<_> = vec![&mine[0] + &blinds[0], &mine[1] + &blinds[1], elem(0), elem(0)];
        let mut outputs = blinds.clone();
        outputs.push(elem(15));
        assert_eq!(RevealProbe::<F>::check(my_id, k, &mine, &revealed, &outputs).unwrap(), 2);

        revealed[1] = &revealed[1] + elem(1);
        let err = RevealProbe::<F>::check(my_id, k, &mine, &revealed, &outputs).unwrap_err();
        assert!(err.to_string().contains("wire 1"), "{err}");

        revealed[1] = &revealed[1] - elem(1);
        outputs[4] = elem(16);
        assert!(RevealProbe::<F>::check(my_id, k, &mine, &revealed, &outputs).unwrap_err().to_string().contains("product"));
        assert!(RevealProbe::<F>::check(1, k, &mine, &revealed, &outputs).is_err(), "party 1's block is zeros here");
    }
}
