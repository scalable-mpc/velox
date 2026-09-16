//! Protocol data-model types exposed by the [`Application`](crate::Application)
//! trait's hooks. Kept minimal: only what the trait actually exchanges.

use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// The random wires preprocessing produced for the application, field for
/// field what [`RandomWires`](crate::RandomWires) asked for.
///
/// Delivered whole, in one call, so the application never has to reason
/// about which kind arrived first; a new kind of random wire is a new field
/// here and in `RandomWires`, not a new hook.
pub struct RandomWireShares<F: ProtocolField> {
    /// Sharings of `±1`, one per requested bit.
    pub bits: Vec<FieldElement<F>>,
    /// Sharings of uniformly random field elements, one per requested sharing.
    pub sharings: Vec<FieldElement<F>>,
}

impl<F: ProtocolField> RandomWireShares<F> {
    pub fn new(bits: Vec<FieldElement<F>>, sharings: Vec<FieldElement<F>>) -> Self {
        Self { bits, sharings }
    }

    /// No wires at all — what an application that asked for none receives.
    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

/// Counts only, like [`DepthInput`]'s: shares mean nothing to a reader alone.
impl<F: ProtocolField> std::fmt::Debug for RandomWireShares<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RandomWireShares({} bits, {} sharings)", self.bits.len(), self.sharings.len())
    }
}

/// What an application tells the engine to do next.
///
/// Every hook returns one of these. The variants are the things an application
/// can actually say, which the previous `Option`-and-sentinel encoding could
/// not distinguish: a `Multiplication` carrying `output: Some` meant "circuit
/// finished", one carrying `output: None` meant "run this batch", and an empty
/// `DepthInput` meant *both* "not ready yet" and "something went wrong and I
/// gave up". The engine could not tell those last two apart, so an application
/// error surfaced as a silent hang. Errors now travel as `Err` on the hook's
/// `Result` instead.
///
/// # One batch per depth
///
/// The three batch variants — [`Multiply`](Self::Multiply),
/// [`Reveal`](Self::Reveal), [`MaskedMultiply`](Self::MaskedMultiply) — all
/// name a `depth`, and the engine keys its per-depth state by that number. A
/// depth number therefore carries exactly one batch, of one kind; scheduling a
/// reveal and a multiplication under the same number is an error, not a way
/// to run them in parallel.
pub enum DepthInput<F: ProtocolField> {
    /// Nothing to schedule yet — the application is still waiting on input
    /// sharings, on preprocessing, or on another depth's results. The engine
    /// stops and waits to be called again.
    Waiting,
    /// Multiply these operands pairwise and call back with the results.
    ///
    /// The whole batch runs in one round, so a depth costs the same whether it
    /// carries one gate or a million.
    ///
    /// `depth` is the batch's position in the circuit, counting from 1, and must
    /// match the application's declared
    /// [`gates_per_depth`](crate::PreprocessingCounts::gates_per_depth) profile.
    /// It is the application's to state, not a counter the engine keeps, because
    /// it selects which slice of the preprocessing pool the batch consumes: a
    /// counter incremented as batches are scheduled would number depths by local
    /// timing, and depths need not be scheduled in order.
    Multiply {
        depth: usize,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
    },
    /// Publicly reconstruct these degree-`t` sharings and call back with the
    /// values through
    /// [`on_reveal_complete`](crate::Application::on_reveal_complete), as field
    /// elements every party then knows.
    ///
    /// Consumes no preprocessing: the reconstruction is the same two-level
    /// exchange a multiplication ends with, without the mask. The depth
    /// declares `0` gates in
    /// [`gates_per_depth`](crate::PreprocessingCounts::gates_per_depth), and
    /// the engine records every `([v], v)` pair for the verification phase, so
    /// a corrupt party cannot shift a revealed value undetected.
    Reveal {
        depth: usize,
        values: Vec<FieldElement<F>>,
    },
    /// Multiply the operands pairwise, add the caller's mask to each product,
    /// and publicly reconstruct `c_i = x_i · y_i + mask_i`, while protecting the privacy of x_i,y_i. 
    /// The `c_i` come back through [`on_reveal_complete`](crate::Application::on_reveal_complete).
    ///
    /// This is a multiplication whose mask the application chose — one it
    /// holds as a sharing and can subtract afterwards, or one with structure
    /// it wants to exploit, such as a random value whose bits it also holds.
    /// The engine still draws the `2t` zero-sharing that randomises the
    /// degree-`2t` opening, but no mask from its pool, and it registers
    /// `(x_i, y_i, c_i − [mask_i])` for tuple verification, so the malicious
    /// guarantee is the one `Multiply` has. Declared in
    /// [`masked_gates_per_depth`](crate::PreprocessingCounts::masked_gates_per_depth).
    MaskedMultiply {
        depth: usize,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
        mask: Vec<FieldElement<F>>,
    },
    /// The circuit is finished; these are its output sharings, in the order
    /// they should be reconstructed. The engine verifies every multiplication
    /// the circuit ran, reconstructs these wires publicly, and reports the
    /// result through [`Application::on_output`](crate::Application::on_output).
    ///
    /// An empty vector is legitimate: the circuit's product is then whatever
    /// sharings the application kept for itself, and `on_output` is the
    /// signal that they are verified.
    Done(Vec<FieldElement<F>>),
}

/// Shapes, not contents: a batch can carry millions of sharings, and none of
/// them mean anything to a reader without the shares of the other parties.
impl<F: ProtocolField> std::fmt::Debug for DepthInput<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Waiting => write!(f, "Waiting"),
            Self::Multiply { depth, x, .. } => write!(f, "Multiply(depth {}, {} gates)", depth, x.len()),
            Self::Reveal { depth, values } => write!(f, "Reveal(depth {}, {} values)", depth, values.len()),
            Self::MaskedMultiply { depth, x, .. } => {
                write!(f, "MaskedMultiply(depth {}, {} gates)", depth, x.len())
            }
            Self::Done(outputs) => write!(f, "Done({} output wires)", outputs.len()),
        }
    }
}

impl<F: ProtocolField> DepthInput<F> {
    /// A multiplication batch at circuit depth `depth`, counting from 1. The two
    /// operand vectors must be the same length.
    pub fn multiply(
        depth: usize,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
    ) -> anyhow::Result<Self> {
        if depth == 0 {
            anyhow::bail!("multiplication batch at depth 0; circuit depths count from 1");
        }
        if x.len() != y.len() {
            anyhow::bail!(
                "multiplication batch operand length mismatch at depth {}: x has {}, y has {}",
                depth,
                x.len(),
                y.len()
            );
        }
        Ok(Self::Multiply { depth, x, y })
    }

    /// A reveal batch at circuit depth `depth`, counting from 1.
    pub fn reveal(depth: usize, values: Vec<FieldElement<F>>) -> anyhow::Result<Self> {
        if depth == 0 {
            anyhow::bail!("reveal batch at depth 0; circuit depths count from 1");
        }
        Ok(Self::Reveal { depth, values })
    }

    /// A masked multiplication batch at circuit depth `depth`, counting from 1.
    /// The three vectors must be the same length.
    pub fn masked_multiply(
        depth: usize,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
        mask: Vec<FieldElement<F>>,
    ) -> anyhow::Result<Self> {
        if depth == 0 {
            anyhow::bail!("masked multiplication batch at depth 0; circuit depths count from 1");
        }
        if x.len() != y.len() || x.len() != mask.len() {
            anyhow::bail!(
                "masked multiplication batch length mismatch at depth {}: x has {}, y has {}, mask has {}",
                depth,
                x.len(),
                y.len(),
                mask.len()
            );
        }
        Ok(Self::MaskedMultiply { depth, x, y, mask })
    }

    /// True if the application has nothing to schedule right now.
    pub fn is_waiting(&self) -> bool {
        matches!(self, Self::Waiting)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = fields::DefaultField;

    fn elems(n: usize) -> Vec<FieldElement<F>> {
        (0..n as u64).map(FieldElement::<F>::from).collect()
    }

    #[test]
    fn depths_count_from_one() {
        assert!(DepthInput::<F>::multiply(0, elems(1), elems(1)).is_err());
        assert!(DepthInput::<F>::reveal(0, elems(1)).is_err());
        assert!(DepthInput::<F>::masked_multiply(0, elems(1), elems(1), elems(1)).is_err());
        assert!(DepthInput::<F>::multiply(1, elems(1), elems(1)).is_ok());
        assert!(DepthInput::<F>::reveal(1, elems(1)).is_ok());
        assert!(DepthInput::<F>::masked_multiply(1, elems(1), elems(1), elems(1)).is_ok());
    }

    /// Empty batches are accepted, as `multiply` always has; the engine is
    /// what warns about them.
    #[test]
    fn empty_batches_are_accepted() {
        assert!(DepthInput::<F>::reveal(1, Vec::new()).is_ok());
        assert!(DepthInput::<F>::masked_multiply(1, Vec::new(), Vec::new(), Vec::new()).is_ok());
    }

    #[test]
    fn masked_multiply_rejects_every_length_mismatch() {
        let err = DepthInput::<F>::masked_multiply(2, elems(3), elems(2), elems(3)).unwrap_err();
        assert!(err.to_string().contains("x has 3, y has 2, mask has 3"), "got {err}");
        assert!(DepthInput::<F>::masked_multiply(2, elems(3), elems(3), elems(2)).is_err());
        assert!(DepthInput::<F>::masked_multiply(2, elems(2), elems(3), elems(3)).is_err());
    }

    /// Debug prints shapes and never a share.
    #[test]
    fn debug_names_the_kind_depth_and_width() {
        let m = DepthInput::<F>::multiply(3, elems(5), elems(5)).unwrap();
        assert_eq!(format!("{m:?}"), "Multiply(depth 3, 5 gates)");
        let r = DepthInput::<F>::reveal(4, elems(7)).unwrap();
        assert_eq!(format!("{r:?}"), "Reveal(depth 4, 7 values)");
        let mm = DepthInput::<F>::masked_multiply(5, elems(2), elems(2), elems(2)).unwrap();
        assert_eq!(format!("{mm:?}"), "MaskedMultiply(depth 5, 2 gates)");
        assert_eq!(format!("{:?}", DepthInput::<F>::Waiting), "Waiting");
    }
}
