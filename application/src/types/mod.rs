//! Protocol data-model types exposed by the [`Application`](crate::Application)
//! trait's hooks. Kept minimal: only what the trait actually exchanges.

use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// What an application tells the engine to do next.
///
/// Every hook returns one of these. The three variants are the three things an
/// application can actually say, which the previous `Option`-and-sentinel
/// encoding could not distinguish: a `Multiplication` carrying `output: Some`
/// meant "circuit finished", one carrying `output: None` meant "run this batch",
/// and an empty `DepthInput` meant *both* "not ready yet" and "something went
/// wrong and I gave up". The engine could not tell those last two apart, so an
/// application error surfaced as a silent hang. Errors now travel as `Err` on
/// the hook's `Result` instead.
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

    /// True if the application has nothing to schedule right now.
    pub fn is_waiting(&self) -> bool {
        matches!(self, Self::Waiting)
    }
}
