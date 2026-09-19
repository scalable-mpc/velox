//! The engine-facing API: what the MPC engine asks of the application it
//! hosts.
//!
//! The engine drives the protocol phases — preprocessing, multiplication,
//! verification, output reconstruction — and at each phase boundary calls
//! into the [`Application`] it was given, acting on the [`DepthInput`] each
//! hook returns. Its vocabulary is deliberately small: a multiplication
//! batch, a reveal batch, a masked-multiplication batch. Everything richer —
//! comparison, truncation, fixed-point multiplication — is built above it by
//! the Planner, which is itself an `Application` (see `bridge`) hosting a
//! [`PlannerApplication`](super::application::PlannerApplication).
//!
//! An application may also implement this trait directly, as
//! `anonymous_broadcast`, `btx_setup` and `reveal_probe` do, when it needs
//! nothing beyond the engine's own batches.
//!
//! # Division of labour
//!
//! The engine owns the protocol; the application owns the circuit. Concretely,
//! the application says *what to multiply* and *what to reveal* and the engine
//! decides *how*: it numbers the depths, draws the preprocessing each batch
//! consumes, picks the multiplication protocol, runs the public
//! reconstruction, and verifies the resulting tuples and revealed values. An
//! application that needed to know `2t+1` to portion its own preprocessing was
//! doing the engine's job, and doing it once per application.

use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;
use rand::random;


/// The random wires preprocessing produced for the application, field for
/// field what [`RandomWires`](RandomWires) asked for.
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

/// What the Planner tells the engine to do next.
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
    /// [`gates_per_depth`](PreprocessingCounts::gates_per_depth) profile.
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
    /// [`on_reveal_complete`](Application::on_reveal_complete), as field
    /// elements every party then knows.
    ///
    /// Consumes no preprocessing: the reconstruction is the same two-level
    /// exchange a multiplication ends with, without the mask. The depth
    /// declares `0` gates in
    /// [`gates_per_depth`](PreprocessingCounts::gates_per_depth), and
    /// the engine records every `([v], v)` pair for the verification phase, so
    /// a corrupt party cannot shift a revealed value undetected.
    Reveal {
        depth: usize,
        values: Vec<FieldElement<F>>,
    },
    /// Multiply the operands pairwise, add the caller's mask to each product,
    /// and publicly reconstruct `c_i = x_i · y_i + mask_i`, while protecting the privacy of x_i,y_i. 
    /// The `c_i` come back through [`on_reveal_complete`](Application::on_reveal_complete).
    ///
    /// This is a multiplication whose mask the application chose — one it
    /// holds as a sharing and can subtract afterwards, or one with structure
    /// it wants to exploit, such as a random value whose bits it also holds.
    /// The engine still draws the `2t` zero-sharing that randomises the
    /// degree-`2t` opening, but no mask from its pool, and it registers
    /// `(x_i, y_i, c_i − [mask_i])` for tuple verification, so the malicious
    /// guarantee is the one `Multiply` has. Declared in
    /// [`masked_gates_per_depth`](PreprocessingCounts::masked_gates_per_depth).
    MaskedMultiply {
        depth: usize,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
        mask: Vec<FieldElement<F>>,
    },
    /// The circuit is finished; these are its output sharings, in the order
    /// they should be reconstructed. The engine verifies every multiplication
    /// the circuit ran, reconstructs these wires publicly, and reports the
    /// result through [`Application::on_output`](Application::on_output).
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
mod depth_input_tests {
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

/// What the engine consumes evaluating an application's circuit.
///
/// Raw demand, in the application's own terms; the engine converts it into
/// ACSS/Sh2t batch sizes in `init_rand_sh`, adding what verification and the
/// common coin draw. Everything here is spent *by the engine* — masks and
/// zero-sharings per multiplication, masks per output wire — and the
/// application never sees it. What the application consumes itself, as
/// wires, is declared separately in [`RandomWires`].
#[derive(Default, Clone, Debug)]
pub struct PreprocessingCounts {
    /// Number of multiplication gates at each depth, depth 1 first.
    ///
    /// A profile rather than a total, because the engine reserves preprocessing material 
    /// for each depth as a
    /// *fixed slice* of the preprocessing pool, computed from this vector. Every
    /// party derives the same table from the same circuit, so depth `d` binds to
    /// the same random sharings everywhere however the depths happen to be
    /// scheduled locally — out of order, or fast-forwarded past a depth whose
    /// reconstruction already arrived. Handing out masks in the order batches
    /// happen to be scheduled would make that binding depend on local timing,
    /// and two parties masking a gate differently do not reconstruct.
    ///
    /// A depth may run *fewer* gates than it declares — it then uses a prefix of
    /// its slice, which is still the same prefix everywhere — but never more.
    ///
    /// A depth that carries a [`Reveal`](DepthInput::Reveal) declares `0`
    /// here: a reveal consumes no preprocessing.
    pub gates_per_depth: Vec<usize>,
    /// Number of [`MaskedMultiply`](DepthInput::MaskedMultiply) gates at each
    /// depth, depth 1 first, under the same fixed-slice rule as
    /// [`gates_per_depth`](Self::gates_per_depth). Shorter than that vector
    /// means the remaining depths have none; empty (the default) means the
    /// circuit has no masked multiplications at all.
    ///
    /// A masked gate is charged a share of the `2t` zero-sharings a batch
    /// draws but no mask, since the application supplies the mask. A depth
    /// declares gates of one kind only — one batch per depth, see
    /// [`DepthInput`].
    pub masked_gates_per_depth: Vec<usize>,
    /// Number of output wires to be reconstructed, each of which needs a mask.
    pub output: usize,
}

impl PreprocessingCounts {
    pub fn new(gates_per_depth: Vec<usize>, output: usize) -> Self {
        Self {
            gates_per_depth,
            masked_gates_per_depth: Vec::new(),
            output,
        }
    }

    /// Declare masked multiplication gates per depth; see
    /// [`masked_gates_per_depth`](Self::masked_gates_per_depth).
    pub fn with_masked_gates(mut self, masked_gates_per_depth: Vec<usize>) -> Self {
        self.masked_gates_per_depth = masked_gates_per_depth;
        self
    }

    /// Total number of multiplication gates across every depth, plain and
    /// masked: every one of them produces a tuple the verification phase
    /// checks, which is what this total sizes.
    pub fn mult_gates(&self) -> usize {
        self.plain_gates() + self.masked_gates_per_depth.iter().sum::<usize>()
    }

    /// Gates that draw a mask from the engine's pool — the plain ones.
    pub fn plain_gates(&self) -> usize {
        self.gates_per_depth.iter().sum()
    }

    /// Number of multiplication depths — protocol rounds, not gate depth.
    /// Linear gates are evaluated locally and do not count.
    pub fn depth(&self) -> usize {
        self.gates_per_depth.len().max(self.masked_gates_per_depth.len())
    }

    /// Number of plain gates depth `depth` declared, counting depths from 1;
    /// `None` past the declared depths.
    pub fn gates_at_depth(&self, depth: usize) -> Option<usize> {
        Self::at_depth(&self.gates_per_depth, self.depth(), depth)
    }

    /// Number of masked gates depth `depth` declared, counting depths from 1;
    /// `Some(0)` for a declared depth the shorter vector does not reach.
    pub fn masked_gates_at_depth(&self, depth: usize) -> Option<usize> {
        Self::at_depth(&self.masked_gates_per_depth, self.depth(), depth)
    }

    fn at_depth(profile: &[usize], depths: usize, depth: usize) -> Option<usize> {
        let index = depth.checked_sub(1)?;
        if index >= depths {
            return None;
        }
        Some(profile.get(index).copied().unwrap_or(0))
    }
}

/// Random wires an application's circuit consumes.
///
/// These are values the application reads *as sharings* — a random bit to
/// switch on, a random secret to raise to powers — as opposed to the
/// material in [`PreprocessingCounts`] that the engine spends on the
/// application's behalf and never hands over. Both are produced in the
/// preprocessing phase from the same ACS-agreed dealers, and both come with
/// the same guarantee: degree-`t`, uniformly random, unknown to any
/// `t`-coalition, and available without waiting on any particular dealer.
///
/// The engine delivers them in [`RandomWireShares`], field for field.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct RandomWires {
    /// Random bits, delivered as sharings of `±1`.
    pub bits: usize,
    /// Sharings of uniformly random field elements.
    pub sharings: usize,
}

impl RandomWires {
    pub fn new(bits: usize, sharings: usize) -> Self {
        Self { bits, sharings }
    }
}

/// Application-level interface for the MPC protocol.
///
/// The base MPC `Context<A>` drives the event loop and protocol phases
/// (preprocessing, multiplication, verification, output reconstruction). At
/// phase boundaries it calls into the application via these hooks, and acts on
/// the [`DepthInput`] each one returns.
///
/// Each application (anonymous broadcast, arithmetic circuits, decision trees,
/// …) implements this trait to define application-specific behavior.
///
/// # Extending
/// Add new hook methods with default implementations so existing applications
/// continue to compile without changes. Do not add a hook the engine does not
/// call: three such hooks accumulated here before, and every application had to
/// implement them to satisfy the trait even though none could ever fire.
#[async_trait]
pub trait Application<F: ProtocolField>: Send + 'static {
    /// How much preprocessing the circuit needs.
    ///
    /// Takes `&self` and must be pure, and must return the same answer at every
    /// party: the engine reads it once, at the start of preprocessing, and lays
    /// out the per-depth preprocessing slices from it. Two parties disagreeing
    /// on `gates_per_depth` would bind different random sharings to the same
    /// gate.
    fn preprocessing_count(&self) -> PreprocessingCounts;

    /// The random wires the circuit consumes. Same contract as
    /// [`preprocessing_count`](Application::preprocessing_count): pure, read
    /// once at the start of preprocessing, the same answer at every party.
    /// The default asks for none, so a circuit without random wires need not
    /// mention them.
    fn random_wires(&self) -> RandomWires {
        RandomWires::default()
    }

    /// The secrets this party contributes to the circuit's input wires, in wire
    /// order.
    ///
    /// Parties that deal no inputs return an empty `Vec`; the default
    /// implementation does exactly that, so applications without input sharing
    /// need not override it.
    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        Vec::new()
    }

    /// A party's input ACSS has terminated, so its `shares` are now available.
    async fn input_sharing_termination(
        &mut self,
        party: usize,
        shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;

    /// Preprocessing is complete. `wires` are the random wires the application
    /// asked for in [`random_wires`](Application::random_wires); the
    /// multiplication masks stay with the engine, which draws them per batch.
    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>>;

    /// Every multiplication at `depth` has completed; `results` are the output
    /// sharings, in the order the operands were given.
    async fn on_depth_complete(
        &mut self,
        depth: usize,
        results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>>;

    /// A [`Reveal`](DepthInput::Reveal) or
    /// [`MaskedMultiply`](DepthInput::MaskedMultiply) batch at `depth` has
    /// completed; `values` are the reconstructed field elements, in operand
    /// order. They are public — every party holds the same vector.
    ///
    /// The default is an error, not [`DepthInput::Waiting`]: an application
    /// that scheduled a reveal and does not handle its result has a bug, and
    /// `Waiting` would turn that into the silent hang the `Err` path exists to
    /// surface. Applications that never reveal need not override it.
    async fn on_reveal_complete(
        &mut self,
        depth: usize,
        values: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        let _ = values;
        anyhow::bail!(
            "a reveal batch completed at depth {} but this application does not implement on_reveal_complete",
            depth
        )
    }

    /// The protocol has terminated: every multiplication the circuit ran has
    /// been verified, and the parties have agreed that at least `t+1` of them
    /// reconstructed the output. `outputs` are the reconstructed values of the
    /// wires the application returned in [`DepthInput::Done`], in that order —
    /// empty if it returned none.
    ///
    /// This is the only point at which an application knows its run was
    /// verified *and* agreed on, so it is where an application whose real
    /// product is the sharings it still holds — a key-generation setup, say —
    /// persists them. Nothing the application holds should be treated as
    /// final before this fires. The default does nothing, so applications
    /// that only care about the reconstructed output need not override it.
    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let _ = outputs;
        Ok(())
    }
}

/// Default no-op application for running the base MPC protocol
/// without application-specific logic (e.g. benchmarking preprocessing).
pub struct DefaultApplication<F: ProtocolField>(PhantomData<F>);

impl<F: ProtocolField> Default for DefaultApplication<F> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<F: ProtocolField> DefaultApplication<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl<F: ProtocolField> Application<F> for DefaultApplication<F> {
    fn preprocessing_count(&self) -> PreprocessingCounts {
        let counts = PreprocessingCounts::new(vec![1000; 10], 100);
        log::info!(
            "DefaultApplication::preprocessing_count -> mult_gates={}, depth={}, output={}",
            counts.mult_gates(),
            counts.depth(),
            counts.output
        );
        counts
    }

    fn random_wires(&self) -> RandomWires {
        RandomWires::new(10000, 0)
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        (0..1000)
            .map(|_| FieldElement::<F>::from(random::<u64>()))
            .collect()
    }

    async fn input_sharing_termination(
        &mut self,
        _party: usize,
        _shares: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        Ok(DepthInput::Waiting)
    }

    async fn on_preprocessing_complete(&mut self, wires: RandomWireShares<F>) -> Result<DepthInput<F>> {
        log::info!(
            "DefaultApplication: preprocessing complete with {} random bits and {} random sharings",
            wires.bits.len(),
            wires.sharings.len()
        );
        Ok(DepthInput::Waiting)
    }

    async fn on_depth_complete(
        &mut self,
        depth: usize,
        _results: Vec<FieldElement<F>>,
    ) -> Result<DepthInput<F>> {
        log::info!("DefaultApplication: depth {} complete", depth);
        Ok(DepthInput::Waiting)
    }
}

#[cfg(test)]
mod counts_tests {
    use super::*;

    /// Masked gates count towards the tuples verification sizes, not towards
    /// the masks the pool provides.
    #[test]
    fn masked_gates_are_tuples_but_not_masks() {
        let counts = PreprocessingCounts::new(vec![3, 0, 5], 0).with_masked_gates(vec![0, 4]);
        assert_eq!(counts.plain_gates(), 8);
        assert_eq!(counts.mult_gates(), 12);
        assert_eq!(PreprocessingCounts::new(vec![3], 0).mult_gates(), 3);
    }

    /// The depth count covers both profiles, and each lookup fills the shorter
    /// profile with zeros up to that count.
    #[test]
    fn depth_spans_both_profiles() {
        let counts = PreprocessingCounts::new(vec![3, 0], 0).with_masked_gates(vec![0, 4, 6]);
        assert_eq!(counts.depth(), 3);
        assert_eq!(counts.gates_at_depth(1), Some(3));
        assert_eq!(counts.gates_at_depth(3), Some(0), "declared by the masked profile only");
        assert_eq!(counts.masked_gates_at_depth(1), Some(0));
        assert_eq!(counts.masked_gates_at_depth(3), Some(6));
        assert_eq!(counts.gates_at_depth(4), None);
        assert_eq!(counts.masked_gates_at_depth(4), None);
        assert_eq!(counts.gates_at_depth(0), None);
        assert_eq!(counts.masked_gates_at_depth(0), None);
    }

    /// Without masked gates nothing about the old accessors changes.
    #[test]
    fn plain_profile_is_unchanged() {
        let counts = PreprocessingCounts::new(vec![1, 2], 7);
        assert_eq!(counts.depth(), 2);
        assert_eq!(counts.gates_at_depth(2), Some(2));
        assert_eq!(counts.gates_at_depth(3), None);
        assert_eq!(counts.masked_gates_at_depth(2), Some(0));
        assert_eq!(counts.output, 7);
    }
}
