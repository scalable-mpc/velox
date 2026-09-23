//! Private ERC20 payments on the Planner.
//!
//! An ERC20 token is a table of account balances and one operation,
//! `transfer(from, to, amount)`, which moves `amount` from `from` to `to` if
//! `from` holds at least that much. Here the balances and the amounts are
//! secret-shared; which account pays which is public, like addresses on a
//! chain. A transfer that would overdraw its sender, or whose amount is
//! negative, moves nothing — and nobody learns that it was rejected.
//!
//! # The ledger
//!
//! A public text file every party reads (see [`Ledger::parse`]):
//!
//! ```text
//! accounts <N>
//! <from> <to>        one line per transfer, in order
//! ```
//!
//! Account `a` is owned by party `a mod n`. Each party deals, as its input
//! sharings, the initial balances of its accounts (in account order) and then
//! the amounts of the transfers its accounts send (in transfer order).
//!
//! # A transfer, in the Planner's operations
//!
//! 1. `Compare` — `x = [bal_from < amount]` (insufficient funds) and
//!    `y = [amount < 0]`, both in one op-depth.
//! 2. `Mul` — `ok = (1 − x) · (1 − y)`.
//! 3. `Mul` — `δ = ok · amount`.
//! 4. local — `bal_from −= δ`, `bal_to += δ`.
//!
//! # Epochs
//!
//! Transfers run in epochs: every transfer of an epoch is checked against the
//! balances at the start of the epoch, so the whole epoch shares the three
//! op-depths above — 10 rounds at ℓ = 61, 9 at ℓ = 31, however many
//! transfers it holds. [`Ledger::epochs`] starts a new epoch whenever a
//! transfer's sender has already sent or received in the current one, which
//! makes the result exactly that of running the transfers one by one.
//!
//! Balances and amounts must stay inside the comparison's domain,
//! `|v| < 2^{ℓ−2}`. The final balances are the circuit's outputs, opened to
//! every party: a benchmark application, not a wallet — a deployment would
//! open each balance to its owner only, which the engine does not offer.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use async_trait::async_trait;
use velox::{
    FieldElement, MersennePrimeField, Op, OpDepthInput, OpParams, OpResult, OpType, PlannerApplication, PlannerCounts,
    ProtocolField, RandomWireShares,
};

/// The public part of a token: how many accounts, and who pays whom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ledger {
    pub num_accounts: usize,
    /// `(from, to)`, in order.
    pub transfers: Vec<(usize, usize)>,
}

impl Ledger {
    /// Parse `accounts <N>` and then one `<from> <to>` line per transfer;
    /// blank lines and `#` comments are skipped.
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.trim()))
            .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));
        let Some((line_no, header)) = lines.next() else {
            bail!("empty ledger");
        };
        let num_accounts = match header.split_whitespace().collect::<Vec<_>>()[..] {
            ["accounts", n] => n.parse::<usize>().map_err(|_| anyhow::anyhow!("line {}: bad account count {:?}", line_no, n))?,
            _ => bail!("line {}: the ledger starts with `accounts <N>`", line_no),
        };
        let mut transfers = Vec::new();
        for (line_no, line) in lines {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [from, to] = fields[..] else {
                bail!("line {}: a transfer is `<from> <to>`, got {:?}", line_no, line);
            };
            let parse = |s: &str| -> Result<usize> {
                let a = s.parse::<usize>().map_err(|_| anyhow::anyhow!("line {}: bad account {:?}", line_no, s))?;
                if a >= num_accounts {
                    bail!("line {}: account {} is outside the {} declared", line_no, a, num_accounts);
                }
                Ok(a)
            };
            transfers.push((parse(from)?, parse(to)?));
        }
        Ok(Self { num_accounts, transfers })
    }

    /// The transfers grouped into epochs, as indices into `transfers`. A new
    /// epoch starts when a transfer's sender has already sent or received in
    /// the current one: then every sender's check sees exactly the balance
    /// the transfers before it left, as if they ran one by one.
    pub fn epochs(&self) -> Vec<Vec<usize>> {
        let mut epochs: Vec<Vec<usize>> = Vec::new();
        let (mut sent, mut received) = (HashSet::new(), HashSet::new());
        for (i, &(from, to)) in self.transfers.iter().enumerate() {
            if epochs.is_empty() || sent.contains(&from) || received.contains(&from) {
                epochs.push(Vec::new());
                sent.clear();
                received.clear();
            }
            epochs.last_mut().unwrap().push(i);
            sent.insert(from);
            received.insert(to);
        }
        epochs
    }

    /// How many values party `party` deals: its accounts' balances, then the
    /// amounts of the transfers its accounts send.
    pub fn inputs_of_party(&self, party: usize, num_nodes: usize) -> usize {
        let accounts = (0..self.num_accounts).filter(|a| a % num_nodes == party).count();
        let sends = self.transfers.iter().filter(|(from, _)| from % num_nodes == party).count();
        accounts + sends
    }
}

pub struct Erc20<F: ProtocolField + MersennePrimeField> {
    pub num_nodes: usize,
    pub my_id: usize,
    ledger: Ledger,
    /// Transfer indices, epoch by epoch.
    epochs: Vec<Vec<usize>>,

    /// What this party deals: its balances, then its transfers' amounts.
    my_inputs: Option<Vec<FieldElement<F>>>,
    /// Input sharings by dealer, until every dealer is in.
    dealt: HashMap<usize, Vec<FieldElement<F>>>,
    inputs_bound: bool,
    preprocessing_done: bool,
    started: bool,

    /// Shared balance per account; updated at the end of every epoch.
    balances: Vec<FieldElement<F>>,
    /// Shared amount per transfer.
    amounts: Vec<FieldElement<F>>,
}

impl<F: ProtocolField + MersennePrimeField> Erc20<F> {
    pub fn new(num_nodes: usize, my_id: usize, ledger: Ledger) -> Self {
        let epochs = ledger.epochs();
        Self {
            num_nodes,
            my_id,
            epochs,
            my_inputs: None,
            dealt: HashMap::new(),
            inputs_bound: false,
            preprocessing_done: false,
            started: false,
            balances: Vec::new(),
            amounts: Vec::new(),
            ledger,
        }
    }

    /// The values this party deals: its accounts' balances in account order,
    /// then its accounts' transfer amounts in transfer order.
    pub fn with_inputs(mut self, inputs: Vec<FieldElement<F>>) -> Self {
        self.my_inputs = Some(inputs);
        self
    }

    pub fn inputs_per_party(&self) -> usize {
        self.ledger.inputs_of_party(self.my_id, self.num_nodes)
    }

    /// Start once the inputs are bound and preprocessing is in — the two
    /// arrive in either order.
    fn start_when_ready(&mut self) -> Result<OpDepthInput<F>> {
        if self.started || !self.inputs_bound || !self.preprocessing_done {
            return Ok(OpDepthInput::Waiting);
        }
        self.started = true;
        log::info!(
            "Erc20: {} transfers over {} accounts in {} epochs",
            self.ledger.transfers.len(),
            self.ledger.num_accounts,
            self.epochs.len()
        );
        self.check_funds_of_epoch(0)
    }

    /// Op-depth 1 of epoch `epoch`: for each of its transfers, compare the
    /// sender's balance with the amount, and the amount with zero. Past the
    /// last epoch, the final balances are the circuit's outputs.
    fn check_funds_of_epoch(&mut self, epoch: usize) -> Result<OpDepthInput<F>> {
        let Some(transfers) = self.epochs.get(epoch) else {
            log::info!("Erc20: every epoch done, opening {} balances", self.balances.len());
            return Ok(OpDepthInput::Done(self.balances.clone()));
        };
        let zero = FieldElement::<F>::zero();
        let mut a = Vec::with_capacity(2 * transfers.len());
        let mut b = Vec::with_capacity(2 * transfers.len());
        for &t in transfers {
            a.push(self.balances[self.ledger.transfers[t].0].clone());
            b.push(self.amounts[t].clone());
        }
        for &t in transfers {
            a.push(self.amounts[t].clone());
            b.push(zero.clone());
        }
        log::info!("Erc20: epoch {} — checking {} transfers", epoch + 1, transfers.len());
        OpDepthInput::op(3 * epoch + 1, Op::Compare { a, b })
    }
}

#[async_trait]
impl<F: ProtocolField + MersennePrimeField> PlannerApplication<F> for Erc20<F> {
    fn preprocessing_count(&self) -> PlannerCounts {
        let ops = self
            .epochs
            .iter()
            .flat_map(|epoch| {
                [
                    OpParams::new(OpType::Compare, 2 * epoch.len()),
                    OpParams::new(OpType::Mul, epoch.len()),
                    OpParams::new(OpType::Mul, epoch.len()),
                ]
            })
            .collect();
        PlannerCounts::new(ops, self.ledger.num_accounts)
    }

    async fn inputs(&mut self) -> Vec<FieldElement<F>> {
        let count = self.inputs_per_party();
        let mut values = self.my_inputs.clone().unwrap_or_default();
        if values.len() != count {
            log::warn!("Erc20: {} inputs supplied for {} expected; using zeros for the rest", values.len(), count);
            values.resize(count, FieldElement::<F>::zero());
        }
        values
    }

    /// Record a dealer's sharings; once every dealer is in, lay them out as
    /// the balances and amounts they stand for.
    async fn input_sharing_termination(&mut self, party: usize, shares: Vec<FieldElement<F>>) -> Result<OpDepthInput<F>> {
        if self.inputs_bound {
            return Ok(OpDepthInput::Waiting);
        }
        let expected = self.ledger.inputs_of_party(party, self.num_nodes);
        if shares.len() != expected {
            bail!("dealer {} dealt {} sharings, the ledger gives it {}", party, shares.len(), expected);
        }
        self.dealt.insert(party, shares);
        let dealers: Vec<usize> =
            (0..self.num_nodes).filter(|p| self.ledger.inputs_of_party(*p, self.num_nodes) > 0).collect();
        if !dealers.iter().all(|p| self.dealt.contains_key(p)) {
            return Ok(OpDepthInput::Waiting);
        }
        // Each dealer's vector is its accounts' balances in account order,
        // then its accounts' amounts in transfer order.
        let mut next: HashMap<usize, std::vec::IntoIter<FieldElement<F>>> =
            std::mem::take(&mut self.dealt).into_iter().map(|(p, v)| (p, v.into_iter())).collect();
        let mut take = |owner: usize| next.get_mut(&owner).and_then(|it| it.next()).expect("counted against the ledger");
        self.balances = (0..self.ledger.num_accounts).map(|a| take(a % self.num_nodes)).collect();
        self.amounts = self.ledger.transfers.iter().map(|(from, _)| take(from % self.num_nodes)).collect();
        self.inputs_bound = true;
        log::info!("Erc20: bound {} balances and {} amounts from {} dealers", self.balances.len(), self.amounts.len(), dealers.len());
        self.start_when_ready()
    }

    async fn on_preprocessing_complete(&mut self, _wires: RandomWireShares<F>) -> Result<OpDepthInput<F>> {
        self.preprocessing_done = true;
        self.start_when_ready()
    }

    /// Each epoch is three op-depths: the checks, `ok`, and the amount moved.
    async fn on_depth_complete(&mut self, depth: usize, result: OpResult<F>) -> Result<OpDepthInput<F>> {
        let (epoch, stage) = ((depth - 1) / 3, (depth - 1) % 3);
        let transfers = self.epochs[epoch].clone();
        let results = result.shares()?;
        let one = FieldElement::<F>::one();
        match stage {
            // x = [bal < amount], y = [amount < 0]: ok = (1 − x)(1 − y).
            0 => {
                let (x, y) = results.split_at(transfers.len());
                let op = Op::Mul {
                    x: x.iter().map(|x| &one - x).collect(),
                    y: y.iter().map(|y| &one - y).collect(),
                };
                OpDepthInput::op(depth + 1, op)
            }
            // δ = ok · amount.
            1 => {
                let amounts = transfers.iter().map(|&t| self.amounts[t].clone()).collect();
                OpDepthInput::op(depth + 1, Op::Mul { x: results, y: amounts })
            }
            // Move δ, then the next epoch.
            _ => {
                for (&t, delta) in transfers.iter().zip(results.iter()) {
                    let (from, to) = self.ledger.transfers[t];
                    self.balances[from] = &self.balances[from] - delta;
                    self.balances[to] = &self.balances[to] + delta;
                }
                self.check_funds_of_epoch(epoch + 1)
            }
        }
    }

    async fn on_output(&mut self, outputs: Vec<FieldElement<F>>) -> Result<()> {
        let p = (1i128 << <F as MersennePrimeField>::BITS) - 1;
        let signed: Vec<i128> = outputs
            .iter()
            .map(|e| {
                let c = <F as MersennePrimeField>::to_canonical_u64(e) as i128;
                if c <= (p - 1) / 2 { c } else { c - p }
            })
            .collect();
        log::info!("Erc20: {} final balances: {:?}", signed.len(), signed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velox::{fields::Mersenne61Field, Application, DepthInput, Planner, RandomWires};

    type F = Mersenne61Field;
    const N: usize = 4;

    fn elem(v: i128) -> FieldElement<F> {
        if v >= 0 { FieldElement::<F>::from(v as u64) } else { -FieldElement::<F>::from((-v) as u64) }
    }

    fn signed(e: &FieldElement<F>) -> i128 {
        let p = (1i128 << 61) - 1;
        let c = <F as MersennePrimeField>::to_canonical_u64(e) as i128;
        if c <= (p - 1) / 2 { c } else { c - p }
    }

    /// The transfers one by one, in the clear.
    fn reference(ledger: &Ledger, balances: &[i128], amounts: &[i128]) -> Vec<i128> {
        let mut bal = balances.to_vec();
        for (&(from, to), &amount) in ledger.transfers.iter().zip(amounts.iter()) {
            if amount >= 0 && bal[from] >= amount {
                bal[from] -= amount;
                bal[to] += amount;
            }
        }
        bal
    }

    /// Each party's inputs, from the balances and amounts.
    fn dealt(ledger: &Ledger, balances: &[i128], amounts: &[i128]) -> Vec<Vec<FieldElement<F>>> {
        (0..N)
            .map(|p| {
                let mut v: Vec<_> = (0..ledger.num_accounts).filter(|a| a % N == p).map(|a| elem(balances[a])).collect();
                v.extend(ledger.transfers.iter().zip(amounts).filter(|((f, _), _)| f % N == p).map(|(_, &x)| elem(x)));
                v
            })
            .collect()
    }

    /// The application hosted by the real Planner, over a plaintext engine.
    async fn run(ledger: Ledger, balances: &[i128], amounts: &[i128]) -> (Vec<i128>, usize) {
        let mut planner = Planner::new(Erc20::<F>::new(N, 0, ledger.clone())).unwrap();
        let wires: RandomWires = planner.random_wires();
        let mut s = 0x1234_5678u64;
        let bits = (0..wires.bits)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s & 1 == 1 { FieldElement::<F>::one() } else { -FieldElement::<F>::one() }
            })
            .collect();
        assert!(planner.on_preprocessing_complete(RandomWireShares::new(bits, Vec::new())).await.unwrap().is_waiting());
        let mut next = DepthInput::Waiting;
        for (p, shares) in dealt(&ledger, balances, amounts).into_iter().enumerate() {
            next = planner.input_sharing_termination(p, shares).await.unwrap();
        }
        let mut depths = 0;
        loop {
            next = match next {
                DepthInput::Done(out) => return (out.iter().map(signed).collect(), depths),
                DepthInput::Multiply { depth, x, y } => {
                    depths += 1;
                    planner.on_depth_complete(depth, x.iter().zip(&y).map(|(a, b)| a * b).collect()).await.unwrap()
                }
                DepthInput::Reveal { depth, values } => {
                    depths += 1;
                    planner.on_reveal_complete(depth, values).await.unwrap()
                }
                other => panic!("unexpected {:?}", other),
            };
        }
    }

    #[test]
    fn parses_and_groups_into_epochs() {
        let ledger = Ledger::parse("# token\naccounts 6\n0 1\n2 3\n1 4\n0 5\n4 2\n").unwrap();
        assert_eq!(ledger.num_accounts, 6);
        assert_eq!(ledger.transfers.len(), 5);
        // 1 received in epoch 1, so its send opens epoch 2; 0 sent in epoch 1
        // but not in epoch 2; 4 received in epoch 2, so it opens epoch 3.
        assert_eq!(ledger.epochs(), vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(ledger.inputs_of_party(0, N), 2 + 3); // accounts 0, 4; sends 0→1, 0→5, 4→2
        assert!(Ledger::parse("accounts 2\n0 2\n").is_err());
        assert!(Ledger::parse("0 1\n").is_err());

        // The tracked example: the epochs its header comment describes.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/erc20/example.ledger");
        let example = Ledger::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(example.epochs(), vec![vec![0, 1], vec![2, 3], vec![4], vec![5, 6]]);
    }

    /// Funded, overdrawn, negative and chained transfers against the
    /// one-by-one reference, in 10 engine rounds per epoch.
    #[tokio::test]
    async fn transfers_match_the_reference_ledger() {
        let ledger = Ledger::parse("accounts 6\n0 1\n2 3\n1 4\n3 0\n5 2\n4 5\n0 2\n").unwrap();
        let balances = [100, 5, 40, 0, 0, 7];
        let amounts = [30, 50, 35, 20, -3, 10, 70];
        let want = reference(&ledger, &balances, &amounts);
        let (got, depths) = run(ledger.clone(), &balances, &amounts).await;
        assert_eq!(got, want);
        assert_eq!(depths, 10 * ledger.epochs().len());
        assert_eq!(got.iter().sum::<i128>(), balances.iter().sum::<i128>(), "value is conserved");
    }

    #[test]
    fn declares_three_op_depths_per_epoch() {
        let ledger = Ledger::parse("accounts 4\n0 1\n2 3\n1 0\n").unwrap();
        let counts = Erc20::<F>::new(N, 0, ledger).preprocessing_count();
        let ops: Vec<(OpType, usize)> = counts.ops.iter().map(|p| (p.op_type, p.elements)).collect();
        assert_eq!(
            ops,
            vec![(OpType::Compare, 4), (OpType::Mul, 2), (OpType::Mul, 2), (OpType::Compare, 2), (OpType::Mul, 1), (OpType::Mul, 1)]
        );
        assert_eq!(counts.output, 4);
    }
}
