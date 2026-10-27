//! `CanonicalMod { x, d }` — `x mod d` for a shared `x` read as an integer in
//! `[0, p)` and a public `d` that divides `p − 1`, over a field with uniform
//! masks, exactly. ML-DSA-44 needs it with `d = 44`: the high bits `w1` of
//! `w` are `(−44·w + (p − 1)/2 mod p) mod 44` (Quorus, Protocol 8).
//!
//! The method is Quorus's `πMod-44`, with the final reduction that
//! Protocol 8 does not have. Let `g` be an element of order `d`, `r` the
//! mask and `ρ = r mod d`. Then `y = g^r = g^ρ`, and every function `f` of
//! `ρ` is a public linear combination of the powers of `y`:
//! `f(ρ) = Σ_i c_i·y^i` with `c_i = (1/d)·Σ_j f(j)·g^{−ij}`.
//!
//! As integers, `x = a − r + p·w` with `a = x + r mod p` and the wrap bit
//! `w = [a < r]`; and `p ≡ 1 mod d`, so `x mod d = (a₀ − ρ + w) mod d` with
//! `a₀ = a mod d`. With `v₀(ρ) = a₀ − ρ + d·[a₀ < ρ]`, which is in `[0, d)`:
//!
//! `x mod d = v₀(ρ) + w·g_w(ρ)`, with
//! `g_w(ρ) = 1 + d·([a₀ + 1 < ρ] − [a₀ < ρ]) − d·[a₀ = d − 1]·[ρ = 0]`.
//!
//! Rounds: `2 + L + P`, with `L = ⌈log₂ mask_bits⌉` and `P = ⌈log₂ (d − 1)⌉`
//! (13 over the ML-DSA field with `d = 44`). Steps:
//!
//! - step `0`, reveal: open `a = x + r mod p`. `a₀` is public.
//! - steps `1..=L`, multiply: per element, side by side, the carry tree for
//!   `[w] = [a < r]` and the product tree of
//!   `y = Π_i (1 + [r_i]·(g^{2^i} − 1))` over the bits of `r`.
//! - steps `L + 1..=L + P`, multiply: the powers `y^2, …, y^{d−1}`; step
//!   `L + j` makes `y^{s + i} = y^s · y^i` for `s = 2^{j−1}`.
//! - step `L + P + 1`, multiply: `[w]·[g_w(ρ)]`, and then `[v₀(ρ)] +` the
//!   product, locally.
//!
//! Multiplications per element over the ML-DSA field with `d = 44`: the
//! 23-bit carry tree (29), the product tree (22), the powers (42) and 1: 94.
//! Preprocessing: one edaBit per element.

use anyhow::{bail, Result};
use fields::ProtocolField;

use crate::{
    api::application::OpResult,
    primitives::{
        carry_tree::{CarryTree, Slot},
        edabit::EdaBit,
        mersenne,
    },
};

use super::{EngineOperationType, EngineOperands, Operation, OpStep, E};

/// The multiplications of `tree` at `level`; 0 once the tree has folded.
fn tree_width(tree: &CarryTree, level: usize) -> usize {
    tree.level_widths().get(level).copied().unwrap_or(0)
}

/// The multiplications at each level of a product tree over `factors`
/// values: adjacent pairs, the odd one carried up.
fn product_widths(factors: usize) -> Vec<usize> {
    let mut widths = Vec::new();
    let mut count = factors;
    while count > 1 {
        widths.push(count / 2);
        count = count.div_ceil(2);
    }
    widths
}

/// The multiplications of each power step: step `j` (from 0) makes
/// `y^{s+1}..y^{min(2s, d−1)}` from `y^s`, `s = 2^j`.
fn power_widths(d: usize) -> Vec<usize> {
    let mut widths = Vec::new();
    let mut known = 1;
    while known < d - 1 {
        widths.push(known.min(d - 1 - known));
        known *= 2;
    }
    widths
}

/// The engine steps of the op, as in the module doc. `bits` is the edaBit
/// width, and `d` is the public divisor.
pub fn steps(bits: usize, d: usize) -> Vec<OpStep> {
    let wrap = CarryTree::new(bits);
    let product = product_widths(bits);
    let levels = wrap.num_levels().max(product.len());
    let mut steps = vec![OpStep::new(EngineOperationType::Reveal, 1)];
    steps.extend((0..levels).map(|l| {
        OpStep::new(EngineOperationType::Multiply, tree_width(&wrap, l) + product.get(l).copied().unwrap_or(0))
    }));
    steps.extend(power_widths(d).into_iter().map(|w| OpStep::new(EngineOperationType::Multiply, w)));
    steps.push(OpStep::new(EngineOperationType::Multiply, 1));
    steps
}

/// `x mod d` for each shared `x`; see the module doc for the steps.
pub struct CanonicalMod<F: ProtocolField> {
    /// The tree for the wrap bit `[a < r]`, over all the bits of `r`.
    wrap_tree: CarryTree,
    d: usize,
    x: Vec<E<F>>,
    eda: Vec<EdaBit<F>>,
    /// For each public `a₀` in `[0, d)`: the coefficients of `v₀` and of
    /// `g_w` in the powers `y^0..y^{d−1}`.
    coefficients: Vec<[Vec<E<F>>; 2]>,
    /// After the reveal: per element, `a₀`.
    a_low: Vec<usize>,
    /// After the reveal: per element, the wrap tree's slots at the current
    /// level.
    wrap_slots: Vec<Vec<Slot<F>>>,
    /// Per element, the product tree's values at the current level: at the
    /// start the factors `1 + [r_i]·(g^{2^i} − 1)`, at the end `[y]`.
    factors: Vec<Vec<E<F>>>,
    /// After the trees: per element, `[y^0], [y^1], …`.
    powers: Vec<Vec<E<F>>>,
    /// After the trees: per element, the wrap bit `[w]`.
    w: Vec<E<F>>,
    /// After the powers: per element, `[v₀(ρ)]` and `[g_w(ρ)]`.
    v0: Vec<E<F>>,
    gw: Vec<E<F>>,
    /// `[x mod d]`, per element, after the last step.
    out: Vec<E<F>>,
}

impl<F: ProtocolField> CanonicalMod<F> {
    /// `bits` is the edaBit width, and `eda` holds one edaBit per element of
    /// `x`. `d` divides `p − 1`.
    pub fn new(x: Vec<E<F>>, d: usize, bits: usize, eda: Vec<EdaBit<F>>) -> Self {
        let p = F::PRIME_MODULUS.expect("CanonicalMod runs over a prime field");
        let g = element_of_order::<F>(p, d);
        let one = E::<F>::one();
        let mut g_pow2 = g.clone();
        let mut steps_of_g = Vec::with_capacity(bits);
        for _ in 0..bits {
            steps_of_g.push(&g_pow2 - &one);
            g_pow2 = g_pow2.square();
        }
        let factors =
            eda.iter().map(|r| r.bits.iter().zip(&steps_of_g).map(|(bit, step)| &one + bit * step).collect()).collect();
        Self {
            wrap_tree: CarryTree::new(bits),
            d,
            x,
            coefficients: coefficient_table::<F>(&g, d),
            eda,
            a_low: Vec::new(),
            wrap_slots: Vec::new(),
            factors,
            powers: Vec::new(),
            w: Vec::new(),
            v0: Vec::new(),
            gw: Vec::new(),
            out: Vec::new(),
        }
    }

    /// The tree steps: the levels of the deeper tree.
    fn tree_levels(&self) -> usize {
        self.wrap_tree.num_levels().max(product_widths(self.wrap_tree.bits()).len())
    }

    /// `[v₀(ρ)]` and `[g_w(ρ)]` per element, from the powers of `y` and the
    /// coefficients of its `a₀`.
    fn finish_powers(&mut self) {
        for (a0, powers) in self.a_low.iter().zip(&self.powers) {
            let [c_v0, c_gw] = &self.coefficients[*a0];
            let combine = |c: &[E<F>]| c.iter().zip(powers).fold(E::<F>::zero(), |acc, (c, y)| acc + c * y);
            self.v0.push(combine(c_v0));
            self.gw.push(combine(c_gw));
        }
    }
}

/// An element of order exactly `d` in `F_p^*`: `z^{(p−1)/d}` for the first
/// `z = 2, 3, …` that gives order `d`.
fn element_of_order<F: ProtocolField>(p: u128, d: usize) -> E<F> {
    let d = d as u128;
    let primes: Vec<u128> = (2..=d).filter(|f| d % f == 0 && (2..*f).all(|k| f % k != 0)).collect();
    (2u64..)
        .map(|z| E::<F>::from(z).pow((p - 1) / d))
        .find(|g| primes.iter().all(|f| g.pow(d / f) != E::<F>::one()))
        .expect("F_p^* is cyclic, so an element of order d exists when d divides p − 1")
}

/// For each `a₀` in `[0, d)`: the coefficients `c_i = (1/d)·Σ_j f(j)·g^{−ij}`
/// of `v₀` and of `g_w` (see the module doc).
fn coefficient_table<F: ProtocolField>(g: &E<F>, d: usize) -> Vec<[Vec<E<F>>; 2]> {
    let g_inv = g.inv().expect("g has order d, so it is not zero");
    let inv_powers: Vec<E<F>> = (0..d).scan(E::<F>::one(), |acc, _| {
        let current = acc.clone();
        *acc = &*acc * &g_inv;
        Some(current)
    }).collect();
    let d_inv = E::<F>::from(d as u64).inv().expect("d divides p − 1, so it is below p");
    let field = |v: i64| if v < 0 { -E::<F>::from(v.unsigned_abs()) } else { E::<F>::from(v as u64) };
    let interpolate = |f: &dyn Fn(usize) -> i64| -> Vec<E<F>> {
        (0..d)
            .map(|i| &d_inv * (0..d).fold(E::<F>::zero(), |acc, j| acc + field(f(j)) * &inv_powers[(i * j) % d]))
            .collect()
    };
    let d_signed = d as i64;
    (0..d)
        .map(|a0| {
            let below = |bound: usize, rho: usize| i64::from(bound < rho);
            let v0 = interpolate(&|rho| a0 as i64 - rho as i64 + d_signed * below(a0, rho));
            let gw = interpolate(&|rho| {
                1 + d_signed * (below(a0 + 1, rho) - below(a0, rho)) - d_signed * i64::from(a0 == d - 1 && rho == 0)
            });
            [v0, gw]
        })
        .collect()
}

impl<F: ProtocolField> Operation<F> for CanonicalMod<F> {
    fn operands(&self, step: usize) -> EngineOperands<F> {
        let levels = self.tree_levels();
        let power_steps = power_widths(self.d).len();
        match step {
            0 => EngineOperands::Reveal(self.x.iter().zip(&self.eda).map(|(x, r)| x + &r.value).collect()),
            s if s <= levels => {
                let level = s - 1;
                EngineOperands::multiply(self.wrap_slots.iter().zip(&self.factors).flat_map(|(slots, factors)| {
                    // A tree that has folded takes no gates.
                    let mut pairs =
                        if level < self.wrap_tree.num_levels() { self.wrap_tree.level_operands(level, slots) } else { Vec::new() };
                    pairs.extend(factors.chunks_exact(2).map(|pair| (pair[0].clone(), pair[1].clone())));
                    pairs
                }))
            }
            s if s <= levels + power_steps => {
                let known = 1 << (s - levels - 1);
                let count = known.min(self.d - 1 - known);
                EngineOperands::multiply(
                    self.powers.iter().flat_map(|y| (1..=count).map(move |i| (y[known].clone(), y[i].clone()))),
                )
            }
            _ => EngineOperands::multiply(self.w.iter().cloned().zip(self.gw.iter().cloned())),
        }
    }

    fn on_step_complete(&mut self, step: usize, results: Vec<E<F>>) -> Result<()> {
        let levels = self.tree_levels();
        let power_steps = power_widths(self.d).len();
        match step {
            // The reveal: the bits of `a` seed the wrap tree.
            0 => {
                let bits = self.wrap_tree.bits();
                let one = E::<F>::one();
                for (opened, eda) in results.iter().zip(&self.eda) {
                    let a = mersenne::canonical::<F>(opened);
                    let a_bits: Vec<u64> = (0..bits).map(|i| ((a >> i) & 1) as u64).collect();
                    let not_r: Vec<E<F>> = eda.bits.iter().map(|r| &one - r).collect();
                    self.wrap_slots.push(self.wrap_tree.leaves(&a_bits, &not_r));
                    self.a_low.push((a % self.d as u128) as usize);
                }
            }
            // A tree level: each element's products are the wrap tree's, then
            // the product tree's pairs; an odd factor is carried up.
            s if s <= levels => {
                let level = s - 1;
                let wrap_width = tree_width(&self.wrap_tree, level);
                let mut results = results.into_iter();
                for (slots, factors) in self.wrap_slots.iter_mut().zip(self.factors.iter_mut()) {
                    let wrap: Vec<E<F>> = results.by_ref().take(wrap_width).collect();
                    if wrap_width > 0 {
                        *slots = self.wrap_tree.level_absorb(level, slots, &wrap);
                    }
                    let mut next: Vec<E<F>> = results.by_ref().take(factors.len() / 2).collect();
                    if factors.len() % 2 == 1 {
                        next.push(factors.last().expect("odd, so not empty").clone());
                    }
                    *factors = next;
                }
            }
            // A power step: the new powers follow the known ones.
            s if s <= levels + power_steps => {
                let known = 1 << (s - levels - 1);
                let count = known.min(self.d - 1 - known);
                for (powers, chunk) in self.powers.iter_mut().zip(results.chunks(count)) {
                    powers.extend_from_slice(chunk);
                }
            }
            // The selection: `[x mod d] = [v₀] + [w]·[g_w]`.
            s if s == levels + power_steps + 1 => {
                self.out = self.v0.iter().zip(results).map(|(v0, product)| v0 + product).collect();
                return Ok(());
            }
            s => bail!("CanonicalMod {{ d: {} }} has no step {}", self.d, s),
        }
        if step == levels {
            let one = E::<F>::one();
            self.w = self.wrap_slots.iter().map(|slots| &one - self.wrap_tree.carry(slots)).collect();
            self.powers = self.factors.iter().map(|y| vec![one.clone(), y[0].clone()]).collect();
        }
        if step == levels + power_steps {
            self.finish_powers();
        }
        Ok(())
    }

    fn into_result(self: Box<Self>) -> OpResult<F> {
        OpResult::Shares(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{Mersenne61Field, MlDsaField};

    /// Runs the op with a plaintext engine (a plaintext value is a valid
    /// sharing, a multiply is the product, a reveal the identity) and checks
    /// `x mod d` for each `(x, r)`.
    fn check_in_plaintext<F: ProtocolField>(cases: &[(u128, u128)], d: usize, bits: usize) {
        let x = cases.iter().map(|&(x, _)| mersenne::from_u128::<F>(x)).collect();
        let eda = cases
            .iter()
            .map(|&(_, r)| {
                let signs: Vec<E<F>> = (0..bits).map(|i| if (r >> i) & 1 == 1 { E::<F>::one() } else { -E::<F>::one() }).collect();
                EdaBit::from_signs(&signs, bits).unwrap()
            })
            .collect();
        let mut op = Box::new(CanonicalMod::new(x, d, bits, eda));
        for (step, planned) in steps(bits, d).iter().enumerate() {
            let operands = op.operands(step);
            assert_eq!(operands.batch_type(), planned.engine_op, "step {step}");
            assert_eq!(operands.len(), planned.num_engine_operands * cases.len(), "step {step}");
            let results = match operands {
                EngineOperands::Multiply { x, y } => x.iter().zip(&y).map(|(a, b)| a * b).collect(),
                EngineOperands::Reveal(v) => v,
                EngineOperands::MaskedMultiply { .. } => unreachable!(),
            };
            op.on_step_complete(step, results).unwrap();
        }
        let OpResult::Shares(out) = op.into_result() else { panic!("CanonicalMod returns shares") };
        for (&(x, r), got) in cases.iter().zip(out) {
            assert_eq!(mersenne::canonical::<F>(&got), x % d as u128, "x = {x}, r = {r}, d = {d}");
        }
    }

    /// Edge values for `x` and `r` over the ML-DSA field, at `d = 44` and
    /// at small divisors of `q − 1`: both wraps and `a₀ = d − 1` occur.
    #[test]
    fn edges_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let edges = [0, 1, 2, 43, 44, 45, q / 2, q - 45, q - 44, q - 2, q - 1];
        let mut cases = Vec::new();
        for x in edges {
            cases.extend(edges.map(|r| (x, r)));
        }
        for d in [2, 3, 4, 44] {
            check_in_plaintext::<MlDsaField>(&cases, d, 23);
        }
    }

    /// Random `x` and `r` over the ML-DSA field at `d = 44`, and the values
    /// that ML-DSA-44 reduces: `−44·w + (q − 1)/2 mod q`.
    #[test]
    fn random_values_over_the_ml_dsa_field() {
        let q = fields::ml_dsa::ML_DSA_PRIME_FIELD_ORDER as u128;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u128::from(state) % q
        };
        let mut cases: Vec<(u128, u128)> = (0..1500).map(|_| (next(), next())).collect();
        cases.extend((0..1500).map(|_| (((q - 44) * next() + (q - 1) / 2) % q, next())));
        check_in_plaintext::<MlDsaField>(&cases, 44, 23);
    }

    /// Over a Mersenne field the edaBit can be `r = p` (all ones). 22
    /// divides `2^61 − 2`.
    #[test]
    fn a_mersenne_mask_of_all_ones() {
        let p = (1u128 << 61) - 1;
        let cases = [(0, p), (5, p), (p - 1, p), (p - 1, 3), (12345, p - 7), (p / 3, 1)];
        check_in_plaintext::<Mersenne61Field>(&cases, 22, 61);
    }

    #[test]
    fn thirteen_rounds_and_94_multiplications_at_d_44() {
        let steps = steps(23, 44);
        let multiplies: usize =
            steps.iter().filter(|s| s.engine_op == EngineOperationType::Multiply).map(|s| s.num_engine_operands).sum();
        assert_eq!((steps.len(), multiplies), (13, 94));
    }
}
