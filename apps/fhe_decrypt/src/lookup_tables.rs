//! Method B: the noise `z mod 2^63` from preprocessed lookup tables
//! (Zyskind et al., CCS 2025, §3.3 and §4).
//!
//! Both methods compute `[e'] = [z mod L]`, `L = 2^63`, as
//! `e' = z' − r + L·u` for an opened `z' = (z + r) mod L` and the bit
//! `u = [z' < r]`. Method A finds `u` with the carry tree after `z'` is
//! known. Here `u` comes from tables built before `z` exists, so after `z` is
//! known only two values are opened and nothing is multiplied:
//!
//! - **Sign tables.** `r` has 8 base-`2^8` digits `r_i` (the top one 7 bits).
//!   Table `i` holds `[Sign(x − r_i)]` for every digit value `x`. Looking up
//!   the digits `z'_i` of the opened `z'` gives, locally,
//!   `[y] = Σ_i 2^i·[Sign(z'_i − r_i)]`, whose sign is the sign of `z' − r`
//!   (Lemma 4): the most significant differing digit outweighs the rest.
//! - **The ModLTZ table.** `y ∈ (−D, D)`, `D = 2^8`, is opened masked by a
//!   9-bit `r_y`, and the table of `[ModLTZ_9(x − r_y)]` for every `x` mod
//!   `2D` gives `[u] = [y < 0]` (Equation 4).
//!
//! A table is a linear function of the products of every subset of its mask's
//! bits (§4.1): [`SubsetProductTree`] schedules those products, and
//! [`subset_products_to_signs`], [`subset_products_to_carries`] and
//! [`subset_products_to_mod_ltz`] are Procedures 6, 9 and 8. The products are
//! the only multiplications, four rounds of them, before `z` is needed.
//!
//! Over a prime field nothing wraps at `2^63` or `2D`, so the two openings
//! carry a second mask above the table's (`r_hi`, `R_y`) that hides the high
//! part up to a statistical distance of `2^−40`; the receiver reduces the
//! opened value mod `L` or `2D` itself.

use lambdaworks_math::field::traits::IsField;
use velox::{fields::Mersenne127Field, FieldElement, MersennePrimeField};

use crate::NOISE_BITS;

type F = Mersenne127Field;
type E = FieldElement<F>;

/// Bits of a full digit of `r`, `b`.
pub const DIGIT_BITS: usize = 8;
/// Digits of the 63-bit `r`, `d = ⌈63 / 8⌉`; the top digit has 7 bits.
pub const DIGITS: usize = NOISE_BITS.div_ceil(DIGIT_BITS);
/// Bits of the ModLTZ table's input and mask, `d + 1`.
pub const LTZ_BITS: usize = DIGITS + 1;
/// `2^63 + 2^75 · 2^40`: the bits of `r_hi`, which masks `z`'s high part.
pub const HIGH_MASK_BITS: usize = 53;
/// `2^10 · 2^40 / 2^9`: the bits of `R_y`, which masks `y`'s high part.
pub const LTZ_HIGH_MASK_BITS: usize = 41;
/// Random bits per ciphertext: `r` (63), `r_y` (9), `r_hi` (53), `R_y` (41).
pub const MASK_BITS: usize = NOISE_BITS + LTZ_BITS + HIGH_MASK_BITS + LTZ_HIGH_MASK_BITS;

/// `D = 2^d`: the weighted sum `y` lies in `(−D, D)`.
const D: u64 = 1 << DIGITS;

/// Bits of digit `i` of `r`.
fn digit_bits(i: usize) -> usize {
    DIGIT_BITS.min(NOISE_BITS - i * DIGIT_BITS)
}

/// One multiplication round's join: the products of block `out` are the
/// products of block `lo` (the low `lo_bits` bits) times those of block `hi`.
struct Join {
    lo: usize,
    hi: usize,
    lo_bits: usize,
    hi_bits: usize,
    out: usize,
}

/// The schedule of PrepSubsetProds (Procedure 4) for an `a`-bit mask: all
/// `2^a` products of subsets of its bits, in `⌈log₂ a⌉` rounds of
/// `2^a − a − 1` multiplications in all.
///
/// Blocks of bits are nodes: the `a` leaves are the single bits, `[1, r_i]`,
/// and a block of `k` bits joins its low `⌈k/2⌉` bits with its high `⌊k/2⌋`.
/// A block's products are indexed by the subset as a bitmask, bit `j` for the
/// block's bit `j`, so index 0 is the empty product, 1. A block of height `h`
/// is joined in round `h`, when both halves are complete; a join multiplies
/// only the pairs of non-empty subsets, as a product with the empty subset is
/// the other factor.
pub struct SubsetProductTree {
    nodes: usize,
    root: usize,
    /// `rounds[h]` joins the blocks of height `h + 1`.
    rounds: Vec<Vec<Join>>,
}

impl SubsetProductTree {
    pub fn new(bits: usize) -> Self {
        fn build(first: usize, count: usize, nodes: &mut usize, rounds: &mut Vec<Vec<Join>>) -> (usize, usize) {
            if count == 1 {
                return (first, 0);
            }
            let lo_bits = count.div_ceil(2);
            let (lo, lo_height) = build(first, lo_bits, nodes, rounds);
            let (hi, hi_height) = build(first + lo_bits, count - lo_bits, nodes, rounds);
            let height = 1 + lo_height.max(hi_height);
            if rounds.len() < height {
                rounds.resize_with(height, Vec::new);
            }
            let out = *nodes;
            *nodes += 1;
            rounds[height - 1].push(Join { lo, hi, lo_bits, hi_bits: count - lo_bits, out });
            (out, height)
        }
        assert!(bits >= 1, "a mask needs at least one bit");
        let (mut nodes, mut rounds) = (bits, Vec::new());
        let (root, _) = build(0, bits, &mut nodes, &mut rounds);
        Self { nodes, root, rounds }
    }

    pub fn rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Multiplications in round `round` (0 past the last).
    pub fn round_width(&self, round: usize) -> usize {
        self.rounds.get(round).map_or(0, |joins| {
            joins.iter().map(|j| ((1 << j.lo_bits) - 1) * ((1 << j.hi_bits) - 1)).sum()
        })
    }

    /// Every block's products: the leaves `[1, r_i]`, the rest still empty.
    pub fn leaves<G: IsField>(&self, bits: &[FieldElement<G>]) -> Vec<Vec<FieldElement<G>>> {
        let mut nodes = vec![Vec::new(); self.nodes];
        for (node, bit) in nodes.iter_mut().zip(bits) {
            *node = vec![FieldElement::one(), bit.clone()];
        }
        nodes
    }

    /// The pairs round `round` multiplies: per join, every non-empty high
    /// subset times every non-empty low subset.
    pub fn round_operands<G: IsField>(
        &self,
        round: usize,
        nodes: &[Vec<FieldElement<G>>],
    ) -> Vec<(FieldElement<G>, FieldElement<G>)> {
        let mut pairs = Vec::with_capacity(self.round_width(round));
        for join in self.rounds.get(round).into_iter().flatten() {
            for high in &nodes[join.hi][1..] {
                for low in &nodes[join.lo][1..] {
                    pairs.push((low.clone(), high.clone()));
                }
            }
        }
        pairs
    }

    /// Fill the blocks round `round` joins from its products, in
    /// [`round_operands`](Self::round_operands)'s order.
    pub fn round_absorb<G: IsField>(&self, round: usize, nodes: &mut [Vec<FieldElement<G>>], products: &[FieldElement<G>]) {
        let mut products = products.iter();
        for join in self.rounds.get(round).into_iter().flatten() {
            let mut out = Vec::with_capacity(1 << (join.lo_bits + join.hi_bits));
            for high in 0..1 << join.hi_bits {
                for low in 0..1 << join.lo_bits {
                    out.push(match (low, high) {
                        (_, 0) => nodes[join.lo][low].clone(),
                        (0, _) => nodes[join.hi][high].clone(),
                        _ => products.next().expect("one product per pair of non-empty subsets").clone(),
                    });
                }
            }
            nodes[join.out] = out;
        }
    }

    /// The products of every subset of all the bits, once the last round is in.
    pub fn products<G: IsField>(&self, nodes: &mut [Vec<FieldElement<G>>]) -> Vec<FieldElement<G>> {
        std::mem::take(&mut nodes[self.root])
    }
}

/// Procedure 6: from the subset products `p` of the bits of `r ∈ [2^b]`
/// (`p.len() = 2^b`), the table `Sign(x − r)` for every `x ∈ [2^b]`.
///
/// The top bits decide unless they are equal (Equation 6):
/// `Sign(x − r) = (x_top − r_top) + Sign(x' − r')·(x_top ? r_top : 1 − r_top)`.
/// The upper half of `p` is `r_top` times the lower half, so by linearity the
/// lower half's table times `r_top` is the upper half's table.
pub fn subset_products_to_signs<G: IsField>(p: &[FieldElement<G>]) -> Vec<FieldElement<G>> {
    if p.len() == 1 {
        return vec![FieldElement::zero()];
    }
    let (p0, p1) = p.split_at(p.len() / 2);
    let (s0, s1) = (subset_products_to_signs(p0), subset_products_to_signs(p1));
    let low = s0.iter().zip(&s1).map(|(a, b)| a - b - &p1[0]);
    let high = s1.iter().map(|b| b + &p0[0] - &p1[0]);
    low.chain(high).collect()
}

/// Procedure 9: from the subset products `p` of the bits of `r ∈ [2^d]`, the
/// table `Carry_d(x, r̄)` — the carry out of `x + r̄ + 1` with `r̄ = 2^d − 1 − r`,
/// which is `[x ≥ r]` — for every `x ∈ [2^d]` (Equation 7).
pub fn subset_products_to_carries<G: IsField>(p: &[FieldElement<G>]) -> Vec<FieldElement<G>> {
    if p.len() == 1 {
        return vec![p[0].clone()];
    }
    let (p0, p1) = p.split_at(p.len() / 2);
    let (c0, c1) = (subset_products_to_carries(p0), subset_products_to_carries(p1));
    let low = c0.iter().zip(&c1).map(|(a, b)| a - b);
    let high = c1.iter().map(|b| b + &p0[0] - &p1[0]);
    low.chain(high).collect()
}

/// Procedure 8: from the subset products `p` of the bits of `r ∈ Z_{2D}`, the
/// table `ModLTZ_{d+1}(x − r)` — whether `(x − r) mod 2D ≥ D` — for every
/// `x ∈ Z_{2D}`. That is bit `d` of `x + r̄ + 1`: `x_d ⊕ r̄_d ⊕ Carry_d`
/// (Equation 8).
pub fn subset_products_to_mod_ltz<G: IsField>(p: &[FieldElement<G>]) -> Vec<FieldElement<G>> {
    let (p0, p1) = p.split_at(p.len() / 2);
    let (c0, c1) = (subset_products_to_carries(p0), subset_products_to_carries(p1));
    let low = c0.iter().zip(&c1).map(|(a, b)| &p0[0] - &p1[0] - a + b.double());
    let high = c0.iter().zip(&c1).map(|(a, b)| &p1[0] + a - b.double());
    low.chain(high).collect()
}

/// `Σ 2^j·[b_j]` of 0/1 sharings, LSB first.
fn compose(bits: &[E]) -> E {
    bits.iter().rev().fold(E::zero(), |acc, bit| acc.double() + bit)
}

/// The paper's LTRand, over a batch of ciphertexts: the tables' preparation
/// (four `Mul` op-depths), then the two openings that turn `[z]` into the
/// noise `[z mod 2^63]`.
pub struct LtRand {
    /// One schedule per table: the `DIGITS` Sign tables, then ModLTZ.
    trees: Vec<SubsetProductTree>,
    /// `[ciphertext][table][block]`, while the tables are being prepared.
    blocks: Vec<Vec<Vec<Vec<E>>>>,
    /// `[ciphertext][digit][x]`, once prepared.
    sign_tables: Vec<Vec<Vec<E>>>,
    /// `[ciphertext][x]`, once prepared.
    ltz_tables: Vec<Vec<E>>,
    /// Per ciphertext: `[r]`, `[r_hi]`, `[r_y]`, `[R_y]`.
    r: Vec<E>,
    r_hi: Vec<E>,
    r_y: Vec<E>,
    r_y_high: Vec<E>,
    /// Per ciphertext, the opened `z' = (z + r) mod L`.
    z_low: Vec<u64>,
}

impl LtRand {
    fn trees() -> Vec<SubsetProductTree> {
        (0..DIGITS).map(|i| SubsetProductTree::new(digit_bits(i))).chain([SubsetProductTree::new(LTZ_BITS)]).collect()
    }

    /// Multiplications per ciphertext in each preparation round, which the
    /// application declares as its `Mul` op-depths before the tables exist.
    pub fn preparation_widths() -> Vec<usize> {
        let trees = Self::trees();
        let rounds = trees.iter().map(|t| t.rounds()).max().unwrap_or(0);
        (0..rounds).map(|round| trees.iter().map(|t| t.round_width(round)).sum()).collect()
    }

    /// From [`MASK_BITS`] 0/1 sharings per ciphertext: `r`'s 63 bits, then
    /// `r_y`'s 9, then `r_hi`'s 53, then `R_y`'s 41.
    pub fn new(bits: &[E]) -> Self {
        let trees = Self::trees();
        let (mut blocks, mut r, mut r_hi, mut r_y, mut r_y_high) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for ct_bits in bits.chunks(MASK_BITS) {
            let (r_bits, rest) = ct_bits.split_at(NOISE_BITS);
            let (r_y_bits, rest) = rest.split_at(LTZ_BITS);
            let (r_hi_bits, r_y_high_bits) = rest.split_at(HIGH_MASK_BITS);
            // The digit masks are r's bits 8 at a time; the ModLTZ mask is r_y.
            let masks = r_bits.chunks(DIGIT_BITS).chain([r_y_bits]);
            blocks.push(trees.iter().zip(masks).map(|(tree, mask)| tree.leaves(mask)).collect());
            r.push(compose(r_bits));
            r_y.push(compose(r_y_bits));
            r_hi.push(compose(r_hi_bits));
            r_y_high.push(compose(r_y_high_bits));
        }
        Self { trees, blocks, sign_tables: Vec::new(), ltz_tables: Vec::new(), r, r_hi, r_y, r_y_high, z_low: Vec::new() }
    }

    /// The `Mul` operands of preparation round `round`: per ciphertext, per
    /// table, that table's pairs.
    pub fn preparation_operands(&self, round: usize) -> (Vec<E>, Vec<E>) {
        self.blocks
            .iter()
            .flat_map(|tables| self.trees.iter().zip(tables).flat_map(move |(tree, blocks)| tree.round_operands(round, blocks)))
            .unzip()
    }

    /// Take round `round`'s products; after the last round, turn each mask's
    /// subset products into its table. Returns whether the tables are ready.
    pub fn preparation_complete(&mut self, round: usize, products: &[E]) -> bool {
        let mut products = products;
        for tables in self.blocks.iter_mut() {
            for (tree, blocks) in self.trees.iter().zip(tables.iter_mut()) {
                let (mine, rest) = products.split_at(tree.round_width(round));
                tree.round_absorb(round, blocks, mine);
                products = rest;
            }
        }
        if round + 1 < self.trees.iter().map(|t| t.rounds()).max().unwrap_or(0) {
            return false;
        }
        for mut tables in std::mem::take(&mut self.blocks) {
            let mut digit_tables: Vec<Vec<E>> = self
                .trees
                .iter()
                .zip(tables.iter_mut())
                .map(|(tree, blocks)| tree.products(blocks))
                .collect();
            let ltz = digit_tables.pop().expect("the ModLTZ mask is the last");
            self.sign_tables.push(digit_tables.iter().map(|p| subset_products_to_signs(p)).collect());
            self.ltz_tables.push(subset_products_to_mod_ltz(&ltz));
        }
        true
    }

    /// The first opening: `[z] + [r] + L·[r_hi]`. `z + r < 2^76`, so `r_hi`'s
    /// 53 bits hide the high part to `2^76 / 2^116 = 2^−40`.
    pub fn mask_inputs(&self, z: &[E]) -> Vec<E> {
        let l = E::from(1u64 << NOISE_BITS);
        z.iter().zip(&self.r).zip(&self.r_hi).map(|((z, r), r_hi)| z + r + &l * r_hi).collect()
    }

    /// After the first opening: look up `Sign(z'_i − r_i)` for the digits of
    /// `z' = c mod L`, and give the second opening,
    /// `[y] + 2D + [r_y] + 2D·[R_y]`. `y + 2D + r_y ∈ [D, 4D) ⊂ [0, 2^10)`, so
    /// `R_y`'s 41 bits hide it to `2^10 / 2^50 = 2^−40`.
    pub fn look_up_signs(&mut self, opened: &[E]) -> Vec<E> {
        let two_d = E::from(2 * D);
        let mut second = Vec::with_capacity(opened.len());
        for (i, c) in opened.iter().enumerate() {
            let z_low = (F::to_canonical_u128(c) & ((1u128 << NOISE_BITS) - 1)) as u64;
            let y = (0..DIGITS).fold(E::zero(), |acc, digit| {
                let x = (z_low >> (digit * DIGIT_BITS)) as usize & ((1 << digit_bits(digit)) - 1);
                acc + E::from(1u64 << digit) * &self.sign_tables[i][digit][x]
            });
            second.push(y + &two_d + &self.r_y[i] + &two_d * &self.r_y_high[i]);
            self.z_low.push(z_low);
        }
        second
    }

    /// After the second opening: `[u] = ModLTZ(x' − r_y)` at `x' = c mod 2D`,
    /// which is `[y < 0] = [z' < r]`, and the noise
    /// `[z mod L] = z' − [r] + L·[u]` (Procedure 1).
    pub fn noise(&self, opened: &[E]) -> Vec<E> {
        let l = E::from(1u64 << NOISE_BITS);
        opened
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let x = (F::to_canonical_u128(c) % (2 * D as u128)) as usize;
                E::from(self.z_low[i]) - &self.r[i] + &l * &self.ltz_tables[i][x]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_of(r: u64, width: usize) -> Vec<E> {
        (0..width).map(|j| E::from((r >> j) & 1)).collect()
    }

    /// The subset products of `r`'s bits, computed directly.
    fn direct_products(r: u64, width: usize) -> Vec<E> {
        (0..1u64 << width).map(|subset| E::from(u64::from(subset & r == subset))).collect()
    }

    /// The tree's products, with the multiplications done in the clear.
    fn tree_products(r: u64, width: usize) -> Vec<E> {
        let tree = SubsetProductTree::new(width);
        let mut blocks = tree.leaves(&bits_of(r, width));
        for round in 0..tree.rounds() {
            let products: Vec<E> = tree.round_operands(round, &blocks).iter().map(|(a, b)| a * b).collect();
            assert_eq!(products.len(), tree.round_width(round));
            tree.round_absorb(round, &mut blocks, &products);
        }
        tree.products(&mut blocks)
    }

    fn signed(v: i64) -> E {
        if v >= 0 { E::from(v as u64) } else { -E::from((-v) as u64) }
    }

    #[test]
    fn subset_products_match_the_direct_products() {
        for width in 1..=9 {
            for r in [0, 1, (1 << width) - 1, 0x155 & ((1 << width) - 1), 0xaa & ((1 << width) - 1)] {
                assert_eq!(tree_products(r, width), direct_products(r, width), "width {width}, r={r:b}");
            }
            let tree = SubsetProductTree::new(width);
            let total: usize = (0..tree.rounds()).map(|round| tree.round_width(round)).sum();
            assert_eq!(total, (1 << width) - width - 1, "2^a − a − 1 multiplications at width {width}");
            assert_eq!(tree.rounds(), (width as f64).log2().ceil() as usize);
        }
    }

    /// The widths the paper's §5 counts: 2351 multiplications per ciphertext.
    #[test]
    fn preparation_rounds() {
        let widths = |a| { let t = SubsetProductTree::new(a); (0..t.rounds()).map(|r| t.round_width(r)).collect::<Vec<_>>() };
        assert_eq!(widths(8), vec![4, 18, 225]);
        assert_eq!(widths(7), vec![3, 12, 105]);
        assert_eq!(widths(9), vec![4, 12, 21, 465]);
        assert_eq!(LtRand::preparation_widths(), vec![35, 150, 1701, 465]);
        assert_eq!(MASK_BITS, 166);
    }

    #[test]
    fn tables_match_their_definitions() {
        for width in 1..=6 {
            let size = 1i64 << width;
            for r in 0..size {
                let p = direct_products(r as u64, width);
                let signs = subset_products_to_signs(&p);
                let carries = subset_products_to_carries(&p);
                let ltz = subset_products_to_mod_ltz(&p);
                for x in 0..size {
                    assert_eq!(signs[x as usize], signed((x - r).signum()), "Sign({x} − {r}), {width} bits");
                    assert_eq!(carries[x as usize], E::from(u64::from(x >= r)), "Carry({x}, ¬{r}), {width} bits");
                    let wrapped = (x - r).rem_euclid(size);
                    assert_eq!(ltz[x as usize], E::from(u64::from(wrapped >= size / 2)), "ModLTZ({x} − {r}), {width} bits");
                }
            }
        }
    }

    /// LTRand end to end in the clear, for `z` across its range and the
    /// masks at their edges: the noise is `z mod 2^63`.
    #[test]
    fn noise_is_z_mod_l() {
        let mut state = 0x9e37u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let zs: Vec<u128> = [0, 1, (1 << 63) - 1, 1 << 63, (1 << 75) - 1]
            .into_iter()
            .chain((0..40).map(|_| (((next() as u128) << 64) | next() as u128) % (1 << 75)))
            .collect();
        let mut bits = Vec::new();
        for i in 0..zs.len() {
            // All-zero, all-one and random masks.
            bits.extend((0..MASK_BITS).map(|_| match i {
                0 => E::zero(),
                1 => E::one(),
                _ => E::from(next() & 1),
            }));
        }
        let mut lt = LtRand::new(&bits);
        for (round, width) in LtRand::preparation_widths().into_iter().enumerate() {
            let (x, y) = lt.preparation_operands(round);
            assert_eq!(x.len(), width * zs.len());
            let products: Vec<E> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
            assert_eq!(lt.preparation_complete(round, &products), round == 3);
        }
        let z: Vec<E> = zs.iter().map(|&z| E::new(velox::fields::mersenne_127::M127(z))).collect();
        let first = lt.mask_inputs(&z);
        let second = lt.look_up_signs(&first);
        for (noise, z) in lt.noise(&second).iter().zip(&zs) {
            assert_eq!(F::to_canonical_u128(noise), z % (1 << 63), "z={z}");
        }
    }
}
