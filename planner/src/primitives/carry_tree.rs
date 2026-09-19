//! `CarryOutL` (Catrina–de Hoogh, Table 3) specialised to a public first
//! operand: the schedule of a bitwise less-than, and the local arithmetic
//! around each multiplication level.
//!
//! `[a < b] = 1 − CarryOut(a, ¬b, c_in = 1)` — the carry out of the top bit
//! of `2^k + a − b`. The carry is a fold of the associative operator
//!
//! ```text
//! (P_L, G_L) ∘ (P_H, G_H) = (P_L · P_H, G_H + P_H · G_L)
//! ```
//!
//! over the leaves `p_i = a_i ⊕ ¬b_i`, `g_i = a_i ∧ ¬b_i`, as a balanced
//! binary tree: `⌈log₂ k⌉` levels, each one multiplication round. Nothing is
//! opened — that is the whole reason this replaces Liu et al.'s prefix-OR,
//! whose `ΠMultPub` opens bits.
//!
//! Two specialisations because `a` is public:
//!
//! - A pair of leaves costs one multiplication. `g_i = a_i (1 − p_i)`, so
//!   `P = p_L p_H` is the product and `G = a_H (1 − p_H) + a_L (p_H − P)` is
//!   local.
//! - The carry-in is folded into the lowest leaf: `(0, 1) ∘ (p_0, g_0) =
//!   (0, g_0 + p_0)`, local. Every node containing bit 0 — the left spine,
//!   root included — then has `P = 0` for free and costs one multiplication,
//!   `P_H · G_L`.
//!
//! At `k = 61`: levels of 30, 29, 15, 7, 3, 1 multiplications, 85 in all.
//! At `k = 31`: 15, 15, 7, 3, 1 — 41.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// How the node is formed from the two slots below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineType {
    /// Two affine leaves: one multiplication, `P` is the product.
    LeafPair,
    /// The lower slot has `P = 0`: one multiplication, `G = g_H + p_H · G_L`.
    Spine,
    /// Two multiplications.
    Inner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Combine { lo: usize, hi: usize, combine: CombineType },
    /// An odd slot carried up unchanged.
    Carry(usize),
}

/// A `(P, G)` pair. `leaf_a` is set on an affine leaf — one whose `G` is
/// `a (1 − P)` for a public bit `a` — which is what lets a leaf pair skip
/// its second multiplication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot<F: IsField> {
    pub p: FieldElement<F>,
    pub g: FieldElement<F>,
    pub leaf_a: Option<u64>,
}

/// The schedule for `k`-bit operands: `levels[l]` forms level `l + 1` from
/// level `l`'s slots. Level 0 holds the `k` leaves.
#[derive(Clone, Debug)]
pub struct CarryTree {
    k: usize,
    levels: Vec<Vec<Node>>,
}

impl CarryTree {
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "a carry tree needs at least one bit");
        let mut levels = Vec::new();
        let mut count = k;
        let mut level = 0;
        while count > 1 {
            let mut nodes = Vec::with_capacity(count.div_ceil(2));
            for i in 0..count / 2 {
                let combine = if i == 0 {
                    CombineType::Spine
                } else if level == 0 {
                    CombineType::LeafPair
                } else {
                    CombineType::Inner
                };
                nodes.push(Node::Combine { lo: 2 * i, hi: 2 * i + 1, combine });
            }
            if count % 2 == 1 {
                nodes.push(Node::Carry(count - 1));
            }
            levels.push(nodes);
            count = count.div_ceil(2);
            level += 1;
        }
        Self { k, levels }
    }

    pub fn bits(&self) -> usize {
        self.k
    }

    /// Multiplication rounds.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Multiplications at each level.
    pub fn level_widths(&self) -> Vec<usize> {
        self.levels
            .iter()
            .map(|nodes| {
                nodes
                    .iter()
                    .map(|node| match node {
                        Node::Combine { combine: CombineType::Inner, .. } => 2,
                        Node::Combine { .. } => 1,
                        Node::Carry(_) => 0,
                    })
                    .sum()
            })
            .collect()
    }

    /// The leaves from the public bits of `a` and the sharings of `¬b`
    /// (`not_b[i] = 1 − [b_i]`), LSB first, with the carry-in folded into
    /// leaf 0.
    pub fn leaves<F: IsField>(&self, a_bits: &[u64], not_b: &[FieldElement<F>]) -> Vec<Slot<F>> {
        assert_eq!(a_bits.len(), self.k);
        assert_eq!(not_b.len(), self.k);
        let one = FieldElement::<F>::one();
        let mut slots: Vec<Slot<F>> = (0..self.k)
            .map(|i| {
                // p = a ⊕ ¬b; g = a ∧ ¬b = a (1 − p).
                let p = if a_bits[i] == 1 { &one - &not_b[i] } else { not_b[i].clone() };
                let g = if a_bits[i] == 1 { &one - &p } else { FieldElement::<F>::zero() };
                Slot { p, g, leaf_a: Some(a_bits[i]) }
            })
            .collect();
        // Carry-in: (0, 1) ∘ (p_0, g_0) = (0, g_0 + p_0).
        let leaf0 = &slots[0];
        slots[0] = Slot { p: FieldElement::<F>::zero(), g: &leaf0.g + &leaf0.p, leaf_a: None };
        slots
    }

    /// The multiplications of `level`, in the order `level_absorb` expects
    /// their products.
    pub fn level_operands<F: IsField>(&self, level: usize, slots: &[Slot<F>]) -> Vec<(FieldElement<F>, FieldElement<F>)> {
        let mut operands = Vec::with_capacity(self.level_widths()[level]);
        for node in &self.levels[level] {
            if let Node::Combine { lo, hi, combine } = node {
                let (lo, hi) = (&slots[*lo], &slots[*hi]);
                match combine {
                    CombineType::LeafPair => operands.push((lo.p.clone(), hi.p.clone())),
                    CombineType::Spine => operands.push((hi.p.clone(), lo.g.clone())),
                    CombineType::Inner => {
                        operands.push((lo.p.clone(), hi.p.clone()));
                        operands.push((hi.p.clone(), lo.g.clone()));
                    }
                }
            }
        }
        operands
    }

    /// Form level `level + 1` from the products of `level_operands`.
    pub fn level_absorb<F: IsField>(&self, level: usize, slots: &[Slot<F>], products: &[FieldElement<F>]) -> Vec<Slot<F>> {
        let one = FieldElement::<F>::one();
        let mut products = products.iter();
        let mut next = Vec::with_capacity(self.levels[level].len());
        for node in &self.levels[level] {
            match node {
                Node::Carry(from) => next.push(slots[*from].clone()),
                Node::Combine { lo, hi, combine } => {
                    let (lo, hi) = (&slots[*lo], &slots[*hi]);
                    let slot = match combine {
                        CombineType::LeafPair => {
                            let p = products.next().expect("a product per leaf pair").clone();
                            let (a_lo, a_hi) = (lo.leaf_a.expect("affine leaf"), hi.leaf_a.expect("affine leaf"));
                            let bit = |a: u64, e: FieldElement<F>| if a == 1 { e } else { FieldElement::<F>::zero() };
                            // G = a_H (1 − p_H) + a_L (p_H − P)
                            let g = bit(a_hi, &one - &hi.p) + bit(a_lo, &hi.p - &p);
                            Slot { p, g, leaf_a: None }
                        }
                        CombineType::Spine => {
                            let prod = products.next().expect("a product per spine node");
                            Slot { p: FieldElement::<F>::zero(), g: &hi.g + prod, leaf_a: None }
                        }
                        CombineType::Inner => {
                            let p = products.next().expect("two products per inner node").clone();
                            let prod = products.next().expect("two products per inner node");
                            Slot { p, g: &hi.g + prod, leaf_a: None }
                        }
                    };
                    next.push(slot);
                }
            }
        }
        assert!(products.next().is_none(), "more products than the level consumes");
        next
    }

    /// The carry out, once one slot is left. The root contains bit 0, so its
    /// `P` is 0 and the carry is its `G`; the carry-in was folded in.
    pub fn carry<F: IsField>(&self, slots: &[Slot<F>]) -> FieldElement<F> {
        assert_eq!(slots.len(), 1, "the tree is not folded down to one slot");
        slots[0].g.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::Mersenne61Field;

    type F = Mersenne61Field;
    type E = FieldElement<F>;

    /// The whole tree on plaintext bits, multiplication being the product.
    fn less_than(k: usize, a: u64, b: u64) -> u64 {
        let tree = CarryTree::new(k);
        let a_bits: Vec<u64> = (0..k).map(|i| (a >> i) & 1).collect();
        let not_b: Vec<E> = (0..k).map(|i| E::from(1 - ((b >> i) & 1))).collect();
        let mut slots = tree.leaves(&a_bits, &not_b);
        for level in 0..tree.num_levels() {
            let operands = tree.level_operands(level, &slots);
            assert_eq!(operands.len(), tree.level_widths()[level], "k={k} level {level}");
            let products: Vec<E> = operands.iter().map(|(x, y)| x * y).collect();
            slots = tree.level_absorb(level, &slots, &products);
        }
        let carry = tree.carry(&slots);
        let lt = E::one() - carry;
        assert!(lt == E::zero() || lt == E::one(), "not a bit");
        if lt == E::one() { 1 } else { 0 }
    }

    #[test]
    fn exhaustive_up_to_eight_bits() {
        for k in 1..=8 {
            for a in 0..(1u64 << k) {
                for b in 0..(1u64 << k) {
                    assert_eq!(less_than(k, a, b), (a < b) as u64, "k={k} a={a} b={b}");
                }
            }
        }
    }

    #[test]
    fn randomised_at_31_and_61_bits() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for k in [31usize, 61] {
            let mask = (1u64 << k) - 1;
            for _ in 0..300 {
                let (a, b) = (next() & mask, next() & mask);
                assert_eq!(less_than(k, a, b), (a < b) as u64, "k={k} a={a} b={b}");
            }
            for (a, b) in [(0, 0), (mask, mask), (0, mask), (mask, 0), (mask - 1, mask), (1, 0)] {
                assert_eq!(less_than(k, a, b), (a < b) as u64, "k={k} a={a} b={b}");
            }
        }
    }

    #[test]
    fn widths_are_as_planned() {
        assert_eq!(CarryTree::new(61).level_widths(), vec![30, 29, 15, 7, 3, 1]);
        assert_eq!(CarryTree::new(61).level_widths().iter().sum::<usize>(), 85);
        assert_eq!(CarryTree::new(31).level_widths(), vec![15, 15, 7, 3, 1]);
        assert_eq!(CarryTree::new(1).num_levels(), 0);
        assert_eq!(CarryTree::new(2).level_widths(), vec![1]);
        assert_eq!(CarryTree::new(8).level_widths(), vec![4, 3, 1]);
    }
}
