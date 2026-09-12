//! AVX2 lanes over the Mersenne-61 base field and its Fp2/Fp4 tower: four
//! independent elements per 256-bit register.
//!
//! The long-form walkthrough, with the derivations and the measurements, is
//! `docs/simd-m61-avx2.md`. This header states the invariants the code relies
//! on; each operation's doc comment states its own bounds.
//!
//! # Representation
//!
//! Each 64-bit lane of an [`M61x4`] holds one element. Lanes never interact:
//! in the GEMM they are four consecutive output columns.
//!
//! Values are kept **weakly reduced**, not canonical. With `p = 2^61 − 1` and
//! `2^61 ≡ 1 (mod p)`, any 64-bit `x` satisfies
//!
//! ```text
//! x = (x >> 61)·2^61 + (x & p)  ≡  (x >> 61) + (x & p)     (mod p)
//! ```
//!
//! and that three-instruction fold (`weak_reduce`) yields a value `≤ p + 7`.
//! The invariant is therefore:
//!
//! > every lane produced by this module is in `[0, 2^61 + 8)` — the *weak
//! > range* — and every operation accepts inputs in that range.
//!
//! Skipping canonicalisation avoids a compare-and-select after each of the
//! ~30 base operations in an Fp4 multiply; the cost is that each operation
//! must prove its intermediate sums never wrap 64 bits from weak-range inputs.
//! Those proofs are the bounds quoted on each method. Scalar elements may
//! carry the raw non-canonical value `p` (e.g. the result of `a − a`); it is
//! inside the weak range and needs no special handling. Canonicalisation
//! happens once, when the GEMM unpacks its output through `FieldElement::new`.
//!
//! # Multiplication without a 64×64 multiply
//!
//! AVX2's only wide integer multiply is `vpmuludq` (`_mm256_mul_epu32`): in
//! each 64-bit lane it multiplies the *low 32 bits* of both operands into a
//! 64-bit product. A 61-bit product is assembled from 32-bit halves,
//!
//! ```text
//! a = a_hi·2^32 + a_lo,   b = b_hi·2^32 + b_lo
//! a·b = hh·2^64 + mid·2^32 + ll,      hh = a_hi·b_hi,  mid = a_hi·b_lo + a_lo·b_hi,  ll = a_lo·b_lo
//! ```
//!
//! and each term folded with `2^61 ≡ 1`, `2^64 ≡ 8`:
//!
//! ```text
//! hh·2^64  ≡ 8·hh
//! mid·2^32 ≡ (mid >> 29) + ((mid mod 2^29) << 32)      because 29 + 32 = 61
//! ll       ≡ (ll >> 61) + (ll & p)
//! ```
//!
//! The five pieces sum to `< 2^63 + 2^62 + 2^34 + 8 < 2^64`, so one final
//! `weak_reduce` gives the product. 21 instructions for four products; see
//! [`M61x4::mul`].
//!
//! # Tower
//!
//! [`Fp2x4`] (`i² = −1`) and [`Fp4x4`] (`w² = 4 + i`) are the formulas of
//! `mersenne_61::extensions` applied lane-wise, with the `z − x − y` and
//! `4x ∓ y` patterns fused into single-reduction operations
//! ([`M61x4::sub2`], [`M61x4::mul4_sub`], [`M61x4::add_mul4`]).

use core::arch::x86_64::*;

use crate::mersenne_61::MERSENNE_61_PRIME_FIELD_ORDER as P;

#[derive(Clone, Copy)]
pub struct M61x4(pub __m256i);

impl M61x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self(_mm256_setzero_si256())
    }

    #[inline(always)]
    pub unsafe fn splat(x: u64) -> Self {
        Self(_mm256_set1_epi64x(x as i64))
    }

    #[inline(always)]
    pub unsafe fn load(p: *const u64) -> Self {
        Self(_mm256_loadu_si256(p as *const __m256i))
    }

    #[inline(always)]
    pub unsafe fn store(self, p: *mut u64) {
        _mm256_storeu_si256(p as *mut __m256i, self.0)
    }

    #[inline(always)]
    unsafe fn p_vec() -> __m256i {
        _mm256_set1_epi64x(P as i64)
    }

    /// `x ≡ (x >> 61) + (x & p)` for any 64-bit `x`; the result is `≤ p + 7`,
    /// i.e. inside the weak range `[0, 2^61 + 8)`. Three instructions, no
    /// compare.
    #[inline(always)]
    unsafe fn weak_reduce(x: __m256i) -> __m256i {
        _mm256_add_epi64(_mm256_srli_epi64::<61>(x), _mm256_and_si256(x, Self::p_vec()))
    }

    /// `a + b`. Precondition `a, b < 2^63` so the sum cannot wrap; weak-range
    /// inputs satisfy `a + b < 2^62 + 16`. 4 instructions.
    #[inline(always)]
    pub unsafe fn add(self, rhs: Self) -> Self {
        Self(Self::weak_reduce(_mm256_add_epi64(self.0, rhs.0)))
    }

    /// `a − b`, computed as `a + 2p − b` so no compare is needed.
    /// Preconditions `a < 2^62` (so `a + 2p < 2^63`) and `b ≤ 2p` (so the
    /// difference is non-negative); weak-range inputs satisfy both.
    /// 5 instructions.
    #[inline(always)]
    pub unsafe fn sub(self, rhs: Self) -> Self {
        let two_p = _mm256_set1_epi64x((2 * P) as i64);
        Self(Self::weak_reduce(_mm256_sub_epi64(
            _mm256_add_epi64(self.0, two_p),
            rhs.0,
        )))
    }

    /// `self − a − b` with a single reduction, computed as `self + 4p − a − b`.
    /// Preconditions `self < 2^62` (so `self + 4p < 2^64`) and `a + b ≤ 4p`;
    /// weak-range inputs give `a + b < 2^62 + 16 < 4p`. 6 instructions, versus
    /// 10 for two `sub`s. Used for the Karatsuba `z − a0b0 − a1b1` terms.
    #[inline(always)]
    pub unsafe fn sub2(self, a: Self, b: Self) -> Self {
        let four_p = _mm256_set1_epi64x((4 * P) as i64);
        let t = _mm256_sub_epi64(_mm256_sub_epi64(_mm256_add_epi64(self.0, four_p), a.0), b.0);
        Self(Self::weak_reduce(t))
    }

    /// `4·self − r` with a single reduction, computed as `(self << 2) + 2p − r`.
    /// Preconditions `self < 3·2^60` (so `4·self + 2p < 2^64`) and `r ≤ 2p`;
    /// weak-range inputs give `4·self ≤ 2^63 + 24`. 6 instructions. Used for
    /// the real part of `(4 + i)·x`.
    #[inline(always)]
    pub unsafe fn mul4_sub(self, r: Self) -> Self {
        let two_p = _mm256_set1_epi64x((2 * P) as i64);
        let t = _mm256_sub_epi64(_mm256_add_epi64(_mm256_slli_epi64::<2>(self.0), two_p), r.0);
        Self(Self::weak_reduce(t))
    }

    /// `self + 4·r` with a single reduction. Precondition `self + 4·r < 2^64`;
    /// weak-range inputs give `< 2^61 + 8 + 2^63 + 32`. 5 instructions. Used
    /// for the imaginary part of `(4 + i)·x`.
    #[inline(always)]
    pub unsafe fn add_mul4(self, r: Self) -> Self {
        Self(Self::weak_reduce(_mm256_add_epi64(self.0, _mm256_slli_epi64::<2>(r.0))))
    }

    /// `a · b mod p`, weakly reduced. Precondition `a, b < 2^62` (so that
    /// `a_hi, b_hi < 2^30` and the bounds below hold); weak-range inputs have
    /// a factor of two to spare. 21 instructions for four products.
    ///
    /// See the module docs for the derivation; the comments give the bound
    /// on each intermediate for `a, b < 2^62`.
    #[inline(always)]
    pub unsafe fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;

        // Step 1: 32-bit halves and the four partial products.
        // vpmuludq multiplies the low 32 bits of each lane, so `a`/`b` serve
        // as `a_lo`/`b_lo` unmasked, and the shifted copies as `a_hi`/`b_hi`.
        let a_hi = _mm256_srli_epi64::<32>(a); // < 2^30
        let b_hi = _mm256_srli_epi64::<32>(b); // < 2^30
        let ll = _mm256_mul_epu32(a, b); // a_lo·b_lo < 2^64
        let lh = _mm256_mul_epu32(a, b_hi); // a_lo·b_hi < 2^62
        let hl = _mm256_mul_epu32(a_hi, b); // a_hi·b_lo < 2^62
        let hh = _mm256_mul_epu32(a_hi, b_hi); // a_hi·b_hi < 2^60
        let mid = _mm256_add_epi64(lh, hl); // < 2^63, no wrap
        // Now a·b = hh·2^64 + mid·2^32 + ll.

        // Step 2: fold each term with 2^61 ≡ 1.
        //   hh·2^64 = hh·8·2^61 ≡ 8·hh
        let hh8 = _mm256_slli_epi64::<3>(hh); // < 2^63
        //   mid·2^32 = (mid >> 29)·2^61 + (mid mod 2^29)·2^32
        //            ≡ (mid >> 29) + (mid mod 2^29) << 32
        let mid_hi = _mm256_srli_epi64::<29>(mid); // < 2^34
        let mask29 = _mm256_set1_epi64x((1i64 << 29) - 1);
        let mid_lo = _mm256_slli_epi64::<32>(_mm256_and_si256(mid, mask29)); // < 2^61
        //   ll ≡ (ll >> 61) + (ll & p)
        let ll_hi = _mm256_srli_epi64::<61>(ll); // < 8
        let ll_lo = _mm256_and_si256(ll, Self::p_vec()); // < 2^61

        // Step 3: sum the five pieces and weak-reduce once.
        // Bound: 2^63 + 2^34 + 2^61 + 8 + 2^61 = 2^63 + 2^62 + 2^34 + 8 < 2^64.
        let s = _mm256_add_epi64(
            _mm256_add_epi64(hh8, mid_hi),
            _mm256_add_epi64(_mm256_add_epi64(mid_lo, ll_hi), ll_lo),
        );
        Self(Self::weak_reduce(s))
    }
}

/// Four Fp2 elements, `c0 + c1·i` with `i² = −1`: `Mersenne61Degree2ExtensionField`
/// applied lane-wise. Every method is the scalar formula with lane operations
/// substituted; all results stay in the weak range because every base
/// operation does.
#[derive(Clone, Copy)]
pub struct Fp2x4 {
    pub c0: M61x4,
    pub c1: M61x4,
}

impl Fp2x4 {
    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self { c0: M61x4::zero(), c1: M61x4::zero() }
    }

    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self { c0: self.c0.add(r.c0), c1: self.c1.add(r.c1) }
    }

    #[inline(always)]
    pub unsafe fn sub(self, r: Self) -> Self {
        Self { c0: self.c0.sub(r.c0), c1: self.c1.sub(r.c1) }
    }

    /// Karatsuba with three base multiplies:
    /// `c0 = a0b0 − a1b1`, `c1 = (a0 + a1)(b0 + b1) − a0b0 − a1b1`.
    /// The `add`s feeding the third multiply keep it within its `< 2^62` bound.
    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a0b0 = self.c0.mul(r.c0);
        let a1b1 = self.c1.mul(r.c1);
        let z = self.c0.add(self.c1).mul(r.c0.add(r.c1));
        Self { c0: a0b0.sub(a1b1), c1: z.sub2(a0b0, a1b1) }
    }

    /// `self - a - b`, one reduction per limb.
    #[inline(always)]
    pub unsafe fn sub2(self, a: Self, b: Self) -> Self {
        Self { c0: self.c0.sub2(a.c0, b.c0), c1: self.c1.sub2(a.c1, b.c1) }
    }

    /// Multiply by the Fp4 non-residue `4 + i`:
    /// `(a0 + a1·i)(4 + i) = (4a0 − a1) + (a0 + 4a1)·i`, one reduction per part.
    #[inline(always)]
    pub unsafe fn mul_nonresidue(self) -> Self {
        Self {
            c0: self.c0.mul4_sub(self.c1),
            c1: self.c0.add_mul4(self.c1),
        }
    }
}

/// Four Fp4 elements, `c0 + c1·w` with `w² = 4 + i`:
/// `Mersenne61Degree4ExtensionField` applied lane-wise.
#[derive(Clone, Copy)]
pub struct Fp4x4 {
    pub c0: Fp2x4,
    pub c1: Fp2x4,
}

impl Fp4x4 {
    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self { c0: Fp2x4::zero(), c1: Fp2x4::zero() }
    }

    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self { c0: self.c0.add(r.c0), c1: self.c1.add(r.c1) }
    }

    /// `(a0 + a1·w)(b0 + b1·w) = (a0b0 + (4 + i)·a1b1) + ((a0 + a1)(b0 + b1) − a0b0 − a1b1)·w`
    ///
    /// Three Fp2 multiplies = 9 base multiplies; ≈ 293 instructions for four
    /// products, ≈ 77 per multiply-add once the accumulate is included.
    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a0b0 = self.c0.mul(r.c0);
        let a1b1 = self.c1.mul(r.c1);
        let z = self.c0.add(self.c1).mul(r.c0.add(r.c1));
        Self {
            c0: a0b0.add(a1b1.mul_nonresidue()),
            c1: z.sub2(a0b0, a1b1),
        }
    }
}
