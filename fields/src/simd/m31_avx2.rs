//! AVX2 lanes over the Mersenne-31 base field and its Fp2/Fp4/Fp8 tower:
//! eight independent elements per 256-bit register.
//!
//! Representation: every lane holds a value in `[0, p]` — the same weak range
//! lambdaworks' scalar `Mersenne31Field` uses (`p` itself is a legal encoding
//! of zero). Every op accepts and returns that range.
//!
//! A 31x31-bit product fits in 64 bits, so unlike M61 the multiply needs no
//! limb splitting: `vpmuludq` on the even lanes and on the odd lanes (shifted
//! down) gives eight full products, and `2^31 ≡ 1` folds each with one
//! shift-and-add.

use core::arch::x86_64::*;

pub const P: u32 = (1 << 31) - 1;

#[derive(Clone, Copy)]
pub struct M31x8(pub __m256i);

impl M31x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self(_mm256_setzero_si256())
    }

    #[inline(always)]
    pub unsafe fn splat(x: u32) -> Self {
        Self(_mm256_set1_epi32(x as i32))
    }

    #[inline(always)]
    pub unsafe fn load(p: *const u32) -> Self {
        Self(_mm256_loadu_si256(p as *const __m256i))
    }

    #[inline(always)]
    pub unsafe fn store(self, p: *mut u32) {
        _mm256_storeu_si256(p as *mut __m256i, self.0)
    }

    #[inline(always)]
    unsafe fn p_vec() -> __m256i {
        _mm256_set1_epi32(P as i32)
    }

    /// `x <= 2^32 - 2` -> `(x >> 31) + (x & p)`, which is `<= p`.
    #[inline(always)]
    unsafe fn weak_reduce(x: __m256i) -> __m256i {
        _mm256_add_epi32(_mm256_srli_epi32::<31>(x), _mm256_and_si256(x, Self::p_vec()))
    }

    /// `a + b <= 2p = 2^32 - 2`: no wrap.
    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self(Self::weak_reduce(_mm256_add_epi32(self.0, r.0)))
    }

    /// `a + p - b` is in `[0, 2p]`.
    #[inline(always)]
    pub unsafe fn sub(self, r: Self) -> Self {
        Self(Self::weak_reduce(_mm256_sub_epi32(
            _mm256_add_epi32(self.0, Self::p_vec()),
            r.0,
        )))
    }

    #[inline(always)]
    pub unsafe fn double(self) -> Self {
        self.add(self)
    }

    #[inline(always)]
    pub unsafe fn neg(self) -> Self {
        Self(_mm256_sub_epi32(Self::p_vec(), self.0))
    }

    /// Fold a 64-bit product `< 2^62` (in each 64-bit lane) to `< 2^32`.
    #[inline(always)]
    unsafe fn fold64(prod: __m256i) -> __m256i {
        let p64 = _mm256_set1_epi64x(P as i64);
        _mm256_add_epi64(_mm256_srli_epi64::<31>(prod), _mm256_and_si256(prod, p64))
    }

    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a = self.0;
        let b = r.0;
        // Even 32-bit lanes: vpmuludq reads the low 32 bits of each 64-bit lane.
        let even = Self::fold64(_mm256_mul_epu32(a, b));
        // Odd lanes: move them down first.
        let odd = Self::fold64(_mm256_mul_epu32(
            _mm256_srli_epi64::<32>(a),
            _mm256_srli_epi64::<32>(b),
        ));
        // Each fold is < 2^32 and sits in the low half of its 64-bit lane;
        // re-interleave into 8 x u32 and finish with one 32-bit reduction.
        let merged = _mm256_blend_epi32::<0b1010_1010>(even, _mm256_slli_epi64::<32>(odd));
        Self(Self::weak_reduce(merged))
    }
}

/// Eight Fp2 elements, `c0 + c1*i`, `i^2 = -1` (lambdaworks `Degree2ExtensionField`).
#[derive(Clone, Copy)]
pub struct Fp2x8 {
    pub c0: M31x8,
    pub c1: M31x8,
}

impl Fp2x8 {
    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self { c0: M31x8::zero(), c1: M31x8::zero() }
    }
    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self { c0: self.c0.add(r.c0), c1: self.c1.add(r.c1) }
    }
    #[inline(always)]
    pub unsafe fn sub(self, r: Self) -> Self {
        Self { c0: self.c0.sub(r.c0), c1: self.c1.sub(r.c1) }
    }
    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a0b0 = self.c0.mul(r.c0);
        let a1b1 = self.c1.mul(r.c1);
        let z = self.c0.add(self.c1).mul(r.c0.add(r.c1));
        Self { c0: a0b0.sub(a1b1), c1: z.sub(a0b0).sub(a1b1) }
    }
    /// Times the Fp4 non-residue `2 + i`: `(2a0 - a1) + (2a1 + a0) i`.
    #[inline(always)]
    pub unsafe fn mul_nonresidue(self) -> Self {
        Self {
            c0: self.c0.double().sub(self.c1),
            c1: self.c1.double().add(self.c0),
        }
    }
    /// Times `1 - 2i`: `(a0 + 2a1) + (a1 - 2a0) i`.
    #[inline(always)]
    pub unsafe fn mul_one_minus_2i(self) -> Self {
        Self {
            c0: self.c0.add(self.c1.double()),
            c1: self.c1.sub(self.c0.double()),
        }
    }
    /// Times `i`: `-a1 + a0 i`.
    #[inline(always)]
    pub unsafe fn mul_i(self) -> Self {
        Self { c0: self.c1.neg(), c1: self.c0 }
    }
}

/// Eight Fp4 elements, `c0 + c1*j`, `j^2 = 2 + i` (lambdaworks `Degree4ExtensionField`).
#[derive(Clone, Copy)]
pub struct Fp4x8 {
    pub c0: Fp2x8,
    pub c1: Fp2x8,
}

impl Fp4x8 {
    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self { c0: Fp2x8::zero(), c1: Fp2x8::zero() }
    }
    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self { c0: self.c0.add(r.c0), c1: self.c1.add(r.c1) }
    }
    #[inline(always)]
    pub unsafe fn sub(self, r: Self) -> Self {
        Self { c0: self.c0.sub(r.c0), c1: self.c1.sub(r.c1) }
    }
    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a0b0 = self.c0.mul(r.c0);
        let a1b1 = self.c1.mul(r.c1);
        let z = self.c0.add(self.c1).mul(r.c0.add(r.c1));
        Self { c0: a0b0.add(a1b1.mul_nonresidue()), c1: z.sub(a0b0).sub(a1b1) }
    }
    /// Times the Fp8 non-residue `n = (1 - 2i) + i·j` (scalable_mpc's
    /// `mul_fp4_by_nonresidue`, specialised): with `a = a0 + a1 j`,
    /// `a·n = (a0(1-2i) + (2+i)·(a1·i)) + (a0·i + a1(1-2i)) j`.
    #[inline(always)]
    pub unsafe fn mul_nonresidue(self) -> Self {
        Self {
            c0: self.c0.mul_one_minus_2i().add(self.c1.mul_i().mul_nonresidue()),
            c1: self.c0.mul_i().add(self.c1.mul_one_minus_2i()),
        }
    }
}

/// Eight Fp8 elements, `c0 + c1*k`, `k^2 = 1 - 2i + ji` (scalable_mpc `Degree8ExtensionField`).
#[derive(Clone, Copy)]
pub struct Fp8x8 {
    pub c0: Fp4x8,
    pub c1: Fp4x8,
}

impl Fp8x8 {
    #[inline(always)]
    pub unsafe fn zero() -> Self {
        Self { c0: Fp4x8::zero(), c1: Fp4x8::zero() }
    }
    #[inline(always)]
    pub unsafe fn add(self, r: Self) -> Self {
        Self { c0: self.c0.add(r.c0), c1: self.c1.add(r.c1) }
    }
    #[inline(always)]
    pub unsafe fn mul(self, r: Self) -> Self {
        let a0b0 = self.c0.mul(r.c0);
        let a1b1 = self.c1.mul(r.c1);
        let z = self.c0.add(self.c1).mul(r.c0.add(r.c1));
        Self { c0: a0b0.add(a1b1.mul_nonresidue()), c1: z.sub(a0b0).sub(a1b1) }
    }
}
