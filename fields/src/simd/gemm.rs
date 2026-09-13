//! The lane-generic GEMM kernel and its field ↔ limb packing.
//!
//! `out[r][i] = Σ_l matrix[r][l] · vectors[i][l]` is vectorised across `i`
//! (the K dimension — the large one at every call site): one register holds
//! the same limb of `LANES` consecutive output columns, and the scalar tower
//! formulas are applied lane-wise. Every field with a [`SimdField`] impl runs
//! through the same `row_block` kernel; only the [`Elem`] type (limb width,
//! limb count, lane count, multiply) changes.
//!
//! # Layouts
//!
//! * `vt` — the right operand, limb-major and transposed:
//!   `vt[(j·C + l)·kp + i]` is limb `j` of `vectors[i][l]`, with `kp` = K
//!   rounded up to a multiple of `LANES` (zero padded). A `LANES`-wide load
//!   at `(j, l, i..i+LANES)` is then contiguous.
//! * `mat` — the left operand, one flat array per row, element-major:
//!   `mat[r][l·LIMBS + j]`. Each element is broadcast (`splat`) into a
//!   register once per (row, lane group).
//! * `out` — per row, limb-major: `out[r][j·kp + i]`; unpacked and
//!   canonicalised into `FieldElement`s at the end.
//!
//! # Cost model
//!
//! The arithmetic is R·C·K lane-wise multiply-adds. On top of that the pack
//! and unpack steps cost a fixed amount per *element*: K·C for `vectors`, R·C
//! for `matrix`, R·K for the output. Per multiply-add that is
//! `a/R + b/K + c/C`, so the overhead is only noticeable when both R and C
//! are small (the 6×6 interpolation shape) — see `docs/simd-m61-avx2.md`.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use rayon::prelude::*;

use super::m31_avx2::{Fp2x8, Fp4x8, Fp8x8, M31x8};
use super::m61_avx2::{Fp2x4, Fp4x4, M61x4};
use crate::mersenne_31::{Degree4ExtensionField as M31Fp4, Degree8ExtensionField as M31Fp8, Mersenne31Field};
use crate::mersenne_61::{Mersenne61Degree4ExtensionField, Mersenne61Field};

/// Elements of one row handled per rayon task (a multiple of every lane count).
const BLOCK: usize = 256;
const LIMBS_MAX: usize = 8;

/// A machine word a field element is made of.
pub trait Limb: Copy + Default + Send + Sync + 'static {
    fn random() -> Self;
}
impl Limb for u64 {
    fn random() -> Self { rand::random() }
}
impl Limb for u32 {
    fn random() -> Self { rand::random() }
}

/// A pack of `LANES` field elements held in vector registers, one register
/// per base-field coefficient ("limb").
///
/// This is the only thing the GEMM kernel (`row_block`) knows about a
/// field. It never sees a `FieldElement`; it moves limbs between memory and
/// registers with [`load`](Elem::load) / [`splat`](Elem::splat) /
/// [`store`](Elem::store), and accumulates with
/// [`mul_add`](Elem::mul_add). Everything field-specific — how many limbs an
/// element has, how wide they are, how many fit in a register, and what
/// multiplication means — is carried by the implementing type, so one
/// monomorphised kernel serves every field.
///
/// Implementors: [`M61x4`], [`Fp4x4`] (Mersenne-61), [`M31x8`], [`Fp4x8`],
/// [`Fp8x8`] (Mersenne-31).
///
/// # Safety
///
/// Every method is `unsafe` because the implementations are AVX2 intrinsics
/// (`#[inline(always)]`, without their own `#[target_feature]`), which are
/// only valid to execute inside a function compiled with AVX2 enabled and on
/// a CPU that has it. The kernel provides the former; `super::gemm_with`
/// checks the latter before calling the kernel.
pub trait Elem: Copy {
    /// The machine word one base-field coefficient occupies in memory:
    /// `u64` for Mersenne-61, `u32` for Mersenne-31.
    ///
    /// Fixes the element type of the packed buffers (`vt`, `mat`, `out`)
    /// the kernel reads and writes, and — together with the 256-bit register
    /// width — fixes [`LANES`](Elem::LANES).
    type Limb: Limb;

    /// Number of limbs per field element: the extension degree over the
    /// prime field. `1` for a base field, `4` for the Fp4 towers, `8` for
    /// Fp8.
    ///
    /// Determines the stride of the element-major `mat` buffer
    /// (`mat[l * LIMBS + j]`), the number of limb planes in `vt` and `out`,
    /// and the number of registers one pack occupies — an `Fp4x4` is four
    /// `__m256i`, an `Fp8x8` eight.
    const LIMBS: usize;

    /// Number of field elements one pack holds, i.e. how many consecutive
    /// output columns of the GEMM a single `mul_add` advances:
    /// `256 / (8 · size_of::<Limb>())` — 4 for `u64` limbs, 8 for `u32`.
    ///
    /// The kernel steps `i` by this amount, and the packed K dimension `kp`
    /// is rounded up to a multiple of it so the last load never runs past
    /// the buffer.
    const LANES: usize;

    /// A pack whose every lane is the field's zero — the accumulator's
    /// starting value for each `(row, lane group)`.
    unsafe fn zero() -> Self;

    /// Broadcast one element to all lanes: its `LIMBS` limbs are read
    /// consecutively from `limbs`, and limb `j` fills every lane of
    /// register `j`.
    ///
    /// Used for the left operand: `matrix[r][l]` is the same for all output
    /// columns in the group, so one broadcast (`vpbroadcastq`/`d` per limb)
    /// pairs it with `LANES` different right-operand elements.
    unsafe fn splat(limbs: *const Self::Limb) -> Self;

    /// Load `LANES` *different* elements, one per lane, from the limb-major
    /// `vt` buffer: limb `j` of all `LANES` elements is the contiguous run at
    /// `ptr.add(j * stride)`.
    ///
    /// `stride` is the distance between limb planes (`C · kp` in the
    /// kernel). Consecutive elements sharing a limb plane is what makes each
    /// limb a single unaligned 256-bit load.
    unsafe fn load(ptr: *const Self::Limb, stride: usize) -> Self;

    /// Inverse of [`load`](Elem::load): write limb `j` of the `LANES` lanes
    /// to `ptr.add(j * stride)`. Called once per `(row, lane group)` after
    /// the `l` loop, into the row's limb-major output buffer (`stride = kp`).
    ///
    /// Values are left weakly reduced; canonicalisation happens when the
    /// output is unpacked through `SimdField::from_limbs`.
    unsafe fn store(self, ptr: *mut Self::Limb, stride: usize);

    /// `self + a · b`, lane-wise in the field — the kernel's entire inner
    /// loop body. `a` is the broadcast matrix element, `b` the loaded
    /// vector elements, `self` the running accumulator.
    ///
    /// For an extension field this is the full tower multiply (9 base
    /// multiplies for Fp4, 27 for Fp8) followed by a lane-wise add; the
    /// result is weakly reduced and can be fed straight back in as `self`.
    unsafe fn mul_add(self, a: Self, b: Self) -> Self;
}

// ---- M61 ---------------------------------------------------------------

impl Elem for M61x4 {
    type Limb = u64;
    const LIMBS: usize = 1;
    const LANES: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self { M61x4::zero() }
    #[inline(always)]
    unsafe fn splat(l: *const u64) -> Self { M61x4::splat(*l) }
    #[inline(always)]
    unsafe fn load(p: *const u64, _s: usize) -> Self { M61x4::load(p) }
    #[inline(always)]
    unsafe fn store(self, p: *mut u64, _s: usize) { M61x4::store(self, p) }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self { self.add(a.mul(b)) }
}

impl Elem for Fp4x4 {
    type Limb = u64;
    const LIMBS: usize = 4;
    const LANES: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self { Fp4x4::zero() }
    #[inline(always)]
    unsafe fn splat(l: *const u64) -> Self {
        Fp4x4 {
            c0: Fp2x4 { c0: M61x4::splat(*l), c1: M61x4::splat(*l.add(1)) },
            c1: Fp2x4 { c0: M61x4::splat(*l.add(2)), c1: M61x4::splat(*l.add(3)) },
        }
    }
    #[inline(always)]
    unsafe fn load(p: *const u64, s: usize) -> Self {
        Fp4x4 {
            c0: Fp2x4 { c0: M61x4::load(p), c1: M61x4::load(p.add(s)) },
            c1: Fp2x4 { c0: M61x4::load(p.add(2 * s)), c1: M61x4::load(p.add(3 * s)) },
        }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u64, s: usize) {
        self.c0.c0.store(p);
        self.c0.c1.store(p.add(s));
        self.c1.c0.store(p.add(2 * s));
        self.c1.c1.store(p.add(3 * s));
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self { self.add(a.mul(b)) }
}

// ---- M31 ---------------------------------------------------------------

impl Elem for M31x8 {
    type Limb = u32;
    const LIMBS: usize = 1;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self { M31x8::zero() }
    #[inline(always)]
    unsafe fn splat(l: *const u32) -> Self { M31x8::splat(*l) }
    #[inline(always)]
    unsafe fn load(p: *const u32, _s: usize) -> Self { M31x8::load(p) }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32, _s: usize) { M31x8::store(self, p) }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self { self.add(a.mul(b)) }
}

#[inline(always)]
unsafe fn fp4x8_splat(l: *const u32) -> Fp4x8 {
    Fp4x8 {
        c0: Fp2x8 { c0: M31x8::splat(*l), c1: M31x8::splat(*l.add(1)) },
        c1: Fp2x8 { c0: M31x8::splat(*l.add(2)), c1: M31x8::splat(*l.add(3)) },
    }
}
#[inline(always)]
unsafe fn fp4x8_load(p: *const u32, s: usize) -> Fp4x8 {
    Fp4x8 {
        c0: Fp2x8 { c0: M31x8::load(p), c1: M31x8::load(p.add(s)) },
        c1: Fp2x8 { c0: M31x8::load(p.add(2 * s)), c1: M31x8::load(p.add(3 * s)) },
    }
}
#[inline(always)]
unsafe fn fp4x8_store(e: Fp4x8, p: *mut u32, s: usize) {
    e.c0.c0.store(p);
    e.c0.c1.store(p.add(s));
    e.c1.c0.store(p.add(2 * s));
    e.c1.c1.store(p.add(3 * s));
}

impl Elem for Fp4x8 {
    type Limb = u32;
    const LIMBS: usize = 4;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self { Fp4x8::zero() }
    #[inline(always)]
    unsafe fn splat(l: *const u32) -> Self { fp4x8_splat(l) }
    #[inline(always)]
    unsafe fn load(p: *const u32, s: usize) -> Self { fp4x8_load(p, s) }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32, s: usize) { fp4x8_store(self, p, s) }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self { self.add(a.mul(b)) }
}

impl Elem for Fp8x8 {
    type Limb = u32;
    const LIMBS: usize = 8;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self { Fp8x8::zero() }
    #[inline(always)]
    unsafe fn splat(l: *const u32) -> Self {
        Fp8x8 { c0: fp4x8_splat(l), c1: fp4x8_splat(l.add(4)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u32, s: usize) -> Self {
        Fp8x8 { c0: fp4x8_load(p, s), c1: fp4x8_load(p.add(4 * s), s) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32, s: usize) {
        fp4x8_store(self.c0, p, s);
        fp4x8_store(self.c1, p.add(4 * s), s);
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self { self.add(a.mul(b)) }
}

// ---- Field <-> limbs -------------------------------------------------------

/// A field whose elements are `Lanes::LIMBS` limbs, in the order the lane
/// type expects.
pub trait SimdField: IsField + Sized {
    type Lanes: Elem;
    fn to_limbs(e: &FieldElement<Self>, out: &mut [<Self::Lanes as Elem>::Limb]);
    /// Canonicalises: the kernel leaves limbs weakly reduced.
    fn from_limbs(limbs: &[<Self::Lanes as Elem>::Limb]) -> FieldElement<Self>;
    /// Uniform element (limbs drawn at random, then canonicalised).
    fn rand_elem() -> FieldElement<Self> {
        let mut l = [<Self::Lanes as Elem>::Limb::default(); LIMBS_MAX];
        for x in l.iter_mut() {
            *x = <Self::Lanes as Elem>::Limb::random();
        }
        Self::from_limbs(&l[..<Self::Lanes as Elem>::LIMBS])
    }
}

impl SimdField for Mersenne61Field {
    type Lanes = M61x4;
    #[inline(always)]
    fn to_limbs(e: &FieldElement<Self>, out: &mut [u64]) { out[0] = *e.value(); }
    #[inline(always)]
    fn from_limbs(l: &[u64]) -> FieldElement<Self> { FieldElement::new(l[0]) }
}

impl SimdField for Mersenne61Degree4ExtensionField {
    type Lanes = Fp4x4;
    #[inline(always)]
    fn to_limbs(e: &FieldElement<Self>, out: &mut [u64]) {
        let v = e.value();
        out[0] = *v[0].value()[0].value();
        out[1] = *v[0].value()[1].value();
        out[2] = *v[1].value()[0].value();
        out[3] = *v[1].value()[1].value();
    }
    #[inline(always)]
    fn from_limbs(l: &[u64]) -> FieldElement<Self> {
        let c = |x: u64| FieldElement::<Mersenne61Field>::new(x);
        Mersenne61Degree4ExtensionField::const_from_fe(&[c(l[0]), c(l[1]), c(l[2]), c(l[3])])
    }
}

impl SimdField for Mersenne31Field {
    type Lanes = M31x8;
    #[inline(always)]
    fn to_limbs(e: &FieldElement<Self>, out: &mut [u32]) { out[0] = *e.value(); }
    #[inline(always)]
    fn from_limbs(l: &[u32]) -> FieldElement<Self> { FieldElement::new(l[0]) }
}

impl SimdField for M31Fp4 {
    type Lanes = Fp4x8;
    #[inline(always)]
    fn to_limbs(e: &FieldElement<Self>, out: &mut [u32]) {
        let v = e.value();
        out[0] = *v[0].value()[0].value();
        out[1] = *v[0].value()[1].value();
        out[2] = *v[1].value()[0].value();
        out[3] = *v[1].value()[1].value();
    }
    #[inline(always)]
    fn from_limbs(l: &[u32]) -> FieldElement<Self> {
        type Fp = FieldElement<Mersenne31Field>;
        type Fp2 = FieldElement<crate::mersenne_31::Degree2ExtensionField>;
        FieldElement::new([
            Fp2::new([Fp::new(l[0]), Fp::new(l[1])]),
            Fp2::new([Fp::new(l[2]), Fp::new(l[3])]),
        ])
    }
}

impl SimdField for M31Fp8 {
    type Lanes = Fp8x8;
    #[inline(always)]
    fn to_limbs(e: &FieldElement<Self>, out: &mut [u32]) {
        let v = e.value();
        M31Fp4::to_limbs(&v[0], &mut out[..4]);
        M31Fp4::to_limbs(&v[1], &mut out[4..8]);
    }
    #[inline(always)]
    fn from_limbs(l: &[u32]) -> FieldElement<Self> {
        FieldElement::new([M31Fp4::from_limbs(&l[..4]), M31Fp4::from_limbs(&l[4..8])])
    }
}

// ---- Kernel ----------------------------------------------------------------

/// One (row, block) task: `out[i] = Σ_l m_row[l] * vt[l][i]` for `i` in the block.
///
/// `m_row`: C elements, `LIMBS` limbs each (AoS).
/// `vt`: limb-major transposed vectors, `vt[(j*C + l)*kp + i]`.
/// `out`: `LIMBS * kp` limbs for this row, limb-major (`out[j*kp + i]`).
#[target_feature(enable = "avx2")]
unsafe fn row_block<E: Elem>(
    m_row: &[E::Limb],
    vt: &[E::Limb],
    c: usize,
    kp: usize,
    out: &mut [E::Limb],
    i0: usize,
    i1: usize,
) {
    let mut i = i0;
    while i < i1 {
        let mut acc = E::zero();
        for l in 0..c {
            let m = E::splat(m_row.as_ptr().add(l * E::LIMBS));
            let v = E::load(vt.as_ptr().add(l * kp + i), c * kp);
            acc = acc.mul_add(m, v);
        }
        acc.store(out.as_mut_ptr().add(i), kp);
        i += E::LANES;
    }
}

/// Scalar reference with the same loop as `poly::matrix_matrix_multiply_cpu`,
/// but needing only `IsField` so fields without a `ProtocolField` impl (the
/// M31 tower, for now) can be measured against it.
pub fn scalar_gemm<F: IsField>(
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>>
where
    FieldElement<F>: Clone + Send + Sync,
{
    let k = vectors.len();
    if k == 0 || matrix.is_empty() {
        return Vec::new();
    }
    let cols = matrix[0].len();
    if vectors.iter().any(|v| v.len() != cols) {
        log::error!(
            "scalar_gemm: matrix column count ({}) does not match vector lengths {:?}",
            cols,
            vectors.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        return Vec::new();
    }
    let results: Vec<Vec<FieldElement<F>>> = matrix
        .par_iter()
        .map(|m_row| {
            let mut row_results = vec![FieldElement::<F>::zero(); k];
            for i in 0..k {
                let m_col = &vectors[i];
                let mut sum = FieldElement::<F>::zero();
                for l in 0..m_row.len() {
                    sum += &m_row[l] * &m_col[l];
                }
                row_results[i] = sum;
            }
            row_results
        })
        .collect();
    if row_major { results } else { transpose_any(results) }
}

fn transpose_any<T: Clone + Send + Sync>(m: Vec<Vec<T>>) -> Vec<Vec<T>> {
    if m.is_empty() {
        return Vec::new();
    }
    let (rows, cols) = (m.len(), m[0].len());
    (0..cols)
        .into_par_iter()
        .map(|j| (0..rows).map(|i| m[i][j].clone()).collect())
        .collect()
}

/// The AVX2 GEMM proper: `out[r][i] = Σ_l matrix[r][l] · vectors[i][l]`.
///
/// Callers must have checked `is_x86_feature_detected!("avx2")` (see
/// [`super::gemm_with`]); the kernel is `#[target_feature(enable = "avx2")]`
/// and executes AVX2 instructions unconditionally.
///
/// # Shape
///
/// `matrix` is R×C (R = `matrix.len()`, C = `matrix[0].len()`); `vectors` is
/// K vectors of length C. The result is the R×K product `M · Vᵀ` if
/// `row_major`, else its K×R transpose — the same contract as
/// `poly::matrix_matrix_multiply_cpu`.
///
/// # Phases
///
/// The function is a pipeline of four phases. The arithmetic happens only in
/// phase 3; phases 1, 2 and 4 exist to get data into and out of the layout
/// the vector kernel needs. Their cost is linear in the number of *elements*
/// (K·C, R·C, R·K) against the R·C·K multiply-adds of phase 3, which is why
/// the overhead only shows at small R and C (see the module docs).
///
/// 1. **Pack `vectors` → `vt`** (limb-major, transposed, padded).
/// 2. **Pack `matrix` → `mat`** (flat limbs per row).
/// 3. **Kernel** — one task per (row, block of output columns), each running
///    [`row_block`] over its columns.
/// 4. **Unpack `out` → `FieldElement`s**, canonicalising, then transpose if
///    the caller wants K×R.
///
/// # Why this layout
///
/// The kernel vectorises across `i`: one register holds the same limb of
/// `LANES` consecutive output columns `i..i+LANES`, and the inner loop runs
/// over `l`. For each `(l, i)` it needs limb `j` of `vectors[i..i+LANES][l]`
/// as one contiguous 256-bit load — which the caller's `Vec<Vec<FieldElement>>`
/// (element-major, one heap allocation per vector) cannot provide. So the
/// right operand is repacked once into `vt`, and `matrix[r][l]`, which is the
/// same for all columns in a group, is simply broadcast.
///
/// `FieldElement<F>` is not `#[repr(transparent)]`, so no slice is ever
/// reinterpreted as limbs: packing goes through `SimdField::to_limbs` and
/// unpacking through `SimdField::from_limbs`.
pub(super) fn gemm_avx2<F>(
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>>
where
    F: SimdField,
    FieldElement<F>: Clone + Send + Sync,
{
    debug_assert!(is_x86_feature_detected!("avx2"));

    // ---- Shape checks --------------------------------------------------
    //
    // Same behaviour as the scalar path: an empty operand gives an empty
    // result, and a length mismatch logs and gives an empty result rather
    // than panicking mid-protocol.
    let rows = matrix.len();
    let k = vectors.len();
    if rows == 0 || k == 0 {
        return Vec::new();
    }
    let c = matrix[0].len();
    if vectors.iter().any(|v| v.len() != c) || matrix.iter().any(|r| r.len() != c) {
        log::error!(
            "gemm_avx2: matrix column count ({}) does not match vector lengths {:?}",
            c,
            vectors.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        return Vec::new();
    }

    // ---- Field geometry ------------------------------------------------
    //
    // `L`     the limb word (u64 for M61, u32 for M31)
    // `limbs` limbs per element (1 base, 4 Fp4, 8 Fp8)
    // `lanes` elements per register (4 for u64 limbs, 8 for u32)
    // `kp`    K rounded up to a multiple of `lanes`. The kernel always loads
    //         and stores whole registers, so the last group of columns must
    //         exist in the buffers even when K is not a multiple of `lanes`;
    //         the padding columns are zero and are dropped in phase 4.
    type L<F> = <<F as SimdField>::Lanes as Elem>::Limb;
    let limbs = <F::Lanes as Elem>::LIMBS;
    let lanes = <F::Lanes as Elem>::LANES;
    let kp = (k + lanes - 1) / lanes * lanes;

    // ---- Phase 1: pack the right operand ------------------------------
    //
    // Target layout (limb-major, then column of the matrix, then output
    // column):
    //
    //     vt[(j·C + l)·kp + i]  =  limb j of vectors[i][l]
    //
    // So for fixed (j, l) the K values run contiguously over i, and the
    // kernel's load of limb j for columns i..i+lanes is one vector load at
    // `vt + (j·C + l)·kp + i`. The plane for limb j starts at `j·C·kp`,
    // which is the `stride` the kernel passes to `Elem::load`.
    //
    // Work is split over blocks of BLOCK consecutive `i` (the same blocking
    // as phase 3, so the buffer is written by many cores when K is large).
    // Each task reads every element in its columns exactly once
    // (`to_limbs`) and scatters the limbs into the `limbs·C` planes. Tasks
    // write disjoint `i` ranges of every plane, so they cannot overlap;
    // Rust cannot see that through a shared `&mut [L]`, hence the raw
    // pointer handed to each task via `SyncPtr`.
    let mut vt = vec![L::<F>::default(); limbs * c * kp];
    {
        let ptr = SyncPtr(vt.as_mut_ptr());
        let len = vt.len();
        vectors.par_chunks(BLOCK).enumerate().for_each(|(b, chunk)| {
            let p = ptr;
            // SAFETY: every task writes only indices with `i` in its own
            // block; blocks are disjoint, and `vt` outlives the parallel
            // iterator.
            let vt = unsafe { std::slice::from_raw_parts_mut(p.0, len) };
            let mut tmp = [L::<F>::default(); LIMBS_MAX];
            for (di, vec) in chunk.iter().enumerate() {
                let i = b * BLOCK + di;
                for l in 0..c {
                    F::to_limbs(&vec[l], &mut tmp[..limbs]);
                    for j in 0..limbs {
                        vt[(j * c + l) * kp + i] = tmp[j];
                    }
                }
            }
        });
    }

    // ---- Phase 2: pack the left operand -------------------------------
    //
    // One flat limb array per row, element-major:
    //
    //     mat[r][l·limbs + j]  =  limb j of matrix[r][l]
    //
    // The kernel broadcasts `matrix[r][l]` to all lanes (`Elem::splat`), so
    // it only needs the element's limbs to be adjacent; no transposition.
    let mat: Vec<Vec<L<F>>> = matrix
        .par_iter()
        .map(|row| {
            let mut out = vec![L::<F>::default(); c * limbs];
            for (l, e) in row.iter().enumerate() {
                F::to_limbs(e, &mut out[l * limbs..(l + 1) * limbs]);
            }
            out
        })
        .collect();

    // ---- Phase 3: the kernel -------------------------------------------
    //
    // Output buffer per row, limb-major like `vt` but with a single `l`:
    //
    //     out[r][j·kp + i]  =  limb j of the (still weakly reduced) result
    //
    // Parallelism is two-level. The outer `par_iter` hands each row to a
    // task; that is enough when R is large (64 rows in the bench). When R
    // is small and K large (6 rows × 65536 polynomials in interpolation)
    // the row is further split into blocks of BLOCK columns so all cores
    // still take part. Both levels call the same `row_block`, which does,
    // for each group of `lanes` columns in its range: zero an accumulator,
    // `mul_add` it with (broadcast `mat[r][l]`, loaded `vt[·][l][i..]`)
    // for l in 0..C, store it.
    let out: Vec<Vec<L<F>>> = mat
        .par_iter()
        .map(|m_row| {
            let mut row_out = vec![L::<F>::default(); limbs * kp];
            let nblocks = (kp + BLOCK - 1) / BLOCK;
            if nblocks <= 1 {
                // Short row: one task, the whole column range.
                // SAFETY: AVX2 was checked by the caller (and asserted above).
                unsafe { row_block::<F::Lanes>(m_row, &vt, c, kp, &mut row_out, 0, kp) };
            } else {
                // Long row: blocks write disjoint column ranges of every limb
                // plane, so they can run concurrently on the same buffer.
                // As in phase 1 the disjointness is invisible to the borrow
                // checker, so each task gets the buffer via a raw pointer.
                let ptr = SyncPtr(row_out.as_mut_ptr());
                let len = row_out.len();
                (0..nblocks).into_par_iter().for_each(|b| {
                    let i0 = b * BLOCK;
                    let i1 = (i0 + BLOCK).min(kp);
                    let p = ptr;
                    // SAFETY: this task touches only columns i0..i1 of each
                    // limb plane; blocks are disjoint; `row_out` outlives the
                    // iterator; AVX2 was checked by the caller.
                    let slice = unsafe { std::slice::from_raw_parts_mut(p.0, len) };
                    unsafe { row_block::<F::Lanes>(m_row, &vt, c, kp, slice, i0, i1) };
                });
            }
            row_out
        })
        .collect();

    // ---- Phase 4: unpack -----------------------------------------------
    //
    // Gather the `limbs` limbs of output (r, i) from the limb planes and
    // rebuild a `FieldElement`. `from_limbs` canonicalises — the kernel
    // leaves limbs weakly reduced (e.g. the raw value `p` for zero), and
    // this is the one place that is fixed. Padding columns `k..kp` are
    // skipped.
    let results: Vec<Vec<FieldElement<F>>> = out
        .par_iter()
        .map(|row_out| {
            let mut tmp = [L::<F>::default(); LIMBS_MAX];
            (0..k)
                .map(|i| {
                    for j in 0..limbs {
                        tmp[j] = row_out[j * kp + i];
                    }
                    F::from_limbs(&tmp[..limbs])
                })
                .collect()
        })
        .collect();

    // `results` is R×K. The `!row_major` callers (interpolation, which wants
    // one row per polynomial) get the K×R transpose, as the scalar path does.
    if row_major { results } else { transpose_any(results) }
}

/// A raw pointer that Rayon may move across threads.
///
/// Used in phases 1 and 3 to let several tasks write disjoint regions of one
/// buffer. Soundness rests on the callers' disjointness arguments (see the
/// `SAFETY` comments there); the type itself asserts nothing.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
