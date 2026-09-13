# Mersenne-61 GEMM on AVX2

How `fields/src/simd` vectorises `matrix_matrix_multiply` over the Mersenne-61
prime field `p = 2^61 − 1` and its degree-4 extension (the protocol's
`DefaultField`), why the multiply looks the way it does, and what the
measurements say.

Code: [`fields/src/simd/m61_avx2.rs`](../fields/src/simd/m61_avx2.rs) (lanes and
tower), [`fields/src/simd/gemm.rs`](../fields/src/simd/gemm.rs) (kernel and
packing), [`fields/src/simd/mod.rs`](../fields/src/simd/mod.rs) (backend
selection). Tests: [`fields/tests/simd_gemm.rs`](../fields/tests/simd_gemm.rs).
Bench: [`fields/benches/simd_gemm.rs`](../fields/benches/simd_gemm.rs).

Contents

1. [The problem AVX2 poses for a 61-bit field](#1-the-problem-avx2-poses-for-a-61-bit-field)
2. [Representation and the weak-reduction invariant](#2-representation-and-the-weak-reduction-invariant)
3. [The base-field multiply, step by step](#3-the-base-field-multiply-step-by-step)
4. [Addition, subtraction and the fused forms](#4-addition-subtraction-and-the-fused-forms)
5. [The Fp2 / Fp4 tower, lane-wise](#5-the-fp2--fp4-tower-lane-wise)
6. [The GEMM kernel](#6-the-gemm-kernel)
7. [Dispatch and fallback](#7-dispatch-and-fallback)
8. [Testing](#8-testing)
9. [Measurements](#9-measurements)
10. [Where the remaining time goes, and what would recover it](#10-where-the-remaining-time-goes-and-what-would-recover-it)
11. [Appendix A — every AVX2 intrinsic used](#11-appendix-a--every-avx2-intrinsic-used)
12. [Appendix B — notation](#12-appendix-b--notation)

---

## 1. The problem AVX2 poses for a 61-bit field

The scalar `Mersenne61Field::mul` is one instruction plus a short reduction:

```rust
fn mul(a: &u64, b: &u64) -> u64 {
    Self::from_u128(u128::from(*a) * u128::from(*b))   // MUL r64 → 128-bit product
}
```

x86-64 has a 64×64→128 scalar multiply, so the 122-bit product of two 61-bit
values is free to form. **AVX2 has no vector equivalent.** Its only wide
integer multiply is `vpmuludq` (`_mm256_mul_epu32`), which, in each 64-bit
lane, multiplies the *low 32 bits* of the two operands and produces a 64-bit
product. There is no 64×64→64 (`vpmullq` is AVX-512DQ) and no 64×64→128.

So a vector 61-bit multiply has to be assembled from 32-bit halves, the way one
would do long multiplication by hand in base 2^32. That costs four `vpmuludq`
per lane-product and a handful of shifts and adds to recombine — the reason M61
gains less from AVX2 than a field whose product fits in 64 bits (Mersenne-31
gets 8 products from two `vpmuludq`).

What AVX2 *is* good at is everything else: the adds, subtracts and weak
reductions that make up the bulk of the Fp2/Fp4 tower arithmetic vectorise
perfectly, four lanes at a time. That is where most of the measured speedup
comes from (§9).

## 2. Representation and the weak-reduction invariant

A register `M61x4` holds four independent field elements, one per 64-bit lane.
The four lanes are four different output columns of the GEMM (§6); there is no
mixing between lanes anywhere in the arithmetic.

### The reduction identity

Because `p = 2^61 − 1`, we have `2^61 ≡ 1 (mod p)`, and therefore for any
64-bit `x`:

```
x = q·2^61 + r        with  q = x >> 61  (0 ≤ q ≤ 7),   r = x & p  (0 ≤ r ≤ p)
x ≡ q + r  (mod p)
```

This is `weak_reduce`:

```rust
unsafe fn weak_reduce(x: __m256i) -> __m256i {
    _mm256_add_epi64(_mm256_srli_epi64::<61>(x), _mm256_and_si256(x, P_VEC))
}
```

Three instructions (shift, and, add), no compare, no branch. The result is at
most `p + 7 = 2^61 + 6`. It is **not** canonical: it can be `p` (which encodes
zero) or a little above `p`. We deliberately stop here.

### The invariant

> Every lane value produced by any `M61x4` operation lies in the *weak range*
> `[0, 2^61 + 8)`. Every operation accepts inputs in that range (most accept
> anything `< 2^62`).

Not canonicalising after every operation is the single biggest saving in the
whole design: a canonical reduction needs a compare-and-conditional-subtract
(`min_epu64` does not exist in AVX2; it takes a compare, a mask and a blend),
and it would be paid after every one of the ~30 base-field operations in an Fp4
multiply. Instead each operation is written so that its inputs' weak bounds
guarantee its intermediate sums never wrap 64 bits, and its output is weakly
reduced again. §3–§4 state those bounds for each operation.

The scalar code has the same property — `Mersenne61Field::sub(a, a)` returns
the raw value `p`, and `IsField::eq` compares via `as_representative` — so lane
inputs coming from scalar elements may already be `p`. Every lane operation
accepts that. Canonicalisation happens exactly once, when the output limbs are
turned back into `FieldElement`s (`SimdField::from_limbs` calls
`FieldElement::new`, whose `from_u64` maps `[0, 2^64)` to `[0, p)`).

## 3. The base-field multiply, step by step

Inputs: `a, b < 2^62` per lane. Split each into 32-bit halves:

```
a = a_hi·2^32 + a_lo        a_lo < 2^32,   a_hi = a >> 32 < 2^30
b = b_hi·2^32 + b_lo
```

The full product is

```
a·b = a_hi·b_hi·2^64  +  (a_hi·b_lo + a_lo·b_hi)·2^32  +  a_lo·b_lo
    =      hh·2^64    +            mid·2^32            +     ll
```

### Step 1 — the four partial products (4 × `vpmuludq`, 2 shifts)

```rust
let a_hi = _mm256_srli_epi64::<32>(a);
let b_hi = _mm256_srli_epi64::<32>(b);
let ll = _mm256_mul_epu32(a,    b);      // a_lo·b_lo  < 2^64
let lh = _mm256_mul_epu32(a,    b_hi);   // a_lo·b_hi  < 2^62
let hl = _mm256_mul_epu32(a_hi, b);      // a_hi·b_lo  < 2^62
let hh = _mm256_mul_epu32(a_hi, b_hi);   // a_hi·b_hi  < 2^60
```

`vpmuludq` reads only the low 32 bits of each lane, so `a` and `b` are passed
unmasked where `a_lo`/`b_lo` are wanted — no `and` needed. After the shift the
high halves sit in the low 32 bits, so `a_hi`/`b_hi` are directly usable.

```rust
let mid = _mm256_add_epi64(lh, hl);      // < 2^63, cannot wrap
```

### Step 2 — fold each term modulo p

We now have `a·b = hh·2^64 + mid·2^32 + ll` where each term is a 64-bit lane.
Reduce each term separately using `2^61 ≡ 1`:

**`hh·2^64`.** `2^64 = 8·2^61 ≡ 8`, so `hh·2^64 ≡ 8·hh`:

```rust
let hh8 = _mm256_slli_epi64::<3>(hh);    // < 2^63
```

**`mid·2^32`.** Split `mid` at bit 29 (because 29 + 32 = 61):

```
mid = mid_hi·2^29 + mid_lo
=> mid_lo = mid & (2^29 − 1)   
=> mid_hi = mid >> 29
```

Finally, we have:
```
mid·2^32 = mid_hi·2^61 + mid_lo·2^32  ≡  mid_hi + mid_lo·2^32
```

```rust
let mid_hi = _mm256_srli_epi64::<29>(mid);                                  // < 2^34
let mid_lo = _mm256_slli_epi64::<32>(_mm256_and_si256(mid, MASK29));        // < 2^61
```

**`ll`.** Plain weak split:

```rust
let ll_hi = _mm256_srli_epi64::<61>(ll);   // < 8
let ll_lo = _mm256_and_si256(ll, P_VEC);   // < 2^61
```

### Step 3 — sum and weak-reduce

```rust
let s = hh8 + mid_hi + mid_lo + ll_hi + ll_lo;     // four vpaddq
Self(Self::weak_reduce(s))
```

The bound that makes this sound:

```
s  <  2^63 + 2^34 + 2^61 + 8 + 2^61  =  2^63 + 2^62 + 2^34 + 8  <  2^64
```

so the sum does not wrap, and `weak_reduce(s) ≤ 2^61 + 6`, inside the weak
range. Every input bound above assumed `a, b < 2^62`; weak-range inputs
(`< 2^61 + 8`) satisfy that with a factor of two to spare.

### Instruction count

| stage | instructions |
|---|---|
| high halves | 2 shifts |
| partial products | 4 `vpmuludq` |
| `mid` | 1 add |
| folds | 1 shift (`hh8`), 1 shift (`mid_hi`), 1 and + 1 shift (`mid_lo`), 1 shift (`ll_hi`), 1 and (`ll_lo`) |
| sum | 4 adds |
| weak reduce | 1 shift, 1 and, 1 add |
| **total** | **21 for four products ≈ 5.25 per product** |

Constants (`P_VEC`, `MASK29`) are hoisted by the compiler. For comparison the
scalar path is `MUL` + a ~7-instruction `from_u128` per product; the vector
version wins on the multiply alone only modestly (Skylake issues `vpmuludq`
2/cycle versus scalar `MUL` 1/cycle).

## 4. Addition, subtraction and the fused forms

All bounds below assume weak-range inputs unless stated. `2p = 2^62 − 2` and
`4p = 2^63 − 4` are the multiples of `p` used to keep subtractions non-negative
without a compare.

| op | computes | inner value | precondition | instructions |
|---|---|---|---|---|
| `add(a, b)` | `a + b` | `< 2^63` | `a, b < 2^62` | 1 + 3 |
| `sub(a, b)` | `a + 2p − b` | `[0, 2^63)` | `a < 2^62`, `b ≤ 2p` | 2 + 3 |
| `sub2(a, b, c)` | `a + 4p − b − c` | `[0, 2^64)` | `a < 2^62`, `b + c ≤ 4p` | 3 + 3 |
| `mul4_sub(a, b)` | `4a + 2p − b` | `[0, 2^64)` | `a < 3·2^60`, `b ≤ 2p` | 3 + 3 |
| `add_mul4(a, b)` | `a + 4b` | `< 2^64` | `a + 4b < 2^64` | 2 + 3 |

(The "+ 3" is the trailing `weak_reduce`.) The fused forms exist because the
tower formulas contain `z − x − y` and `4x − y` / `x + 4y`; doing those as two
separate operations would cost two reductions where one suffices. In the Fp4
multiply this removes 15 instructions per lane-group (about 5 %).

Why the preconditions hold in practice: every value flowing into these
operations is either a fresh `weak_reduce` output (`≤ 2^61 + 6`) or a raw
scalar element (`≤ p`). Both are far inside every bound in the table — e.g.
`mul4_sub` needs `a < 3·2^60 ≈ 3.5·10^18` and receives `a ≤ 2.3·10^18`.

## 5. The Fp2 / Fp4 tower, lane-wise

The tower is the one in `mersenne_61/extensions.rs`:

* Fp2 = Fp[i] / (i² + 1), elements `c0 + c1·i`
* Fp4 = Fp2[w] / (w² − (4 + i)), elements `c0 + c1·w`

`Fp2x4` and `Fp4x4` are structs of two `M61x4` / two `Fp2x4`, and their
methods are the scalar formulas with every `+ − ×` replaced by the lane
operation. Because the lanes are independent this is a *transliteration*, not a
new algorithm — the equivalence tests (§8) check it against the scalar tower
directly.

**Fp2 multiply** (Karatsuba, 3 base multiplies instead of 4):

```
a0b0 = a0·b0
a1b1 = a1·b1
z    = (a0 + a1)·(b0 + b1)
c0   = a0b0 − a1b1                 // sub
c1   = z − a0b0 − a1b1             // sub2
```

**Fp2 × (4 + i)** (the Fp4 non-residue): `(4a0 − a1) + (a0 + 4a1)·i` —
`mul4_sub` and `add_mul4`, one reduction each.

**Fp4 multiply** (Karatsuba again, 3 Fp2 multiplies):

```
a0b0 = a0·b0
a1b1 = a1·b1
z    = (a0 + a1)·(b0 + b1)
c0   = a0b0 + (4 + i)·a1b1
c1   = z − a0b0 − a1b1             // Fp2 sub2
```

Cost per lane-group of four Fp4 products: 9 base multiplies (189
instructions), 12 adds, 3 subs, 5 sub2, 1 `mul4_sub`, 1 `add_mul4` ≈ 293
instructions, plus 16 for the accumulate → **~77 instructions per Fp4
multiply-add**. The scalar tower is ~180–200 per multiply-add; the measured
ratio (§9) is 2.3–2.5×, consistent with that.

## 6. The GEMM kernel

`matrix_matrix_multiply(matrix, vectors, row_major)` computes
`out[r][i] = Σ_l matrix[r][l] · vectors[i][l]` for an R×C `matrix` and K
`vectors` of length C.

### Which dimension to vectorise

K is the large dimension at every call site (1024 in the bench, 8192 chunks in
`lin_mult`'s party evaluation, up to 65536 polynomials in interpolation),
while C is 6–128. So the four lanes of a register hold **four consecutive
output columns `i..i+4`** of the same row, and the inner loop runs over `l`.
Each iteration broadcasts one `matrix[r][l]` (four `vpbroadcastq`, one per
limb) and loads the corresponding four `vectors[i..i+4][l]` limbs with one
256-bit load per limb.

### Layouts

For those loads to be contiguous the right operand is repacked once:

```
vt[(j·C + l)·kp + i] = limb j of vectors[i][l]        kp = K rounded up to a multiple of 4
```

i.e. limb-major, transposed, zero-padded. The left operand is flattened to
`mat[r][l·4 + j]`. The output is accumulated per row as `out[r][j·kp + i]` and
unpacked into `Vec<Vec<FieldElement>>` at the end (canonicalising, §2).

`FieldElement<F>` is not `#[repr(transparent)]` in lambdaworks, so slices of
elements are never reinterpreted as limbs; packing goes through `value()` and
unpacking through `FieldElement::new`. That costs O(K·C + R·C + R·K) element
copies against O(R·C·K) arithmetic (§10).

### Kernel

```rust
#[target_feature(enable = "avx2")]
unsafe fn row_block<E: Elem>(m_row, vt, c, kp, out, i0, i1) {
    for i in (i0..i1).step_by(E::LANES) {
        let mut acc = E::zero();
        for l in 0..c {
            let m = E::splat(&m_row[l * E::LIMBS]);        // broadcast matrix[r][l]
            let v = E::load(&vt[l * kp + i], c * kp);       // vectors[i..i+4][l]
            acc = acc.mul_add(m, v);
        }
        acc.store(&mut out[i], kp);
    }
}
```

`Elem` is implemented by `M61x4` (1 limb) and `Fp4x4` (4 limbs); the kernel is
monomorphised per field. The `#[inline(always)]` lane methods fold into this
`#[target_feature]` function, which is the only place AVX2 instructions are
emitted — the rest of the crate is compiled for baseline x86-64.

### Parallelism

Rayon parallelises over rows, and within a row over blocks of 256 output
columns when K is large, so both the tall shapes (R = 64) and the flat ones
(R = 6, K = 65536) fill all cores. Packing of `vectors` is parallel over the
same column blocks.

## 7. Dispatch and fallback

`matrix_matrix_multiply` in `poly.rs` tries, in order: GPU (`try_gpu_gemm`,
only with `--features gpu` and above a size threshold), SIMD
(`try_simd_gemm`), scalar (`matrix_matrix_multiply_cpu`). Both hooks are
`ProtocolField` methods defaulting to `None`; `Mersenne61Field` and
`Mersenne61Degree4ExtensionField` override `try_simd_gemm` to call
`simd::gemm`.

Three independent fallbacks guarantee the scalar path is always reachable:

| level | mechanism | effect |
|---|---|---|
| compile time | `#[cfg(target_arch = "x86_64")]` on the `simd` module and on the overrides | other targets never see the module; the default `None` sends everything to scalar |
| run time | `is_x86_feature_detected!("avx2")`, evaluated once and cached in `simd::backend()` | CPUs without AVX2 use scalar; nothing faults |
| operator | `VELOX_SIMD=off` (or `0`, `scalar`) in the environment | scalar even on AVX2 hardware — for A/B benchmarks of the whole protocol |

`simd::gemm_with(Backend::Avx2, ..)` re-checks the feature bit, so even an
explicit request for AVX2 on an unsupported CPU runs the scalar path rather
than executing an illegal instruction.

Shape mismatches (a vector whose length differs from C) are reported the same
way on every path: an error log and an empty result.

## 8. Testing

`fields/tests/simd_gemm.rs`:

* **Equivalence** — for each of M61 base and Fp4 (and the M31 fields), random
  inputs on eight shapes including K below, equal to and one above the lane
  count, both `row_major` settings, compared with `assert_eq` on
  `FieldElement`s (semantic equality, so a raw `p` and a canonical `0` agree).
* **Edge lanes** — matrices built from `{0, 1, p−1, p, p−2, 2^60}` as *raw*
  limb values (`const_from_raw`), placed in every limb of Fp4 elements, so the
  non-canonical `p` encoding and the top of the range pass through every
  operation.
* **Dispatcher** — `matrix_matrix_multiply` equals `matrix_matrix_multiply_cpu`
  for both M61 fields, and also under `VELOX_SIMD=off`.
* **Scalar reference** — `simd::scalar_gemm` (the `IsField`-only copy used as
  the fallback and by the M31 fields) equals `matrix_matrix_multiply_cpu`.
* Empty and mismatched inputs.

The existing `gemm_eval_equiv` and `vandermonde_closed_form` tests now run
through the SIMD path as well, since they call `matrix_matrix_multiply`.

## 9. Measurements

Single core (`RAYON_NUM_THREADS=1`), i7-8565U (AVX2, no AVX-512), rustc
1.97.1, criterion means. The laptop was thermally throttling throughout, so the
absolute times are pessimistic; the ratios are reliable because scalar and AVX2
alternate within each group.

| shape R×C, K | field | scalar | AVX2 | speedup | ns / multiply-add (scalar → AVX2) |
|---|---|---|---|---|---|
| 64×128, 1024 | M61 Fp4 | 241 ms | 119 ms | 2.0× | 28.7 → 14.2 |
| 16×11, 8192 | M61 Fp4 | 45 ms | 19 ms | 2.4× | 31.3 → 12.9 |
| 6×6, 65536 | M61 Fp4 | 78 ms | 32 ms | 2.5× | 33.2 → 13.4 |
| 64×128, 1024 | M61 base | 15.0 ms | 12.2 ms | 1.2× | 1.8 → 1.4 |
| 16×11, 8192 | M61 base | 3.1 ms | 2.1 ms | 1.5× | 2.1 → 1.5 |
| 6×6, 65536 | M61 base | 5.9 ms | 4.7 ms | 1.2× | 2.5 → 2.0 |

For reference, the same kernel structure over Mersenne-31 (8 × u32 lanes, one
`vpmuludq` per product) gives 4–5.6× on its Fp4/Fp8 towers and ~2× on the base
field; at matched soundness (M31-Fp8 vs M61-Fp4, 32 bytes per element either
way) M61-Fp4 remains 1.4× faster under AVX2 (119 ms vs 172 ms at 64×128).

Reading the table:

* **Fp4, the field the protocol runs:** ~2.4× per core, flat across shapes.
  The AVX2 cost per multiply-add (13–14 ns) barely moves with shape; the
  scalar cost rises at small C because each dot product starts with a pointer
  chase into a separate `Vec`.
* **Base field:** the multiply is not the scalar bottleneck (M31-base scalar
  is the same 1.8 ns), so the vector multiply's modest advantage plus the
  per-element pack/unpack overhead nets 1.2–1.5×.

## 10. Where the remaining time goes, and what would recover it

Per multiply-add the work is R·C·K lane operations; per *element* the
pack/unpack costs are fixed. Relative to the arithmetic that is
`a/R + b/K + c/C` — visible only when both R and C are small, and dominant for
the base field where the arithmetic itself is ~1 ns.

In rough order of payoff:

1. **Keep operands in limb layout across consecutive GEMMs.** `poly.rs`
   evaluates `share_powers · (inv_vdm · evals)` — the second GEMM repacks the
   first's output. A limb-layout matrix type passed between them removes one
   pack and one unpack per element.
2. **Register-block over rows.** The kernel reloads each `vt` vector for every
   row; processing 2–4 rows per pass reuses each load 2–4× and halves the
   broadcast traffic. Costs registers (an Fp4 accumulator is 4 of the 16 `ymm`).
3. **Defer the accumulator reduction.** `acc + product` can absorb two or
   three products before a `weak_reduce` (bound `4·2^61 < 2^64`), saving ~3
   instructions per multiply-add of the 77.
4. **AVX-512F** doubles the lane count with the same construction; on Ice
   Lake and later, **AVX-512 IFMA** (`vpmadd52luq/huq`, 52×52→104 fused
   multiply-add) replaces the four-`vpmuludq` product with two 52-bit-limb
   multiplies and would be the real step change for M61.

## 11. Appendix A — every AVX2 intrinsic used

Fifteen intrinsics, all in `fields/src/simd/m61_avx2.rs` and `m31_avx2.rs`
(`grep -rhoE '_mm256_[a-z0-9_]+' fields/src | sort -u`). Each acts on all
lanes of a 256-bit register independently; nothing here crosses lanes except
`vpblendd`, and that only selects, never mixes.

### Memory and constants

| intrinsic | instruction | does | used for |
|---|---|---|---|
| `_mm256_loadu_si256` | `vmovdqu` | load 32 bytes (unaligned) | `load` — one limb of 4 (M61) or 8 (M31) consecutive output columns from the packed `vt` |
| `_mm256_storeu_si256` | `vmovdqu` | store 32 bytes (unaligned) | `store` — an accumulator limb back to the row buffer |
| `_mm256_set1_epi64x` | `vpbroadcastq` | one u64 into all 4 lanes | `M61x4::splat` (the matrix element `matrix[r][l]`); the constants `p`, `2p`, `4p`, `2^29 − 1` |
| `_mm256_set1_epi32` | `vpbroadcastd` | one u32 into all 8 lanes | `M31x8::splat`; the constant `p` |
| `_mm256_setzero_si256` | `vpxor` | all-zero register | `zero()` — accumulator reset |

### Arithmetic

| intrinsic | instruction | does | used for |
|---|---|---|---|
| `_mm256_mul_epu32` | `vpmuludq` | per 64-bit lane, multiply the **low 32 bits** of both operands (unsigned) into a 64-bit product | the only wide multiply AVX2 has. M61: the four partial products `ll, lh, hl, hh` (§3). M31: once for the even lanes, once for the odd lanes shifted down — eight full products |
| `_mm256_add_epi64` | `vpaddq` | 64-bit lane add, wrapping | M61 `add`; summing the five folded pieces in `mul`; the final add of `weak_reduce`; `+2p` / `+4p` in the subtractions; M31 `fold64` |
| `_mm256_sub_epi64` | `vpsubq` | 64-bit lane subtract, wrapping | M61 `sub`, `sub2`, `mul4_sub` — always after adding a multiple of `p`, so the result is non-negative |
| `_mm256_add_epi32` | `vpaddd` | 32-bit lane add | M31 `add`, `weak_reduce`, `+p` in `sub` |
| `_mm256_sub_epi32` | `vpsubd` | 32-bit lane subtract | M31 `sub`, `neg` (`p − a`) |

### Shifts and masks — the reduction machinery

| intrinsic | instruction | does | used for |
|---|---|---|---|
| `_mm256_srli_epi64::<N>` | `vpsrlq` | logical (zero-fill) right shift of each 64-bit lane by `N` | `>>32`: isolate `a_hi`, `b_hi` (M61) / move odd lanes down (M31); `>>61`: high part in `weak_reduce`, `ll_hi`; `>>29`: `mid_hi`; `>>31`: M31 `fold64` |
| `_mm256_slli_epi64::<N>` | `vpsllq` | logical left shift of each 64-bit lane by `N` | `<<3`: `8·hh`; `<<32`: `mid_lo·2^32`, re-raising M31 odd lanes; `<<2`: `4·a` in `mul4_sub`, `add_mul4` |
| `_mm256_srli_epi32::<31>` | `vpsrld` | logical right shift of each 32-bit lane | M31 `weak_reduce`: the carry above 2^31 |
| `_mm256_and_si256` | `vpand` | bitwise AND | `x & p` (the low 61 or 31 bits) in every reduction; `mid & (2^29 − 1)` |
| `_mm256_blend_epi32::<0b10101010>` | `vpblendd` | per 32-bit lane, take from operand A or B by immediate mask | M31 `mul`: interleave the even-lane products with the odd-lane products (shifted back up) into one 8 × u32 register |

### Gating

Not instructions, but what makes the above legal to execute:

* `#[target_feature(enable = "avx2")]` on `row_block` (`gemm.rs`) — the one
  function into which every `#[inline(always)]` lane method is inlined, and so
  the only place AVX2 code is emitted. The rest of the crate compiles for
  baseline x86-64.
* `is_x86_feature_detected!("avx2")` in `simd::backend()` and `gemm_with` —
  the runtime check that keeps `row_block` from being called on a CPU without
  AVX2.

Notably absent: compares, min/max, permutes and cross-lane shuffles, and
`vpmullq` (64-bit low multiply, AVX-512DQ only). Both fields are built from
adds, shifts, masks and `vpmuludq` alone — that is what makes the
weak-reduction design branch-free.

## 12. Appendix B — notation

All operators act on integers held in one lane, independently per lane, and
everything is unsigned.

| symbol | meaning | example |
|---|---|---|
| `x << n` | shift the bits of `x` left by `n`, zero-filling on the right; numerically `x · 2^n` as long as nothing falls off the top | `hh << 3` is `8·hh` |
| `x >> n` | shift right by `n`, dropping the low `n` bits; numerically `⌊x / 2^n⌋`, i.e. the bits of `x` *above* position `n` | `x >> 61` is bits 61–63 of `x`, a value in 0–7; `a >> 32` is the high half of `a` |
| `x & m` | bitwise AND | `x & p` with `p = 2^61 − 1` (61 one-bits) is "the low 61 bits of `x`" |
| `<`, `≤` | ordinary magnitude comparisons on the lane's integer value; they appear **only in comments and bounds**, never as instructions | `lh < 2^62, hl < 2^62 ⇒ mid < 2^63`: the sum fits in 64 bits without wrapping |
| `≡` | congruent modulo `p`: a different integer but the same field element | `2^61 ≡ 1`; `x ≡ (x >> 61) + (x & p)` |

Shift-and-mask together implement "split a number at bit `n`":

```
x = (x >> n)·2^n + (x & (2^n − 1))
      high part        low part
```

which, with `2^61 ≡ 1`, is the whole reduction: `x ≡ (x >> 61) + (x & p)`.
The comparisons are the proof, done once, that each intermediate stays below
2^64 — so the instructions never have to check.
