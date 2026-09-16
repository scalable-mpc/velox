//! The arithmetic of a public reconstruction, as functions of their inputs so
//! it can be tested in-process against a local Shamir sharing, without a
//! network.
//!
//! # The embedding
//!
//! A batch of `m` sharings `[v_1 .. v_m]` is padded to a multiple of `2t+1`
//! and split into chunks. Chunk `i` is read as the coefficients of a
//! polynomial `Z_i(X) = Σ_k v_{i,k} X^k` of degree `2t` in `X` — every party
//! holds shares of the coefficients, hence a share of `Z_i(α_p)` for every
//! party point `α_p`, of the same degree as the input sharings.
//!
//! - **L1.** Each party sends party `p` its share of every `Z_i(α_p)`. From
//!   `n − t` of those, `p` interpolates the values `Z_i(α_p)` — one point per
//!   chunk polynomial.
//! - **L2.** Each party broadcasts its points. From `2t+1` points of the
//!   degree-`2t` polynomial `Z_i`, everyone recovers its coefficients: the
//!   chunk's values.
//!
//! Every party sends `O(1)` field elements per value instead of broadcasting
//! every share.
//!
//! # Privacy of the L1 step
//!
//! What `p` learns at L1 is the sharing polynomial of `Z_i(α_p)`, a public
//! linear combination of the inputs' sharing polynomials. If those are
//! uniformly random — a fresh mask, a multiplication output — that is nothing
//! beyond the values. If they are not — a product `f_a · f_b`, whose upper
//! coefficients are fixed by honest parties' shares of the factors — a fresh
//! degree-`2t` sharing of zero is added to each L1 message, making the
//! polynomial `p` recovers uniformly random in its coefficients `1..2t`.
//! `t+1` zero sharings per chunk suffice: expanded through a Vandermonde
//! matrix, any `t` parties' terms are independent.

use crypto::hash::{do_hash, Hash};
use fields::{
    check_if_all_points_lie_on_degree_x_polynomial, interpolate_at_zero, inverse_vandermonde_from_points,
    lagrange_coefficients_at_zero, matrix_matrix_multiply, matrix_vector_multiply, powers_matrix, FieldSer,
    ProtocolField,
};
use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::*;

use super::config::SharingDegree;

/// Split `values` into chunks of `chunk_size`, padding the last with zero
/// shares — a degree-`t` sharing of zero is the all-zero share vector, so
/// every party pads identically and no preprocessing is spent. Values are
/// moved, not copied, so the batch is not resident twice.
pub fn pad_and_chunk<F: ProtocolField>(
    mut values: Vec<FieldElement<F>>,
    chunk_size: usize,
) -> (Vec<Vec<FieldElement<F>>>, usize) {
    let padding = (chunk_size - values.len() % chunk_size) % chunk_size;
    values.extend(std::iter::repeat_with(FieldElement::<F>::zero).take(padding));
    let num_chunks = values.len() / chunk_size;
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut rest = values;
    for _ in 0..num_chunks {
        let tail = rest.split_off(rest.len() - chunk_size);
        chunks.push(tail);
    }
    // `split_off` peeled chunks off the end; restore batch order.
    chunks.reverse();
    (chunks, padding)
}

/// This party's L1 message to every party: `messages[p][i]` is its share of
/// `Z_i(α_p)`. One GEMM over all chunks.
pub fn l1_messages<F: ProtocolField>(
    chunks: &[Vec<FieldElement<F>>],
    party_points: &[FieldElement<F>],
) -> Vec<Vec<FieldElement<F>>> {
    let chunk_size = chunks.first().map_or(0, |c| c.len());
    matrix_matrix_multiply(&powers_matrix(party_points, chunk_size), chunks, true)
}

/// The zero term of every L1 message: `terms[p][i]` is this party's share of
/// a degree-`2t` sharing of zero, different for every `(p, i)`. The
/// `zero_per_chunk` sharings of chunk `i` are read as coefficients and
/// evaluated at the party points, the same shape as [`l1_messages`].
pub fn zero_terms<F: ProtocolField>(
    zero_sharings: &[FieldElement<F>],
    num_chunks: usize,
    zero_per_chunk: usize,
    party_points: &[FieldElement<F>],
) -> Result<Vec<Vec<FieldElement<F>>>, String> {
    if zero_sharings.len() < num_chunks * zero_per_chunk {
        return Err(format!(
            "privacy needs {} zero sharings for {} chunks, got {}",
            num_chunks * zero_per_chunk,
            num_chunks,
            zero_sharings.len()
        ));
    }
    let zero_chunks: Vec<Vec<FieldElement<F>>> = zero_sharings[..num_chunks * zero_per_chunk]
        .chunks(zero_per_chunk)
        .map(|c| c.to_vec())
        .collect();
    Ok(matrix_matrix_multiply(&powers_matrix(party_points, zero_per_chunk), &zero_chunks, true))
}

/// L1: from the shares `n − t` senders sent this party — `shares[i]` are the
/// shares of `Z_i(α_me)`, `indices` the senders' points — recover each
/// `Z_i(α_me)`.
///
/// For degree-`t` inputs the `n − t ≥ 2t+1` shares have `t` points of
/// redundancy, and every chunk is checked to lie on one degree-`t`
/// polynomial before interpolating; a chunk that does not is a corrupt share
/// and the batch stops here. For degree-`2t` inputs there is nothing to
/// check.
pub fn l1_interpolate<F: ProtocolField>(
    indices: Vec<FieldElement<F>>,
    shares: Vec<Vec<FieldElement<F>>>,
    degree: SharingDegree,
    num_faults: usize,
) -> Result<Vec<FieldElement<F>>, String> {
    if degree == SharingDegree::T && indices.len() > num_faults + 1 {
        let (consistent, _) =
            check_if_all_points_lie_on_degree_x_polynomial(indices.clone(), shares.clone(), num_faults + 1);
        if !consistent {
            return Err(format!(
                "L1 shares of some chunk do not lie on a degree-{} polynomial: a sender sent a corrupt share",
                num_faults
            ));
        }
    }
    let lambdas = lagrange_coefficients_at_zero(&indices);
    Ok(shares.par_iter().map(|chunk| interpolate_at_zero(&lambdas, chunk)).collect())
}

/// L2: from `2t+1` parties' points on every chunk polynomial — `points[i]`
/// holds `Z_i` at `indices` — recover every chunk's coefficients, flattened
/// in batch order. Every coefficient is wanted, so a full inverse.
pub fn l2_interpolate<F: ProtocolField>(
    indices: Vec<FieldElement<F>>,
    points: Vec<Vec<FieldElement<F>>>,
) -> Vec<FieldElement<F>> {
    let inverse = inverse_vandermonde_from_points(&indices);
    points
        .par_iter()
        .map(|chunk| matrix_vector_multiply(&inverse, chunk))
        .flatten()
        .collect()
}

/// The digest the parties agree on before anyone acts on the values.
pub fn hash_of<F: ProtocolField>(values: &[FieldElement<F>]) -> Hash {
    let mut bytes = Vec::with_capacity(values.len() * F::SER_BYTES);
    for value in values {
        bytes.extend(value.ser_be());
    }
    do_hash(&bytes)
}

#[cfg(test)]
mod tests {
    //! The whole exchange, simulated in-process over a local Shamir sharing.

    use super::*;
    use fields::Mersenne61Field;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type F = Mersenne61Field;
    type E = FieldElement<F>;

    const T: usize = 3;
    const N: usize = 3 * T + 1;

    fn points() -> Vec<E> {
        (0..N).map(|p| E::from((p + 1) as u64)).collect()
    }

    /// Shamir shares of `secret` at the party points, on a random polynomial
    /// of the given degree.
    fn share(rng: &mut ChaCha20Rng, secret: E, degree: usize) -> Vec<E> {
        let coeffs: Vec<E> = std::iter::once(secret)
            .chain((0..degree).map(|_| F::from_rng(rng)))
            .collect();
        points()
            .iter()
            .map(|x| coeffs.iter().rev().fold(E::zero(), |acc, c| acc * x + c))
            .collect()
    }

    /// Shares of a batch: `batch[p]` is party `p`'s share of every value.
    fn share_batch(rng: &mut ChaCha20Rng, secrets: &[E], degree: usize) -> Vec<Vec<E>> {
        let mut batch = vec![Vec::new(); N];
        for s in secrets {
            for (p, sh) in share(rng, s.clone(), degree).into_iter().enumerate() {
                batch[p].push(sh);
            }
        }
        batch
    }

    /// Run L1 and L2 for every party with `privacy` on or off; returns what
    /// every party reconstructs, and the degree-2t polynomial party 0
    /// recovers from *all* n L1 shares of its first chunk (to inspect the
    /// privacy term).
    fn run(secrets: &[E], degree: SharingDegree, privacy: bool, corrupt: Option<usize>, seed: u8)
        -> (Result<Vec<E>, String>, Vec<E>)
    {
        let mut rng = ChaCha20Rng::from_seed([seed; 32]);
        let sharing_degree = match degree { SharingDegree::T => T, SharingDegree::TwoT => 2 * T };
        let batch = share_batch(&mut rng, secrets, sharing_degree);
        let chunk_size = 2 * T + 1;
        let pts = points();

        // Every party forms its L1 messages.
        let mut l1: Vec<Vec<Vec<E>>> = Vec::new(); // [sender][receiver][chunk]
        let mut padding = 0;
        let mut num_chunks = 0;
        for p in 0..N {
            let (chunks, pad) = pad_and_chunk(batch[p].clone(), chunk_size);
            padding = pad;
            num_chunks = chunks.len();
            let mut msgs = l1_messages(&chunks, &pts);
            if privacy {
                // t+1 degree-2t zero sharings per chunk, dealt jointly: every
                // party holds its share of the same sharings.
                let zero_secrets = vec![E::zero(); num_chunks * (T + 1)];
                let mut zrng = ChaCha20Rng::from_seed([seed ^ 0x55; 32]);
                let zero_batch = share_batch(&mut zrng, &zero_secrets, 2 * T);
                let terms = zero_terms(&zero_batch[p], num_chunks, T + 1, &pts).unwrap();
                for q in 0..N {
                    for i in 0..num_chunks {
                        msgs[q][i] = &msgs[q][i] + &terms[q][i];
                    }
                }
            }
            l1.push(msgs);
        }
        if let Some(bad) = corrupt {
            l1[bad][0][0] = &l1[bad][0][0] + E::one();
        }

        // Party 0 also fits a degree-2t polynomial through all n shares of its
        // first chunk, to look at what it learnt.
        let all_indices = pts.clone();
        let all_shares: Vec<E> = (0..N).map(|s| l1[s][0][0].clone()).collect();
        let learnt_poly = matrix_vector_multiply(&inverse_vandermonde_from_points(&all_indices), &all_shares);

        // L1 at every receiver from the first n-t senders.
        let mut l2_points: Vec<Vec<E>> = Vec::new(); // [party][chunk]
        for q in 0..N {
            let senders: Vec<usize> = (0..N - T).collect();
            let indices: Vec<E> = senders.iter().map(|s| pts[*s].clone()).collect();
            let shares: Vec<Vec<E>> = (0..num_chunks)
                .map(|i| senders.iter().map(|s| l1[*s][q][i].clone()).collect())
                .collect();
            match l1_interpolate(indices, shares, degree, T) {
                Ok(pointsq) => l2_points.push(pointsq),
                Err(e) => return (Err(e), learnt_poly),
            }
        }

        // L2 at one party from 2t+1 broadcast points.
        let voters: Vec<usize> = (0..2 * T + 1).collect();
        let indices: Vec<E> = voters.iter().map(|v| pts[*v].clone()).collect();
        let chunk_points: Vec<Vec<E>> = (0..num_chunks)
            .map(|i| voters.iter().map(|v| l2_points[*v][i].clone()).collect())
            .collect();
        let mut values = l2_interpolate(indices, chunk_points);
        values.truncate(values.len() - padding);
        (Ok(values), learnt_poly)
    }

    fn secrets(m: usize) -> Vec<E> {
        (0..m).map(|i| E::from(1000 + i as u64)).collect()
    }

    #[test]
    fn chunking_moves_values_and_pads_the_tail() {
        let (chunks, padding) = pad_and_chunk::<F>(secrets(10), 7);
        assert_eq!(padding, 4);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], secrets(7));
        assert_eq!(chunks[1][..3], secrets(10)[7..]);
        assert!(chunks[1][3..].iter().all(|z| *z == E::zero()));
        let (chunks, padding) = pad_and_chunk::<F>(secrets(14), 7);
        assert_eq!((chunks.len(), padding), (2, 0));
        let (chunks, padding) = pad_and_chunk::<F>(Vec::new(), 7);
        assert_eq!((chunks.len(), padding), (0, 0));
    }

    #[test]
    fn degree_t_round_trips_with_and_without_a_full_chunk() {
        for m in [1, 7, 8, 20] {
            let (values, _) = run(&secrets(m), SharingDegree::T, false, None, 1);
            assert_eq!(values.unwrap(), secrets(m), "m={m}");
        }
    }

    #[test]
    fn degree_2t_round_trips_with_and_without_privacy() {
        for privacy in [false, true] {
            let (values, _) = run(&secrets(9), SharingDegree::TwoT, privacy, None, 2);
            assert_eq!(values.unwrap(), secrets(9), "privacy={privacy}");
        }
    }

    /// The zero term changes what a party sees at L1 — the polynomial through
    /// all n shares — without changing the value it interpolates.
    #[test]
    fn privacy_randomises_the_l1_polynomial_but_not_the_value() {
        let (plain, poly_plain) = run(&secrets(7), SharingDegree::TwoT, false, None, 3);
        let (masked, poly_masked) = run(&secrets(7), SharingDegree::TwoT, true, None, 3);
        assert_eq!(plain.unwrap(), masked.unwrap());
        assert_eq!(poly_plain[0], poly_masked[0], "constant term is the point Z(α_0) either way");
        assert_ne!(poly_plain[1..], poly_masked[1..], "the other coefficients are re-randomised");
    }

    /// Degree-t step 1 has t points of redundancy and uses them: one corrupt
    /// L1 share is detected. Degree-2t has none, and the corruption goes
    /// through — which is what the verification phase is for.
    #[test]
    fn a_corrupt_l1_share_is_caught_at_degree_t_only() {
        let (values, _) = run(&secrets(7), SharingDegree::T, false, Some(1), 4);
        assert!(values.unwrap_err().contains("corrupt share"));
        let (values, _) = run(&secrets(7), SharingDegree::TwoT, false, Some(1), 4);
        assert_ne!(values.unwrap(), secrets(7));
    }

    #[test]
    fn hash_is_over_the_values_in_order() {
        assert_eq!(hash_of::<F>(&secrets(3)), hash_of::<F>(&secrets(3)));
        assert_ne!(hash_of::<F>(&secrets(3)), hash_of::<F>(&secrets(4)));
        let mut swapped = secrets(3);
        swapped.swap(0, 1);
        assert_ne!(hash_of::<F>(&secrets(3)), hash_of::<F>(&swapped));
    }
}
