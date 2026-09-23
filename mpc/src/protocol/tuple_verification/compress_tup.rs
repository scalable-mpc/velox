use planner::api::engine::Application;
use lambdaworks_math::{polynomial::Polynomial};
use fields::{LargeFieldSer, inverse_vandermonde_from_points, matrix_matrix_multiply, powers_matrix, rayon_async, ProtocolField};
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator};

use crate::{Context, msg::ProtMsg};

use super::{
    ex_compr_state::ExComprState, StatisticalElement, deser_statistical, inner_products_over_base, join_coeff_rows,
    join_coeffs, open_statistical_sharings, ser_statistical, split_into_coeff_rows,
};

use lambdaworks_math::field::element::FieldElement;

impl<F: ProtocolField, A: Application<F>> Context<F, A>{
    // This method starts compression from the second level onwards
    pub async fn init_compression_level(&mut self, x_vector: Vec<StatisticalElement<F>>, y_vector: Vec<StatisticalElement<F>>, agg_val: StatisticalElement<F>, depth: usize){
        // Split into chunks for compression
        let elements_per_chunk;
        if x_vector.len() >= self.compression_factor{
            // After reaching a threshold level, 
            if x_vector.len() % self.compression_factor != 0{
                elements_per_chunk = (x_vector.len()/self.compression_factor)+1;
            }
            else{
                elements_per_chunk = x_vector.len()/self.compression_factor;
            }
        }
        else{
            elements_per_chunk = 1;
        }
        let mut x_vec_chunks: Vec<Vec<StatisticalElement<F>>> = x_vector.chunks(elements_per_chunk).into_iter().map(|chunk| chunk.to_vec()).collect();
        let mut y_vec_chunks: Vec<Vec<StatisticalElement<F>>> = y_vector.chunks(elements_per_chunk).into_iter().map(|chunk| chunk.to_vec()).collect();
        let mult_value = agg_val;

        // Ensure each vector is of the same size for polynomial interpolation
        x_vec_chunks.iter_mut().for_each(|x|{
            if x.len() < elements_per_chunk{
                let new_chunk = vec![StatisticalElement::<F>::zero(); elements_per_chunk - x.len()];
                x.extend(new_chunk);
            }
        });

        y_vec_chunks.iter_mut().for_each(|x|{
            if x.len() < elements_per_chunk{
                let new_chunk = vec![StatisticalElement::<F>::zero(); elements_per_chunk - x.len()];
                x.extend(new_chunk);
            }
        });

        if !self.verf_state.ex_compr_state.contains_key(&depth){
            let ex_compr_state = ExComprState::<F>::new(depth);
            self.verf_state.ex_compr_state.insert(depth, ex_compr_state);
        }

        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth).unwrap();
        // Save the first tuple in the Structure
        let first_x_chunk = x_vec_chunks.pop().unwrap();
        let first_y_chunk = y_vec_chunks.pop().unwrap();
        ex_compr_state.rem_mult_tup = Some((first_x_chunk, first_y_chunk, mult_value));

        // Save the multiplication results in the structure
        ex_compr_state.x_sharings.extend(x_vec_chunks.clone());
        ex_compr_state.y_sharings.extend(y_vec_chunks.clone());

        if x_vec_chunks[0].len() == 1{
            // Add random mask to the compression 
            log::info!("Final level of compression, adding random mask to the compression");
            let x_rand_mask = vec![self.verf_state.random_mask.0.clone().unwrap()];
            let y_rand_mask = vec![self.verf_state.random_mask.1.clone().unwrap()];

            x_vec_chunks.push(x_rand_mask.clone());
            y_vec_chunks.push(y_rand_mask.clone());

            ex_compr_state.x_sharings.push(x_rand_mask);
            ex_compr_state.y_sharings.push(y_rand_mask);   
        }
        log::info!("Starting tuple compression at depth {} with tuple depth {} and num tuples {}", depth,x_vec_chunks[0].len(), x_vec_chunks.len());
        // Multiply these tuples using ex_mult, as inner products over `F`
        let (x_base, y_base) = inner_products_over_base::<F>(&x_vec_chunks, &y_vec_chunks);
        self.choose_multiplication_protocol(x_base, y_base, depth).await;
    }

    // This function takes a two-layered vector: 
    // First layer is a vector of tuples    
    // Second layer is encompasses a set of k vectors.  
    pub async fn init_ex_compression_tuples(&mut self, depth: usize) {
        // create polynomials on x and y
        if !self.verf_state.ex_compr_state.contains_key(&depth){
            return;
        }
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth).expect("ExComprState<F> should exist for the given depth");
        
        // Moved out, not cloned: `x_sharings` / `y_sharings` have no reader past
        // this function - the level's later steps work off `x_polys` / `y_polys`
        // and `mult_sharings` - and this function runs once per level. At the
        // first compression level these two hold the entire delinearized tuple
        // sequence, so the clone was doubling it.
        let mut x_vectors = std::mem::take(&mut ex_compr_state.x_sharings); // This should be a vector of vectors of shares for x
        let mut y_vectors = std::mem::take(&mut ex_compr_state.y_sharings); // This should be a vector of vectors of shares for y
        let mut mult_vec = ex_compr_state.mult_sharings.clone(); // This should be a vector of shares for the multiplication results

        if ex_compr_state.rem_mult_tup.is_none() {
            log::error!("Ex_compr: No remaining multiplication tuple found at depth {}, returning",depth);
            return; // Handle error: no remaining multiplication tuple
        }
        let (rem_x, rem_y, rem_mult) = ex_compr_state.rem_mult_tup.clone().unwrap();
        
        let mut mult_value_last_round = StatisticalElement::<F>::zero();
        if x_vectors[0].len() == 1{
            log::info!("Final level of compression, removing random mask from the set of multiplication tuples");
            mult_value_last_round = mult_vec.last().clone().unwrap().clone();
        }

        let sum_mult: StatisticalElement<F> = mult_vec.clone().into_iter().sum();
        let sub_mult = rem_mult - sum_mult + mult_value_last_round;
        
        // If this round is the last round, mask the output with a random sharing to ensure adversary does not know any thing about the inputs or gates
        // Add back the sum of multiplication results back into the mix for ex_compr
        x_vectors.push(rem_x);
        y_vectors.push(rem_y);
        mult_vec.push(sub_mult.clone());

        // `mult_sharings` is still read by `handle_level_mult_termination`, so
        // the remaining tuple's product is appended there as before. The x/y
        // halves are not: they live in `x_vectors` / `y_vectors` above, which is
        // where the rest of this function reads them from.
        ex_compr_state.mult_sharings.push(sub_mult);

        if x_vectors.len() != y_vectors.len() || x_vectors.len() != mult_vec.len() {
            log::error!("Ex_compr: X, Y, and Z vectors must be of the same length, returning multiplication");
            return; // Handle error: x and y vectors must be of the same length
        }

        if !ex_compr_state.extended_mult_sharings.is_empty() && ex_compr_state.extended_sharings_generated{
            // Directly go to the extended protocol now. 
            // TODO: something here
            self.handle_level_mult_termination(depth).await;
            return;
        }

        let (first_set_eval_points, second_set_eval_points) = 
            Self::gen_evaluation_points_ex_compr(x_vectors.len());
        
        log::info!("Computing Ex_compr for depth {} with polynomial of degree x: {}, y: {}, inner_tup_dimension: {}",depth, x_vectors.len(), y_vectors.len(), x_vectors[0].len());
        let mut x_polynomial_evaluations_vector = vec![vec![];x_vectors[0].len()]; // This will hold the polynomial evaluations for each x vector
        let mut y_polynomial_evaluations_vector = vec![vec![];x_vectors[0].len()]; // This will hold the polynomial evaluations for each x vector
        for (x_vec, y_vec) in x_vectors.iter().zip(y_vectors.iter()){
            for ((outer_index,x_point),y_point) in x_vec.iter().enumerate().zip(y_vec.iter()){
                x_polynomial_evaluations_vector[outer_index].push(x_point.clone());
                y_polynomial_evaluations_vector[outer_index].push(y_point.clone());
            } 
        }
        // Routed through `matrix_matrix_multiply` so the dispatcher picks up the
        // GPU path under `--features gpu`. The two batches share the same
        // evaluation points, so the inverse-Vandermonde is computed once.
        //
        // The second-set evaluation is a GEMM too. It used to be a sequential
        // `for` over the polynomials with a `par_iter` over the (small) point set
        // inside it - parallel on the short axis, serial on the long one - which
        // paid two rayon dispatches per polynomial while leaving most cores idle
        // and parked the `Context` task for the whole loop. `powers_matrix *
        // coeffs` is the same idiom already used for the first set above.
        //
        // The whole block is one yielded rayon job so the `select!` loop keeps
        // draining while it runs.
        let x_leading = x_vectors[0].len();
        let y_leading = y_vectors[0].len();
        let num_coeffs = x_vectors.len();
        let second_set = second_set_eval_points.clone();

        let (x_polynomials, y_polynomials, x_poly_evals_ss, y_poly_evals_ss) =
            rayon_async(move || {
                // The evaluations are over extension field `K` and the points over base field `F`, 
                // so the interpolation and the evaluation act on each coefficient on
                // its own: every row goes through the GEMMs as `d` rows over
                // `F` and is joined back afterwards.
                let inv_vdm_first_set =
                    inverse_vandermonde_from_points(&first_set_eval_points);
                let x_coeffs_mat =
                    matrix_matrix_multiply(&inv_vdm_first_set, &split_into_coeff_rows::<F>(&x_polynomial_evaluations_vector), false);
                let y_coeffs_mat =
                    matrix_matrix_multiply(&inv_vdm_first_set, &split_into_coeff_rows::<F>(&y_polynomial_evaluations_vector), false);

                // Evaluate every recovered polynomial at every second-set point
                // in one GEMM: row `i` of the result is
                // `[poly_0(p_i), .., poly_{d-1}(p_i)]`, each value as its
                // coefficients over `F`, one after the other.
                let second_powers = powers_matrix(&second_set, num_coeffs);
                let x_evals: Vec<Vec<StatisticalElement<F>>> = matrix_matrix_multiply(&second_powers, &x_coeffs_mat, true)
                    .iter().map(|row| join_coeffs::<F>(row)).collect();
                let y_evals: Vec<Vec<StatisticalElement<F>>> = matrix_matrix_multiply(&second_powers, &y_coeffs_mat, true)
                    .iter().map(|row| join_coeffs::<F>(row)).collect();

                let x_polynomials: Vec<Polynomial<StatisticalElement<F>>> = join_coeff_rows::<F>(&x_coeffs_mat)
                    .par_iter()
                    .map(|row| Polynomial::new(row))
                    .collect();
                let y_polynomials: Vec<Polynomial<StatisticalElement<F>>> = join_coeff_rows::<F>(&y_coeffs_mat)
                    .par_iter()
                    .map(|row| Polynomial::new(row))
                    .collect();

                // The leading zeros are what the previous code produced: it
                // pre-filled each row with `x_vectors[0].len()` zeros and then
                // pushed the evaluations after them. Preserved here so this stays
                // a pure shape change - see the note in TODO.md about whether the
                // prefill was intended.
                let pad_rows = |leading: usize, evals: Vec<Vec<StatisticalElement<F>>>| {
                    evals.into_iter()
                        .map(|row| {
                            let mut out = vec![StatisticalElement::<F>::zero(); leading];
                            out.extend(row);
                            out
                        })
                        .collect::<Vec<Vec<StatisticalElement<F>>>>()
                };
                let x_poly_evals_ss = pad_rows(x_leading, x_evals);
                let y_poly_evals_ss = pad_rows(y_leading, y_evals);

                (x_polynomials, y_polynomials, x_poly_evals_ss, y_poly_evals_ss)
            }).await;

        // Re-acquire: `ex_compr_state` borrowed `self` across the await.
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth)
            .expect("ExComprState<F> should exist for the given depth");
        ex_compr_state.x_polys = Some(x_polynomials);
        ex_compr_state.y_polys = Some(y_polynomials);

        // Record only that the extension happened. The evaluations themselves go
        // straight to multiplication below; the state copy existed solely to be
        // tested with `is_empty()`.
        ex_compr_state.extended_sharings_generated = true;
        // Send these tuples to multiplication
        // Remember, asynchrony can cause extended_mult_sharings to be filled first as well.
        let mult_sharings_filled = ex_compr_state.extended_mult_sharings.len() > 0;
        if mult_sharings_filled{
            //self.handle_ex_mult_termination(depth+1, ).await;
            self.handle_level_mult_termination(depth).await;
        }
        else{
            let (x_base, y_base) = inner_products_over_base::<F>(&x_poly_evals_ss, &y_poly_evals_ss);
            self.choose_multiplication_protocol(x_base, y_base, depth+1).await;
        }
    }

    pub async fn verify_ex_mult_termination_verification(&mut self, depth: usize, mult_result: Vec<FieldElement<F>>){
        // The multiplication protocol returns each product over `K` as its
        // `d` coefficients (see `inner_products_over_base`).
        let mult_result = join_coeffs::<F>(&mult_result);
        if depth % 2 == 0{
            // This is the first level of ex_mult termination, initiate second level of ex_mult at this depth here
            let ex_compr_state = self.verf_state.ex_compr_state.entry(depth).or_insert_with(|| ExComprState::<F>::new(depth));
            ex_compr_state.mult_sharings.extend(mult_result.clone());
            self.init_ex_compression_tuples(depth).await;
        }
        else{
            // This is the second level of ex_mult termination, initiate further compression here
            let depth_state_ex_compr = depth - 1;
            let ex_compr_state = self.verf_state.ex_compr_state.entry(depth_state_ex_compr).or_insert_with(|| ExComprState::<F>::new(depth));
            ex_compr_state.extended_mult_sharings.extend(mult_result.clone()); // Store the multiplication results for the next round of compression
            self.handle_level_mult_termination(depth_state_ex_compr).await;
        }
        // if depth == self.delinearization_depth{
        //     if mult_result.len() == 0{
        //         log::error!("Ex_compr: Mult result is empty for depth {}, returning",depth);
        //         return; // Handle error: multiplication result is empty
        //     }
        //     log::info!("Multiplication terminated at delinearization depth {}, storing result", depth);
        //     let rand_mult_sharing = mult_result[0].clone();
        //     self.verf_state.random_mask.2 = Some(rand_mult_sharing);
        // }
        // else{
            
        // }
    }

    pub async fn handle_level_mult_termination(&mut self, depth: usize){
        if !self.verf_state.ex_compr_state.contains_key(&depth) {
            // This means we haven't even started the ex_compression at this depth, return early
            return;
        }
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth).unwrap();
        // This level is already done. Both halves of a level can call in here,
        // and without this guard a second call would re-interpolate H from the
        // freed buffers and burn another common-coin sharing on a level whose
        // result has already been handed to `depth + 2`.
        if ex_compr_state.ex_compr_terminated{
            return;
        }
        if ex_compr_state.x_polys.is_none() ||
            ex_compr_state.y_polys.is_none() ||
            !ex_compr_state.extended_sharings_generated ||
            ex_compr_state.extended_mult_sharings.is_empty() {
            // We haven't filled the extended sharings yet, return early
            log::error!("handle_level_termination: Not enough data to proceed with level termination at depth {}. x_polys: {}, y_polys: {}, extended_sharings_generated: {}, extended_mult_sharings: {}",
                depth,
                ex_compr_state.x_polys.is_some(),
                ex_compr_state.y_polys.is_some(),
                ex_compr_state.extended_sharings_generated,
                ex_compr_state.extended_mult_sharings.len());
            return;
        }

        // Interpolate h polynomial
        let mut h_shares = ex_compr_state.mult_sharings.clone();
        h_shares.extend(ex_compr_state.extended_mult_sharings.iter().cloned()); // Include the multiplication results for interpolation

        let (mut evaluation_points,evaluation_points_2) = Self::gen_evaluation_points_ex_compr(ex_compr_state.mult_sharings.len());
        evaluation_points.extend(evaluation_points_2);

        // Routed through `matrix_matrix_multiply` for dispatcher-driven GPU path.
        // Single-poly interpolation: the 50k-element bailout will keep it on CPU
        // regardless, but the call site lives on the GEMM pipeline for uniformity.
        let h_inv_vdm = inverse_vandermonde_from_points(&evaluation_points);
        let h_coeffs_mat = matrix_matrix_multiply(&h_inv_vdm, &split_into_coeff_rows::<F>(&[h_shares]), false);
        let h_polynomial = Polynomial::new(&join_coeff_rows::<F>(&h_coeffs_mat)[0]);
        log::info!("Interpolated H polynomial with degree {} at ExCompr at depth {}", h_polynomial.degree(), depth);

        // Evaluate x,y,h polynomials at a random point to get final value at this level
        ex_compr_state.h_poly = Some(h_polynomial);

        // Toss coin here
        self.toss_common_coin(depth).await;
        self.verify_level_termination(depth).await;
    }

    // Satisfy idempotence here
    pub async fn verify_level_termination(&mut self, depth: usize) {
        let ex_compr_state = self.verf_state.ex_compr_state.get_mut(&depth).unwrap();
        if ex_compr_state.ex_compr_terminated{
            return;
        }
        if ex_compr_state.h_poly.is_none() || ex_compr_state.x_polys.is_none() || ex_compr_state.y_polys.is_none() || ex_compr_state.coin_output.is_none() {
            log::warn!("handle_coin_termination: h_poly:{:?}, x_poly {:?}. y_poly {:?}, common coin {:?}, Cannot proceed with coin termination at depth {}"
                            ,ex_compr_state.h_poly.is_some(),
                            ex_compr_state.x_polys.is_some(),
                            ex_compr_state.y_polys.is_some(),
                            ex_compr_state.coin_output.is_some(), 
                            depth
                        );
            return;
        }
        let h_polynomial = ex_compr_state.h_poly.as_ref().unwrap();
        let x_poly_vec = ex_compr_state.x_polys.as_ref().unwrap();
        let y_poly_vec = ex_compr_state.y_polys.as_ref().unwrap();

        let coin_eval_point = ex_compr_state.coin_output.clone().unwrap();
        let h_point = h_polynomial.evaluate(&coin_eval_point);
        let x_points: Vec<StatisticalElement<F>> = x_poly_vec.par_iter().map(|poly| poly.evaluate(&coin_eval_point)).collect();
        let y_points: Vec<StatisticalElement<F>> = y_poly_vec.par_iter().map(|poly| poly.evaluate(&coin_eval_point)).collect();
        if x_points.len() == 1{
            // Last level of compression, reconstruct sharings here
            log::info!("Last level of compression at depth {} with size of vectors {}, proceeding to reconstruct sharings",depth,x_points.len());
        }
        ex_compr_state.ex_compr_terminated = true;
        // Everything this level held has now been reduced to the three points
        // above, which are handed to `depth + 2` (a separate entry) or broadcast
        // for the final reconstruction. Free the sharings, the extended products
        // and the polynomials; the entry, `coin_output` and the coin-toss shares
        // stay so late coin messages for this depth are still deduped.
        ex_compr_state.clear_level_payload();
        log::info!("Terminated compression at depth {} with size of xvector {}, yvector {} hpoint {:?}, proceeding to next depth",depth,x_points.len(),y_points.len(),h_point);
        if x_points.len() == 1{
            log::info!("Last level of compression, reconstructing secrets");
            let prot_msg = ProtMsg::ReconstructVerfOutputSharing(
                ser_statistical::<F>(&x_points[0]), ser_statistical::<F>(&y_points[0]), ser_statistical::<F>(&h_point));
            self.broadcast(prot_msg).await;
        }
        else{
            self.init_compression_level(x_points, y_points, h_point, depth+2).await;
        }
        //self.terminate("Term".to_string()).await;
    }

    pub fn gen_evaluation_points_ex_compr(poly_def_points_count: usize)-> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let mut first_set = Vec::with_capacity(poly_def_points_count);
        let mut second_set = Vec::with_capacity(poly_def_points_count);

        for i in 1..poly_def_points_count+1{
            first_set.push(FieldElement::<F>::from(i as u64)); // Generate first set of evaluation points
            second_set.push(FieldElement::<F>::from((i+poly_def_points_count) as u64));    
        }
        (first_set, second_set)
    }

    pub async fn handle_reconstruct_verf_output_sharing(
        &mut self, 
        x_share: LargeFieldSer, 
        y_share: LargeFieldSer, 
        z_share: LargeFieldSer, 
        sender: usize){
        log::info!("handle_reconstruct_verf_output_sharing: Received shares from sender {}", sender);
        let (Some(x_share), Some(y_share), Some(z_share)) =
            (deser_statistical::<F>(&x_share), deser_statistical::<F>(&y_share), deser_statistical::<F>(&z_share)) else {
            log::warn!("handle_reconstruct_verf_output_sharing: undecodable shares from sender {}", sender);
            return;
        };
        self.verf_state.output_verf_reconstruction_shares.0.push(Self::get_share_evaluation_point(sender, self.use_fft, self.roots_of_unity.clone()));
        self.verf_state.output_verf_reconstruction_shares.1.push(x_share);
        self.verf_state.output_verf_reconstruction_shares.2.push(y_share);
        self.verf_state.output_verf_reconstruction_shares.3.push(z_share);
        
        if self.verf_state.output_verf_reconstruction_shares.0.len() == 2*self.num_faults + 1{
            // Reconstruct points and check if all 2t+1 points lie on the degree t polynomial
            let evaluation_indices = self.verf_state.output_verf_reconstruction_shares.0.clone();
            let vec_eval_points = vec![
                self.verf_state.output_verf_reconstruction_shares.1.clone(),
                self.verf_state.output_verf_reconstruction_shares.2.clone(),
                self.verf_state.output_verf_reconstruction_shares.3.clone()];
            
            let Some(secrets) = open_statistical_sharings::<F>(evaluation_indices, &vec_eval_points, self.num_faults) else {
                log::error!("handle_reconstruct_verf_output_sharing: Verification failed. Points do not lie on the polynomial.");
                return;
            };
            log::info!("handle_reconstruct_verf_output_sharing: Verification passed. Points on all three polynomials lie on degree-t polynomials.");
            log::info!("Checking if the multiplication constraint holds");

            let (a_sec, b_sec, c_sec) = (&secrets[0], &secrets[1], &secrets[2]);

            if a_sec * b_sec == *c_sec{
                log::info!("handle_reconstruct_verf_output_sharing: Multiplication constraint holds.");
                // The tuples are good; the output follows once the reveal
                // check, running alongside, has passed too.
                self.verf_state.tuples_verified = true;
                self.try_finish_verification().await;
            }
            else{
                log::error!("handle_reconstruct_verf_output_sharing: Multiplication constraint does not hold, with {:?} {:?}", a_sec* b_sec, c_sec);
                return;
            }
        }
    }
}