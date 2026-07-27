use lambdaworks_math::polynomial::Polynomial;
use fields::LargeField;

pub struct ExComprState{
    pub depth: usize,

    pub x_sharings: Vec<Vec<LargeField>>,
    pub y_sharings: Vec<Vec<LargeField>>,
    pub mult_sharings: Vec<LargeField>,
    
    pub rem_mult_tup: Option<(Vec<LargeField>, Vec<LargeField>, LargeField)>,

    pub x_polys: Option<Vec<Polynomial<LargeField>>>,
    pub y_polys: Option<Vec<Polynomial<LargeField>>>,
    pub h_poly: Option<Polynomial<LargeField>>,

    /// Whether the extended x/y evaluations have been produced and sent to
    /// multiplication.
    ///
    /// The evaluations themselves are not kept: they were only ever consulted
    /// through `is_empty()` to sequence the two halves of a compression level,
    /// while the values went to `choose_multiplication_protocol`. Storing a
    /// second copy of them - the largest allocation of the first compression
    /// level - to answer a yes/no question is what this flag replaces.
    pub extended_sharings_generated: bool,
    pub extended_mult_sharings: Vec<LargeField>,

    // Tuple represents ordered evaluation indices as well as the shares
    pub coin_toss_shares: (Vec<LargeField>, Vec<LargeField>),
    pub coin_output: Option<LargeField>,

    pub ex_compr_terminated: bool,
}

impl ExComprState{
    /// Frees this compression level's payload once the level has terminated.
    ///
    /// At that point `verify_level_termination` has already evaluated the x, y
    /// and h polynomials at the coin point and handed those three values to the
    /// next level (which lives at `depth + 2`, in its own entry) or to the final
    /// reconstruction. Nothing reads this level's sharings, polynomials or
    /// remaining tuple again, and re-entry is short-circuited by
    /// `ex_compr_terminated`.
    ///
    /// The entry itself is deliberately kept, along with `coin_output` and the
    /// coin-toss shares: coin messages for this depth keep arriving from slower
    /// parties and are deduped against `coin_output`.
    pub fn clear_level_payload(&mut self){
        self.x_sharings = Vec::new();
        self.y_sharings = Vec::new();
        self.mult_sharings = Vec::new();
        self.extended_mult_sharings = Vec::new();
        self.rem_mult_tup = None;
        self.x_polys = None;
        self.y_polys = None;
        self.h_poly = None;
    }
}

impl ExComprState{
    pub fn new(depth: usize) -> Self {
        ExComprState{
            depth,
            x_sharings: Vec::new(),
            y_sharings: Vec::new(),
            mult_sharings: Vec::new(),

            rem_mult_tup: None,

            x_polys: None,
            y_polys: None,
            h_poly: None, // Initialize with a zero polynomial, will be set later

            extended_sharings_generated: false,
            extended_mult_sharings: Vec::new(),

            coin_toss_shares: (Vec::new(), Vec::new()), // Initialize with empty vectors for coin toss shares
            coin_output: None,  // Initialize with a zero value, will be set later

            ex_compr_terminated: false,
        }
    }    
}