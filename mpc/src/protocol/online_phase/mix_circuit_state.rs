use std::collections::{HashMap, VecDeque};

use types::Replica;
use lambdaworks_math::field::element::FieldElement;
use fields::ProtocolField;

pub struct MixCircuitState<F: ProtocolField>{
    pub rand_bit_inp_shares: Vec<FieldElement<F>>,
    pub rand_bit_recon_shares: HashMap<usize, Vec<FieldElement<F>>>,

    pub rand_bit_inverse_recon_values: Vec<FieldElement<F>>,
    pub rand_bit_sharings: VecDeque<FieldElement<F>>,
    pub rand_bit_reconstruction: HashMap<usize, Vec<FieldElement<F>>>,
    /// Public reconstruction of the squared sharings, run as a linear protocol.

    pub input_acss_shares: HashMap<Replica, HashMap<usize,Vec<FieldElement<F>>>>,
    /// Set once the input sharings have been handed to the application, so the
    /// handover happens exactly once.
    pub input_sharings_forwarded: bool,
}

impl<F: ProtocolField> MixCircuitState<F>{
    pub fn new() -> Self {
        MixCircuitState{
            rand_bit_inp_shares: Vec::new(),
            rand_bit_recon_shares: HashMap::new(),

            rand_bit_inverse_recon_values: Vec::new(),
            rand_bit_sharings: VecDeque::new(),
            rand_bit_reconstruction: HashMap::default(),

            input_acss_shares: HashMap::default(),
            input_sharings_forwarded: false,
        }
    }
}