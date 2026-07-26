use std::fmt;

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use super::{Sharing, Wire};

/// Network Routing module uses degree-2t sharings whereas our multiplication module uses degree-3t/2 sharings.
/// So, this struct is to facilitate conversion from degree-3t/2 to degree-2t sharings.

#[derive(Clone)]
pub struct HalfSharingsTransformation<E> {
    // We need to combine and merge these sharings into singular ones
    pub pair_sharings: (Vec<Sharing<E>>, Vec<Sharing<E>>),

    // Degree-2t sharings with r in first half, 0 in second half - l/t.
    pub random_sharings_fhp: Vec<E>,
    // Degree-3t/2 sharings - l/t.
    pub random_sharings_reverse_fhp: Vec<E>,
    // Forward transformation
    pub random_zero_sharings: Vec<E>,

    // Degree-2t sharings with r in first half, 0 in second half, 0 in first half, r in second half
    // l/t pairs
    pub random_dual_sharings_shp: (Vec<E>, Vec<E>),
    // Degree-3t/2 sharings with r in second half, r in first half.
    // l/t pairs
    pub random_dual_sharings_reverse_shp: (Vec<E>, Vec<E>),

    // Degree-3t sharings with 0 in both halves
    // l/t.
    pub random_zero_reverse_sharings: Vec<E>,
}

impl<E> fmt::Display for HalfSharingsTransformation<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "HalfSharingsTransformation {{")?;
        writeln!(f, "    pair_sharings: ({}, {})", self.pair_sharings.0.len(), self.pair_sharings.1.len())?;
        writeln!(f, "    random_sharings_fhp: {}", self.random_sharings_fhp.len())?;
        writeln!(f, "    random_sharings_reverse_fhp: {}", self.random_sharings_reverse_fhp.len())?;
        writeln!(f, "    random_zero_sharings: {}", self.random_zero_sharings.len())?;
        writeln!(f, "    random_dual_sharings_shp: ({}, {})", self.random_dual_sharings_shp.0.len(), self.random_dual_sharings_shp.1.len())?;
        writeln!(f, "    random_dual_sharings_reverse_shp: ({}, {})", self.random_dual_sharings_reverse_shp.0.len(), self.random_dual_sharings_reverse_shp.1.len())?;
        writeln!(f, "    random_zero_reverse_sharings: {}", self.random_zero_reverse_sharings.len())?;
        write!(f, "  }}")
    }
}

impl<F> HalfSharingsTransformation<FieldElement<F>>
where
    F: IsField,
    FieldElement<F>: Clone + Send + Sync,
{
    pub fn new(
        pair_sharings: (Vec<Sharing<FieldElement<F>>>, Vec<Sharing<FieldElement<F>>>),
        random_sharings_fhp: Vec<FieldElement<F>>,
        random_sharings_reverse_fhp: Vec<FieldElement<F>>,
        random_zero_sharings: Vec<FieldElement<F>>,
        random_dual_sharings_shp: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        random_dual_sharings_reverse_shp: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        random_zero_reverse_sharings: Vec<FieldElement<F>>,
    ) -> Result<Self, String> {
        let first_len = pair_sharings.0.len();
        let second_len = pair_sharings.1.len();
        let zero_sharings_needed = (first_len + second_len).div_ceil(2);

        if random_sharings_fhp.len() < first_len {
            return Err(format!(
                "random_sharings_fhp has {} sharings, need at least {}",
                random_sharings_fhp.len(),
                first_len
            ));
        }
        if random_sharings_reverse_fhp.len() < first_len {
            return Err(format!(
                "random_sharings_reverse_fhp has {} sharings, need at least {}",
                random_sharings_reverse_fhp.len(),
                first_len
            ));
        }
        if random_zero_reverse_sharings.len() < zero_sharings_needed {
            return Err(format!(
                "random_zero_reverse_sharings has {} sharings, need at least {}",
                random_zero_reverse_sharings.len(),
                zero_sharings_needed
            ));
        }
        if random_zero_sharings.len() < zero_sharings_needed {
            return Err(format!(
                "random_zero_sharings has {} sharings, need at least {}",
                random_zero_sharings.len(),
                zero_sharings_needed
            ));
        }
        if random_dual_sharings_shp.0.len() < second_len {
            return Err(format!(
                "random_dual_sharings_shp.0 has {} sharings, need at least {}",
                random_dual_sharings_shp.0.len(),
                second_len
            ));
        }
        if random_dual_sharings_shp.1.len() < second_len {
            return Err(format!(
                "random_dual_sharings_shp.1 has {} sharings, need at least {}",
                random_dual_sharings_shp.1.len(),
                second_len
            ));
        }
        if random_dual_sharings_reverse_shp.0.len() < second_len {
            return Err(format!(
                "random_dual_sharings_reverse_shp.0 has {} sharings, need at least {}",
                random_dual_sharings_reverse_shp.0.len(),
                second_len
            ));
        }
        if random_dual_sharings_reverse_shp.1.len() < second_len {
            return Err(format!(
                "random_dual_sharings_reverse_shp.1 has {} sharings, need at least {}",
                random_dual_sharings_reverse_shp.1.len(),
                second_len
            ));
        }

        Ok(Self {
            pair_sharings,
            random_sharings_fhp,
            random_sharings_reverse_fhp,
            random_zero_sharings,
            random_dual_sharings_shp,
            random_dual_sharings_reverse_shp,
            random_zero_reverse_sharings,
        })
    }
}

#[derive(Clone)]
pub struct NetworkRouting<E> {
    pub depth: usize,

    pub half_sharings: Option<HalfSharingsTransformation<E>>,
    pub input_sharings: Option<Vec<Sharing<E>>>,

    /// F_SELECT related stuff
    pub select_target_mapping: Vec<Vec<Wire>>,
    // l/t sharings
    pub select_random_sharings: Vec<E>,
    // l/2t sharings
    pub select_random_zero_sharings: Vec<E>,

    /// Network Routing related stuff
    pub target_mapping: Vec<Vec<Wire>>,
    // l/t
    pub random_sharings_nr_1: Vec<E>,
    // l/t
    pub random_sharings_nr_2: Vec<E>,
    // l/t
    pub random_sharings_nr_3: Vec<E>,

    /// F_SELECT related reverse transformation
    /// So, basically need to isolate wires and bring them all into the same sharing.
    pub select_reverse_target_mapping: Vec<Vec<Wire>>,
    // l/t sharings
    pub select_reverse_random_sharings: Vec<E>,
    // l/t sharings
    pub select_reverse_random_zero_sharings: Vec<E>,
}

impl<E> fmt::Display for NetworkRouting<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NetworkRouting {{")?;
        writeln!(f, "  depth: {}", self.depth)?;
        match &self.half_sharings {
            Some(hs) => writeln!(f, "  half_sharings: {}", hs)?,
            None => writeln!(f, "  half_sharings: None")?,
        }
        writeln!(f, "  input_sharings: {}", match &self.input_sharings {
            Some(v) => format!("Some(len={})", v.len()),
            None => "None".to_string(),
        })?;
        writeln!(f, "  -- F_SELECT forward --")?;
        writeln!(f, "  select_target_mapping: {} groups", self.select_target_mapping.len())?;
        writeln!(f, "  select_random_sharings: {}", self.select_random_sharings.len())?;
        writeln!(f, "  select_random_zero_sharings: {}", self.select_random_zero_sharings.len())?;
        writeln!(f, "  -- Network Routing --")?;
        writeln!(f, "  target_mapping: {} groups", self.target_mapping.len())?;
        writeln!(f, "  random_sharings_nr_1: {}", self.random_sharings_nr_1.len())?;
        writeln!(f, "  random_sharings_nr_2: {}", self.random_sharings_nr_2.len())?;
        writeln!(f, "  random_sharings_nr_3: {}", self.random_sharings_nr_3.len())?;
        writeln!(f, "  -- F_SELECT reverse --")?;
        writeln!(f, "  select_reverse_target_mapping: {} groups", self.select_reverse_target_mapping.len())?;
        writeln!(f, "  select_reverse_random_sharings: {}", self.select_reverse_random_sharings.len())?;
        writeln!(f, "  select_reverse_random_zero_sharings: {}", self.select_reverse_random_zero_sharings.len())?;
        write!(f, "}}")
    }
}
