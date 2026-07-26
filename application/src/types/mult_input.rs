use anyhow::{bail, Result};
use fields::{elem_mul, inner_products};
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// Multiplication request returned by application hooks to tell the
/// MPC layer what to multiply next.
#[derive(Clone)]
pub struct MultInput<E> {
    /// Left operands in two representations (first_half, second_half).
    pub x: (Vec<E>, Vec<E>),
    /// Right operands in two representations (first_half, second_half).
    pub y: (Vec<E>, Vec<E>),
    /// Inner product computations - x values: (first_half, second_half)
    pub x_ip: (Vec<Vec<E>>, Vec<Vec<E>>),
    /// Inner product computations - y values: (first_half, second_half)
    pub y_ip: (Vec<Vec<E>>, Vec<Vec<E>>),
    /// Circuit depth for this multiplication batch.
    pub depth: usize,
}

impl<E> MultInput<E> {
    /// Number of random sharings needed for this batch.
    /// Returns `((left_regular, left_ip), (right_regular, right_ip))` where
    /// "left"/"right" refer to the first/second half tuple positions, and
    /// "regular"/"ip" count element-wise multiplications vs inner products.
    pub fn num_rand_sharings_needed(&self) -> ((usize, usize), (usize, usize)) {
        (
            (self.x.0.len(), self.x_ip.0.len()),
            (self.x.1.len(), self.x_ip.1.len()),
        )
    }
}

impl<F> MultInput<FieldElement<F>>
where
    F: IsField,
    FieldElement<F>: Clone + Send + Sync,
{
    pub fn new(
        x: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        y: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        depth: usize,
    ) -> Result<Self> {
        check_pair_lengths(&x, &y)?;
        Ok(Self {
            x,
            y,
            x_ip: (Vec::new(), Vec::new()),
            y_ip: (Vec::new(), Vec::new()),
            depth,
        })
    }

    pub fn new_ip(
        x: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        y: (Vec<FieldElement<F>>, Vec<FieldElement<F>>),
        x_ip: (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<F>>>),
        y_ip: (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<F>>>),
        depth: usize,
    ) -> Result<Self> {
        check_pair_lengths(&x, &y)?;
        check_ip_lengths(&x_ip.0, &y_ip.0, "first half")?;
        check_ip_lengths(&x_ip.1, &y_ip.1, "second half")?;
        Ok(Self {
            x,
            y,
            x_ip,
            y_ip,
            depth,
        })
    }

    pub fn multiply(&self) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        (elem_mul(&self.x.0, &self.y.0), elem_mul(&self.x.1, &self.y.1))
    }

    pub fn inner_product(&self) -> Option<(Vec<FieldElement<F>>, Vec<FieldElement<F>>)> {
        if self.x_ip.0.is_empty() && self.x_ip.1.is_empty() {
            return None;
        }
        Some((
            inner_products(&self.x_ip.0, &self.y_ip.0),
            inner_products(&self.x_ip.1, &self.y_ip.1),
        ))
    }
}

/// A `MultInput` bundled with the preprocessing material it consumes
/// (random sharings and random zero sharings) and a slot for the
/// resulting output sharings once the multiplication completes.
#[derive(Clone)]
pub struct Multiplication<E> {
    /// The multiplication request.
    pub input: MultInput<E>,
    /// Random sharings for each half: (first_half, second_half).
    pub random_sharings: Vec<E>,
    /// Random zero sharings used by the multiplication protocol.
    pub random_zero_sharings: Vec<E>,
    /// Output sharings, attached after the multiplication completes.
    pub output: Option<(Vec<E>, Vec<E>)>,
}

impl<E> Multiplication<E> {
    pub fn new(
        input: MultInput<E>,
        random_sharings: Vec<E>,
        random_zero_sharings: Vec<E>,
    ) -> Self {
        Self {
            input,
            random_sharings,
            random_zero_sharings,
            output: None,
        }
    }

    /// Constructs a `Multiplication` carrying only output sharings —
    /// the wrapped `MultInput` and preprocessing material are left empty.
    pub fn from_output(output: (Vec<E>, Vec<E>), depth: usize) -> Self {
        Self {
            input: MultInput {
                x: (Vec::new(), Vec::new()),
                y: (Vec::new(), Vec::new()),
                x_ip: (Vec::new(), Vec::new()),
                y_ip: (Vec::new(), Vec::new()),
                depth,
            },
            random_sharings: Vec::new(),
            random_zero_sharings: Vec::new(),
            output: Some(output),
        }
    }
}

fn check_pair_lengths<E>(x: &(Vec<E>, Vec<E>), y: &(Vec<E>, Vec<E>)) -> Result<()> {
    if x.0.len() != y.0.len() {
        bail!("MultInput: x.0 ({}) and y.0 ({}) length mismatch", x.0.len(), y.0.len());
    }
    if x.1.len() != y.1.len() {
        bail!("MultInput: x.1 ({}) and y.1 ({}) length mismatch", x.1.len(), y.1.len());
    }
    Ok(())
}

fn check_ip_lengths<E>(xs: &[Vec<E>], ys: &[Vec<E>], half: &str) -> Result<()> {
    if xs.is_empty() && ys.is_empty() {
        return Ok(());
    }
    if xs.len() != ys.len() {
        bail!("MultInput: x_ip {} ({}) and y_ip {} ({}) length mismatch",
            half, xs.len(), half, ys.len());
    }
    for (i, (xv, yv)) in xs.iter().zip(ys.iter()).enumerate() {
        if xv.len() != yv.len() {
            bail!("MultInput: x_ip {}[{}] ({}) and y_ip {}[{}] ({}) length mismatch",
                half, i, xv.len(), half, i, yv.len());
        }
    }
    Ok(())
}
