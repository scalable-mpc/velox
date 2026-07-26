use super::Wire;

/// A share is a field element representing a party's share of a secret.
/// Generic over the field type F.
pub type Share<F> = F;

/// Represents a sharing of secret values across multiple wires.
///
/// A sharing contains a single share (field element) that represents
/// the values on multiple wires. Sharings are consumed at a specific
/// depth - either used in a gate computation or carried forward to
/// the next depth.
#[derive(Debug, Clone)]
pub struct Sharing<F> {
    /// The share value (field element)
    pub share: Share<F>,
    /// Set of wire indices this sharing represents (O(1) lookup)
    pub wires: Vec<Wire>,
    /// Depth at which this sharing is consumed
    pub depth: usize,
}

impl<F> Sharing<F> {
    /// Creates a new sharing with the given share, wires, and consumption depth.
    pub fn new(share: F, wires: Vec<Wire>, depth: usize) -> Self {
        Self { share, wires, depth }
    }

    /// Creates a sharing for a single wire.
    pub fn single(share: F, wire: Wire, depth: usize) -> Self {
        let mut wires = Vec::new();
        wires.push(wire);
        Self { share, wires, depth }
    }

    /// Returns true if this sharing represents the given wire.
    pub fn contains_wire(&self, wire: Wire) -> bool {
        self.wires.contains(&wire)
    }

    /// Returns the number of wires this sharing represents.
    pub fn num_wires(&self) -> usize {
        self.wires.len()
    }

    /// Returns a list of all wires in this sharing.
    pub fn get_wires(&self) -> Vec<Wire> {
        self.wires.iter().copied().collect()
    }

    /// Returns a sorted list of all wires in this sharing.
    pub fn get_wires_sorted(&self) -> Vec<Wire> {
        let mut wires: Vec<Wire> = self.wires.iter().copied().collect();
        wires.sort();
        wires
    }

    /// Adds a wire to this sharing.
    pub fn add_wire(&mut self, wire: Wire) {
        self.wires.push(wire);
    }
}

impl<F: Clone> Sharing<F> {
    /// Creates a new sharing with the same share but different wires.
    pub fn with_wires(&self, wires: Vec<Wire>) -> Self {
        Self {
            share: self.share.clone(),
            wires,
            depth: self.depth,
        }
    }

    /// Creates a new sharing with the same share but a different depth.
    pub fn with_depth(&self, depth: usize) -> Self {
        Self {
            share: self.share.clone(),
            wires: self.wires.clone(),
            depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharing_creation() {
        let mut wires = Vec::new();
        wires.push(0);
        wires.push(1);
        wires.push(2);

        let sharing: Sharing<u64> = Sharing::new(42, wires, 1);

        assert_eq!(sharing.share, 42);
        assert_eq!(sharing.num_wires(), 3);
        assert_eq!(sharing.depth, 1);
        assert!(sharing.contains_wire(0));
        assert!(sharing.contains_wire(1));
        assert!(sharing.contains_wire(2));
        assert!(!sharing.contains_wire(3));
    }

    #[test]
    fn test_single_wire_sharing() {
        let sharing: Sharing<u64> = Sharing::single(100, 5, 2);

        assert_eq!(sharing.share, 100);
        assert_eq!(sharing.num_wires(), 1);
        assert_eq!(sharing.depth, 2);
        assert!(sharing.contains_wire(5));
    }
}
