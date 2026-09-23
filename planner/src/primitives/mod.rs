//! The building blocks the operations are made of: a random value with its
//! bits, the bitwise less-than schedule, and the Mersenne-prime arithmetic
//! both rest on. None knows about operations, steps or the engine.

pub mod carry_tree;
pub mod edabit;
pub mod mersenne;
