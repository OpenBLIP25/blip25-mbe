//! Generic AMBE wire frame parser and packer.
//!
//! Parses a `bits @ rate_N` stream into the
//! [parameter model](crate::mbe_params) by consulting the rate table for
//! the bit allocation and FEC layout associated with `rate_N`.
