//! Codec generations: analysis (PCM → MbeParams) and synthesis (MbeParams → PCM).
//!
//! Each submodule represents one generation of the MBE codec family. They
//! share the same [parameter model](crate::mbe_params) and are structurally
//! parallel; they differ in algorithmic quality, in the quantization and
//! synthesis techniques they implement, and in the rate-table indices they
//! cover in the DVSI AMBE-3000 rate table.
//!
//! | Submodule         | DVSI name   | Rate table | Primary references                     |
//! |-------------------|-------------|------------|----------------------------------------|
//! | [`mbe_baseline`]  | —           | —          | TIA-102.BABA-A §8, §15 (1993 baseline) |
//! | [`ambe`]          | AMBE-1000   | r0..r15    | US5054072, US5226084                   |
//! | [`ambe_plus`]     | AMBE-2000   | r16..r31   | +US5701390, US6199037                  |
//! | [`ambe_plus2`]    | AMBE-3000   | r32..r63   | +US8595002, US8315860                  |
//!
//! Per TIA-102.BABG the baseline generation explicitly cannot pass the
//! enhanced vocoder performance tests. It exists here for spec completeness
//! and as the reference point against which later generations' improvements
//! are understood.

pub mod ambe;
pub mod ambe_plus;
pub mod ambe_plus2;
pub mod mbe_baseline;
