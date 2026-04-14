//! MBE baseline codec — TIA-102.BABA-A §8 and §15.
//!
//! The analysis and synthesis algorithm described in the 1993 P25 vocoder
//! specification: sinusoidal overlap-add for voiced synthesis, spectrally
//! shaped noise for unvoiced synthesis, basic W_l spectral enhancement.
//!
//! Referenced but not used by modern DVSI silicon. Per TIA-102.BABG this
//! generation explicitly cannot pass the enhanced vocoder performance tests.
//! It is implemented here for spec completeness and as the reference point
//! against which later generations' improvements are measured.
