//! Cross-rate conversion — parse at rate A, requantize, pack at rate B.
//!
//! Parameters flow through:
//!
//! ```text
//!   A-rate frame bits
//!     → parse via ambe_frames / imbe_frames
//!     → MbeParams (source generation's quantizer)
//!     → requantize to B-rate's codebook structure
//!     → pack via ambe_frames / imbe_frames at rate B
//!     → B-rate frame bits
//! ```
//!
//! The requantization step may itself invoke generation-specific
//! parameter transforms — spectral magnitude interpolation based on
//! fundamental-frequency ratios, voicing band normalization to a fixed
//! 8-band representation — per US7634399.
