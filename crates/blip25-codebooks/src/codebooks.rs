//! P25 vocoder vector-quantizer codebooks — derived and verified bit-exact
//! against the reference's IMBE / AMBE+2 software vocoder.
//!
//! **This crate is implementation detail of
//! [`blip25-mbe`](https://crates.io/crates/blip25-mbe).** Depend on that
//! instead; these tables are meaningless without the engine that indexes them,
//! and this crate carries no API stability promise.
//!
//! Everything here is half-rate (AMBE+2, P25 Phase 2) quantizer data, and all
//! of it is **firmware-recovered** rather than published spec data — that is
//! precisely why it needs its own crate. Full-rate IMBE quantizer tables are
//! spec-derived and live as inline arrays in `blip25-codec`'s `imbe::tables`,
//! not here. This is patent-encumbered code: see `PATENT_NOTICE.md` and
//! `ATTRIBUTION.md`, both shipped in this package.
//!
//! ## Storage
//!
//! Tables are `static [i16; N]` in [`generated`], row-major, Q11 — real value
//! = `raw / 2048.0`. They are generated from the raw extract `.bin` files
//! by `.github/scripts/gen_codebooks.py`; those
//! binaries are gitignored, and the generated source is what compiles, so a
//! fresh clone needs no `.bin` present.
//!
//! Accessors borrow the statics — no allocation, no parsing, nothing to cache.
//! Call them freely in a per-frame path.
//!
//! **Read the raw `i16` verbatim for bit-exactness.** Several tables were
//! built at scale ~2045 rather than 2048 and truncate toward zero rather than
//! rounding; reconstructing them from a formula will not reproduce the
//! reference vocoder.

mod generated;

use generated::*;

const Q11: f32 = 2048.0;

/// A row-major VQ codebook of `rows × cols` int16 entries in a fixed Q-format.
///
/// Borrows a `'static` table, so constructing one is free.
#[derive(Clone, Copy, Debug)]
pub struct Codebook {
    /// Number of codebook entries.
    pub rows: usize,
    /// Length of each entry.
    pub cols: usize,
    /// Fixed-point scale (2048.0 for Q11).
    pub scale: f32,
    /// Backing table, row-major, `rows * cols` long.
    pub raw: &'static [i16],
}

impl Codebook {
    const fn new(raw: &'static [i16], rows: usize, cols: usize, scale: f32) -> Self {
        Codebook {
            rows,
            cols,
            scale,
            raw,
        }
    }

    /// Raw int16 row — use this for bit-exact integer-domain work.
    ///
    /// # Panics
    /// If `i >= self.rows`.
    #[inline]
    pub fn raw_row(&self, i: usize) -> &'static [i16] {
        assert!(
            i < self.rows,
            "codebook row {i} out of range (rows = {})",
            self.rows
        );
        &self.raw[i * self.cols..(i + 1) * self.cols]
    }

    /// Row converted to real values via the table's Q-format.
    ///
    /// # Panics
    /// If `i >= self.rows`.
    pub fn row(&self, i: usize) -> Vec<f32> {
        self.raw_row(i)
            .iter()
            .map(|&v| v as f32 / self.scale)
            .collect()
    }
}

// ---- half-rate (AMBE+2, P25 Phase-2) VQ codebooks --------------------------

/// PRBA sub-vector VQ, 512×3, Q11, indexed by b3 (9-bit). Scale exactly 2048.
pub fn prba24() -> Codebook {
    Codebook::new(&PRBA24_RAW, 512, 3, Q11)
}

/// PRBA sub-vector VQ, 128×4, Q11, indexed by b4 (7-bit). Bit-exact vs TIA-102.BABA-A.
pub fn prba58() -> Codebook {
    Codebook::new(&PRBA58_RAW, 128, 4, Q11)
}

/// Higher-order-coefficient VQ book, 32×4, Q11.
///
/// NOTE: the HOC books TRUNCATE toward zero (vs TIA round) and were built at
/// scale ~2045, not 2048. Read the raw int16 verbatim for bit-exactness.
pub fn hoc_b5() -> Codebook {
    Codebook::new(&HOC_B5_RAW, 32, 4, Q11)
}

/// Higher-order-coefficient VQ book, 16×4, Q11. See [`hoc_b5`] on scale.
pub fn hoc_b6() -> Codebook {
    Codebook::new(&HOC_B6_RAW, 16, 4, Q11)
}

/// Higher-order-coefficient VQ book, 16×4, Q11. See [`hoc_b5`] on scale.
pub fn hoc_b7() -> Codebook {
    Codebook::new(&HOC_B7_RAW, 16, 4, Q11)
}

/// Higher-order-coefficient VQ book, 8×4, Q11. See [`hoc_b5`] on scale.
pub fn hoc_b8() -> Codebook {
    Codebook::new(&HOC_B8_RAW, 8, 4, Q11)
}

// ---- gain ladder -----------------------------------------------------------

/// Half-rate differential gain quantizer (= Annex-O), 32 levels, Q11, indexed
/// by b2.
///
/// Predictive: `gamma(0) = gain_O[b2] + 0.5 * gamma(-1)` (Eq. 168, cross-frame
/// state). Borrows a `'static`, so this is free to call per frame.
pub fn gain_o() -> &'static [i16] {
    &GAIN_O_RAW
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_consistent() {
        for (cb, name) in [
            (prba24(), "prba24"),
            (prba58(), "prba58"),
            (hoc_b5(), "hoc_b5"),
            (hoc_b6(), "hoc_b6"),
            (hoc_b7(), "hoc_b7"),
            (hoc_b8(), "hoc_b8"),
        ] {
            assert_eq!(
                cb.raw.len(),
                cb.rows * cb.cols,
                "{name}: table length does not match declared shape"
            );
            assert_eq!(
                cb.raw_row(cb.rows - 1).len(),
                cb.cols,
                "{name}: last row wrong width"
            );
        }
        assert_eq!(gain_o().len(), 32, "gain_o must be 32 levels");
    }

    #[test]
    fn accessors_borrow_the_same_static() {
        // Not merely equal — the same address. Proves there is no per-call
        // allocation or parse hiding behind the accessor.
        assert!(std::ptr::eq(prba24().raw.as_ptr(), prba24().raw.as_ptr()));
        assert!(std::ptr::eq(gain_o().as_ptr(), gain_o().as_ptr()));
    }
}
