//! `w_builder`: constructs the 127-tap window/mask array (`w` in
//! [`super::windowed_complex_correlation::windowed_complex_correlation`]'s signature) that the correlation windows array
//! `A` by, from raw per-band inputs. Called once per `windowed_complex_correlation` invocation, at a
//! fixed call site in the caller's body, immediately before the 4-call chain
//! that leads into the atan2 chain and `windowed_complex_correlation`.
//!
//! **`w` is not a static table.** It is a frame-varying, audio-responsive
//! 8-band binary gate: 12 distinct values across 200 captured calls, with the
//! "constant run" value `0x4000` (16384) and run start/end positions that vary
//! substantially call to call (`total_nonzero` ranges from 0 to 115 of the 127
//! taps). Reading it as a per-tap-position static table -- "first ~9 entries 0,
//! then a long run of 0x0040" -- is wrong on both the value and the constancy.
//!
//! ## What this computes
//!
//! `w_builder(exp_arg, thresh8, table8)` fills 127 output words in 8 chunks:
//!
//! ```text
//! out[0..12)    = 0                                    (12 words, always)
//! out[12]       = gate(table8[0], thresh8[0], exp_arg)  (1 word)
//! out[13..29)   = gate(table8[1], thresh8[1], exp_arg)  (16 words)
//! out[29..45)   = gate(table8[2], thresh8[2], exp_arg)  (16 words)
//! out[45..61)   = gate(table8[3], thresh8[3], exp_arg)  (16 words)
//! out[61..77)   = gate(table8[4], thresh8[4], exp_arg)  (16 words)
//! out[77..93)   = gate(table8[5], thresh8[5], exp_arg)  (16 words)
//! out[93..109)  = gate(table8[6], thresh8[6], exp_arg)  (16 words)
//! out[109..127) = gate(table8[7], thresh8[7], exp_arg)  (18 words)
//! ```
//!
//! Note the chunk boundaries are **not** a uniform 8x16-word partition: words
//! 0-11 are unconditionally zero and word 12 alone uses band 0, so every
//! subsequent chunk is shifted. Do not "regularize" this.
//!
//! Each chunk's fill value is a **two-stage binary gate** ([`gate`], below) that
//! writes `16384` (`0x4000`, a flat Q14 "pass" coefficient) or `0` (mute) for
//! the whole chunk:
//!
//! 1. Block-floating-point normalize `table8[i]` (the same BFP-normalize idiom
//!    as `band_decompress::normalize64`, in a 16/32-bit variant local to this
//!    function): compute `shift = 30 - bsr(|table8[i]| << 16)`. If
//!    `exp_arg - shift < -42`, the whole chunk is gated OFF (`0`) regardless of
//!    `thresh8[i]` -- a per-band "does this band's magnitude clear a global
//!    reference level" test.
//! 2. Otherwise, `16384` if `thresh8[i] >= 6553` (`0x1999`), else `0`.
//!
//! `table8`/`thresh8` are two DIFFERENT incoming pointer arguments -- the
//! caller's locals at the call site, at fixed run-constant offsets from the
//! caller's persistent context argument (`thresh8` at caller arg4 + 1590,
//! `table8` at caller arg4 + 1670) -- each read as 8 `int16` entries, one per
//! chunk.
//!
//! ## What this does NOT close
//!
//! The caller's arg4 is `encoder_buf + 0x1c`, so `thresh8 == encoder_buf+0x652`
//! and `table8 == encoder_buf+0x6a2`.
//!
//! **`table8` needs no separate formula**: it is literally
//! [`super::voicing_fixed::band_voicing_heap_src`]'s output (array B, gap 2)
//! for the PRECEDING call -- the copy-ring function's shift-copy ring moves
//! `table8`'s content into `heap_src`'s slot (`encoder_buf+0x6c2`) every call.
//!
//! `thresh8` is open. It lives in the same ring family -- a parallel,
//! structurally similar but numerically DIFFERENT 5-generation track -- and its
//! writer is somewhere inside the copy-ring function's unread "post-copy-loop
//! tail", which is where `encoder_buf+0x652` (`thresh8`'s generation-0 ring
//! slot) receives fresh content every call.
//!
//! Array B's `window` writer is NOT that writer, despite also writing a
//! second 8-word array shaped like `thresh8`/`table8`. That array (`param_3`)
//! is a small caller-local stack slot, nowhere near `encoder_buf+0x652`, and
//! holds small BFP exponents (e.g. `[4,-2,-6,-9,-9,-9,-10,-11]`) rather than
//! `thresh8`'s large magnitudes (same frame: `[11723,13869,14351,14864,14719,
//! 14253,14285,14475]`). Do not re-test that hypothesis.
//!
//! Both `thresh8` and `table8` are PCM-computable only once `x` -- the pitch
//! tracker's return value, `band_voicing_heap_src`'s required input -- is
//! closed. `C` (`windowed_complex_correlation`'s `arg5`) and the atan2 chain's post-atan2 tail (see
//! [`super::atan2_chain`]) are separately open.

use crate::fixops::u32r::bsr32;

/// The per-chunk two-stage gate inlined in `w_builder`'s loop body.
/// `table_val`/`thresh_val` are one chunk's `table8[i]`/`thresh8[i]`; `exp_arg`
/// is the function's shared 4th argument, the same value for every chunk.
/// Returns `16384` (pass) or `0` (mute).
fn gate(table_val: i16, thresh_val: i16, exp_arg: i16) -> i16 {
    // Sign-extend table_val to 32-bit, then shift left 16 -- places the
    // 16-bit value in the top half of a 32-bit word.
    let x: u32 = ((table_val as i32) << 16) as u32;
    if x == 0 {
        return 0;
    }
    // mag = (x as i32) < 0 ? !x : x. Note NOT of a negative 32-bit value
    // with bit31 set always clears bit31 in the result, so mag here always
    // has bit31 == 0.
    let mag: u32 = if (x as i32) < 0 { !x } else { x };
    let shift_count = 30u32.wrapping_sub(bsr32(mag));
    // x86 SHL masks the shift count to 5 bits, so mirror that (shift_count
    // is always in 0..=30 here in practice, given mag's bit31 is always
    // clear, so this mask is a no-op safety net).
    let sh = shift_count & 0x1f;
    let x_shifted = x.wrapping_shl(sh);
    if x_shifted == 0 {
        return 0;
    }
    // diff = exp_arg - shift_count -- compute in i32 on the sign-extended
    // operands, equivalent to the original 16-bit subtract-and-test.
    let diff = (exp_arg as i32) - (shift_count as i32);
    if diff < -42 {
        return 0;
    }
    if (thresh_val as i32) >= 6553 {
        16384
    } else {
        0
    }
}

/// `thresh8`/`table8` are the caller-local 8-entry per-band arrays (see module
/// doc); `exp_arg` is the shared scalar 4th argument. Returns the 127-word `w`
/// array [`super::windowed_complex_correlation::windowed_complex_correlation`] consumes as its `w` parameter.
pub(crate) fn w_builder(exp_arg: i16, thresh8: [i16; 8], table8: [i16; 8]) -> [i16; 127] {
    let mut out = [0i16; 127];
    // Words 0..12 are unconditionally zero: they fall outside the chunk loop's
    // effective span. See the module doc's note on the chunk boundaries.
    out[12] = gate(table8[0], thresh8[0], exp_arg);
    let chunk_bounds: [(usize, usize); 7] = [
        (13, 29),
        (29, 45),
        (45, 61),
        (61, 77),
        (77, 93),
        (93, 109),
        (109, 127),
    ];
    for (i, (start, end)) in chunk_bounds.iter().enumerate() {
        let val = gate(table8[i + 1], thresh8[i + 1], exp_arg);
        for slot in out[*start..*end].iter_mut() {
            *slot = val;
        }
    }
    out
}
