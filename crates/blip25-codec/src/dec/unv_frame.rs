// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! The **unvoiced FRAME synthesizer** -- the top-level per-subframe unvoiced
//! path of the AMBE+2 decoder in the reference vocoder.
//!
//! The per-subframe synthesis dispatcher calls this **every subframe** (before
//! the voiced OLA), accumulating the unvoiced contribution into the shared
//! `out` buffer -- so it is not optional even for voiced frames.
//!
//! ## The chain (flow-walked)
//!
//! ```text
//! f(acc, n1, state, st, arg5):
//!   bit count: start 8; while (1<<(bits-1)) < sx16(n1): bits++    (n1<=128 -> 8)
//!   SIZE = (1<<bits) & 0xffff                                     (=256)
//!   window_from_table(WIN, n1, SQRT_WINDOW_TABLE, 0x50)
//!   if st.mode == 3:                                              (silence)
//!       zero state.ring[0..0x54]; zero SCRATCH[0..0x100]; count = 0
//!   else:
//!       excitation(SCRATCH, state.seed, state.ring, WIN, n1, SIZE)
//!       fwd_exp = array_forward_transform(SCRATCH, 0, bits, 0)
//!       outexp  = zx16(fwd_exp)  (shape_unvoiced_bands out_exp IN, mode2 only)
//!       shape_unvoiced_bands(SCRATCH, &outexp, SIZE, st, arg5, n1)
//!       count   = zx16(array_a_stage2_refine(SCRATCH, outexp, bits, 0))
//!   unvoiced_overlap_add_wide(acc, SCRATCH, count, state.hist, state.expptr, WIN, n1)
//! ```
//!
//! Args:
//! * arg1 = `acc`  -- OLA accumulator (i32, IN/OUT), the OLA writer's arg1.
//! * arg2 = `n1`   -- pitch/excitation length; the window/excitation n1, the
//!   OLA writer's n, and the driver of the `bitcount`/`SIZE` computation.
//! * arg3 = `state`-- persistent buffer: seed@`+0` (u16 word), 84-word
//!   ring@`+2`, 84-word hist@`+0xaa`, expptr@`+0x152` (i16 word).
//! * arg4 = `st`   -- the `UnvoicedBandParams` (mode@0, L@4, coef@0xc,
//!   amps@0x10, voic ptr@0x80, base_exp@0x84).  `[st+0]==3` selects silence.
//! * arg5 = `arg5` -- forwarded to `shape_unvoiced_bands`'s arg5 (tail zero-fill).
//!
//! (`arg6 = n1` and `n3 = SIZE` are what `shape_unvoiced_bands` receives --
//! **not** `n1` as n3.)
//!
//! ## The faithful OLA (`unvoiced_overlap_add_wide`)
//!
//! [`crate::dec::unvoiced::unvoiced_overlap_add`] takes `newexp: i16`, faithful
//! only for `newexp in [0, 0x7fff]`.  In this chain `newexp = zx16(array_a_stage2_refine ret)`
//! is a **32-bit** value the DLL reads as a full dword (`newexp - 2`, computed
//! in 32-bit).  [`unvoiced_overlap_add_wide`] is the byte-identical writer with
//! `newexp: i32` so the `zx16 >= 0x8000` cases are exact too; a unit test pins
//! it equal to the i16 port over the shared range.

use crate::dec::excitation::{excitation, window_from_table, SQRT_WINDOW_TABLE};
use crate::dec::ola_driver::block_denorm;
use crate::dec::ola_gen::zx16;
use crate::dec::unvoiced::{shape_unvoiced_bands, UnvoicedBandParams, HIST_WORDS_54};
use crate::shared::array_a_stage2::inverse_fft_butterfly_stage;

use crate::shared::loudness_transform::fft_bfp_transform;

/// Table span passed to `window_from_table`: `SQRT_WINDOW_TABLE` has
/// indices `0..=0x50`.
const WIN_TLEN: i16 = 0x50;

/// The `-32768 * -32768 -> 0x7fffffff` guard the OLA writer open-codes twice;
/// identical to `dec::unvoiced`'s private copy.
#[inline]
fn mul2_sat(a: i16, b: i16) -> i32 {
    if a == i16::MIN && b == i16::MIN {
        0x7fff_ffff
    } else {
        (a as i32).wrapping_mul(b as i32).wrapping_mul(2)
    }
}

/// Byte-identical re-port of the unvoiced OLA writer with `newexp` widened to
/// `i32`.  The DLL receives `newexp` as `zx16(array_a_stage2_refine ret)` and reads it
/// as a full 32-bit dword (`newexp - 2`), so the current-arm shift is
/// `newexp - 2` in 32-bit -- exact even when `zx16(array_a_stage2_refine ret) >= 0x8000`.
/// History-arm shift stays `zx16(oldexp - 2)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn unvoiced_overlap_add_wide(
    acc: &mut [i32],
    cur: &[i16],
    newexp: i32, // zx16(array_a_stage2_refine ret) -- 32-bit, as the DLL reads it
    hist: &mut [i16],
    expptr: &mut i16,
    win: &[i16],
    n: i16,
) {
    if n > 0 {
        let count = n as u16 as usize;
        let oldexp = *expptr;
        for i in 0..count {
            let t1 = mul2_sat(hist[i], win[count - i]);
            let r1 = block_denorm(t1, zx16(oldexp.wrapping_sub(2) as i32));
            let t2 = mul2_sat(cur[i], win[i]);
            let r2 = block_denorm(t2, newexp.wrapping_sub(2));
            let sum = r2.saturating_add(r1);
            acc[i] = acc[i].saturating_add(sum);
        }
    }
    let cnt = if n > 0 { n as u16 as usize } else { 0 };
    hist[..cnt].copy_from_slice(&cur[cnt..(cnt + cnt)]);
    for i in cnt..HIST_WORDS_54 {
        hist[i] = 0;
    }
    *expptr = newexp as i16; // write newexp's low word into expptr
}

/// Faithful port of the forward array transform: `fft_bfp_transform` FIRST, then
/// `inverse_fft_butterfly_stage` in the `arg4=0` FORWARD mode with `arg5 = arg4`
/// (= 0 in the decode call), then (if `arg4==0`) zero `arr[1]`.  Returns
/// `zx16(inverse_fft_butterfly_stage ret)`.
///
/// ```text
/// stage1_exp    = fft_bfp_transform(arr, arg2, arg3-1)
/// transform_exp = inverse_fft_butterfly_stage(arr, zx16(stage1_exp), arg3, 0, arg4)  ; array mutated in place
/// if arg4 == 0: arr[1] = 0
/// return zx16(transform_exp)
/// ```
pub(crate) fn array_forward_transform(arr: &mut [i16], arg2: i16, arg3: u32, arg4: i16) -> i16 {
    let stage1_exp = fft_bfp_transform(arr, 0, arg2, arg3 - 1);
    // inverse_fft_butterfly_stage's arg2 is the zero-extended fft_bfp_transform
    // return; it feeds only that fn's scalar return (`arg2 + 1`), whose low 16
    // bits are unchanged by zx vs sx.
    let transform_exp = inverse_fft_butterfly_stage(arr, (stage1_exp as u16) as i16, arg3, 0, arg4);
    if arg4 == 0 {
        arr[1] = 0;
    }
    (transform_exp as u16) as i16
}

/// Faithful port of array A's stage-2 refine dispatcher.
///
/// **Corrects [`array_a_stage2_refine_dispatch`] for `arg4 == 0`.**  The real disassembly passes
/// `inverse_fft_butterfly_stage`'s **arg5 = this function's own arg4**, not a
/// hardcoded `1`:
///
/// ```text
/// if arg4 == 0: arr[1] = 0
/// r1 = inverse_fft_butterfly_stage(arr, arg2, arg3, 1, arg4)  ; 5th arg = this fn's arg4, NOT literal 1
/// r2 = fft_bfp_transform(arr, r1, arg3-1)
/// return r2 - arg3 + 1
/// ```
///
/// The decode call site passes `arg4=0`, which selects
/// `inverse_fft_butterfly_stage`'s `arg5==0` head (`combine_shift(re0,im0,1)`)
/// instead of the `arg5!=0` head; `array_a_stage2_refine_dispatch` hardcodes `arg5=1` and is
/// therefore identical to this only when `arg4==1`.
pub(crate) fn array_a_stage2_refine(arr: &mut [i16], arg2: i16, arg3: u32, arg4: i16) -> i16 {
    if arg4 == 0 {
        arr[1] = 0;
    }
    let r1 = inverse_fft_butterfly_stage(arr, arg2, arg3, 1, arg4); // arg5 = arg4 (real wiring)
    let r2 = fft_bfp_transform(arr, 0, r1, arg3 - 1);
    (r2 as i32 - arg3 as i32 + 1) as i16
}

/// A view over the persistent decoder state buffer (`arg3`), by word offset:
/// `[0]` seed, `[1..85]` ring (84), `[85..169]` hist (84), `[169]` expptr.
pub(crate) const STATE_SEED: usize = 0;
pub(crate) const STATE_RING: usize = 1;
pub(crate) const STATE_HIST: usize = 85; // 0xaa / 2
pub(crate) const STATE_EXPPTR: usize = 169; // 0x152 / 2
/// Minimum length (in i16 words) of the `state` buffer.
pub(crate) const STATE_WORDS: usize = 170;

/// Faithful port of the DLL's unvoiced frame synthesizer.
///
/// * `acc`   -- OLA accumulator, `acc[0..n1]` receive `+=` the unvoiced band.
/// * `n1`    -- the excitation/pitch length.
/// * `state` -- persistent buffer (see `STATE_*`); `>= STATE_WORDS` words.
/// * `st`    -- the amplitude/voicing struct (`st.mode==3` => silence path).
/// * `arg5`  -- forwarded to `shape_unvoiced_bands`'s arg5.
pub(crate) fn synth_unvoiced_subframe(
    acc: &mut [i32],
    n1: i32,
    state: &mut [i16],
    st: &UnvoicedBandParams,
    arg5: i32,
) {
    // ---- bitcount + SIZE ----
    let n1_sx = (n1 as i16) as i32; // sign-extend n1 to 32-bit
    let mut bits: i32 = 8;
    if n1_sx > 0x80 {
        loop {
            bits += 1;
            let pow2 = 1i32 << (bits - 1); // 1 << (bits - 1)
            if pow2 >= n1_sx {
                break;
            }
        }
    }
    let size = (1i32 << bits) & 0xffff; // (1 << bits) truncated to 16 bits

    // ---- WIN = window resample (always, before the mode check) ----
    let mut win = vec![0i16; (n1 as usize) + 2];
    window_from_table(&mut win, n1 as i16, &SQRT_WINDOW_TABLE, WIN_TLEN);

    // SCRATCH holds the excitation / band spectrum; the array transforms read [SIZE]/[SIZE+1].
    let mut scratch = vec![0i16; (size as usize) + 4];

    let count: i32;
    if st.mode == 3 {
        // ---- silence path: zero ring + scratch ----
        for i in 0..HIST_WORDS_54 {
            state[STATE_RING + i] = 0; // zero the ring (0x54 words)
        }
        for s in scratch.iter_mut().take(0x100) {
            *s = 0; // zero scratch (0x100 words)
        }
        count = 0;
    } else {
        // ---- excitation: seed@state[0], ring@state[1..] ----
        {
            let (seed_slot, ring) = state.split_at_mut(STATE_RING);
            let mut seed = seed_slot[0] as u16;
            excitation(&mut scratch, &mut seed, ring, &win, n1 as i16, size as i16);
            seed_slot[0] = seed as i16;
        }
        // ---- forward transform: array_forward_transform(scratch, 0, bits, 0) ----
        let fwd_exp = array_forward_transform(&mut scratch, 0, bits as u32, 0);
        // ---- band synth: out_exp IN = zx16(fwd_exp) ----
        let mut out_exp: i16 = fwd_exp; // low 16 bits; shape_unvoiced_bands reads it as (u16)
        let _shape_ret = shape_unvoiced_bands(
            &mut scratch,
            &mut out_exp,
            size as i16,
            st,
            arg5 as i16,
            n1 as i16,
        );
        // ---- stage-2 refine: array_a_stage2_refine(scratch, out_exp, bits, 0) ----
        let count_ret = array_a_stage2_refine(&mut scratch, out_exp, bits as u32, 0);
        count = (count_ret as u16) as i32; // zero-extend low 16 bits
    }

    // ---- OLA writer: hist@state[85..169], expptr@state[169] ----
    let (front, back) = state.split_at_mut(STATE_EXPPTR);
    let hist = &mut front[STATE_HIST..STATE_EXPPTR];
    let expptr = &mut back[0];
    unvoiced_overlap_add_wide(acc, &scratch, count, hist, expptr, &win, n1 as i16);
}
