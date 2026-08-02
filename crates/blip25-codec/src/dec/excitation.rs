//! Unvoiced **excitation** generator: faithful ports of the excitation
//! builder and its sqrt-window resampler from the real x86 codec DLL
//! (the reference vocoder).
//!
//! # What this is
//!
//! The routine builds the unvoiced (noise) excitation into the 256-word
//! buffer that the decoder's inverse FFT consumes. Read whole, it is
//! self-contained -- **no channel bits and no codec state** reach it:
//!
//! ```text
//!   out[i]      = q15(ring[i], win[i])        i = 0..n1-1   (ascending arm)
//!   ring[k]     = ring[k+n1]                  k = 0..84-n1-1 (shift the ring)
//!   ring[k]     = lcg_next(seed)              k = 84-n1..83  (refill the tail)
//!   out[n1+t]   = q15(ring[t], win[n1-t])     t = 0..n1-1   (descending arm)
//!   out[2*n1..nout] = 0                                     (memset tail)
//! ```
//!
//! Its only two callees are the LCG itself (a 25-byte leaf) and a word-fill.
//! The LCG is a distinct function from the adjacent *sqrt Horner poly*, which
//! sits just past this routine's terminating `ret`; reading past that `ret`
//! invents a dependency that does not exist.
//!
//! The DLL calls this with `n1 = 80`, `nout = 256` on every subframe, and the
//! ring/seed live at `state+0x648` (seed) / `state+0x64a` (84-word ring) --
//! cold-started by seed 0 plus 84 draws, which leaves the seed at
//! 34076 before the first call, so this generator is a *continuous stream*
//! from that cold start.
//!
//! # Caveat on [`window_from_table`]
//!
//! At `n1 = 80` against an 80-span table the step is exactly 1.0, so the
//! resampler degenerates to the identity and the window equals
//! [`SQRT_WINDOW_TABLE`]. Real-frame data therefore does not exercise the
//! interpolation path at all; that path is pinned only by a direct sweep of
//! `n1 = 1..=200`, 199 of which have a step != 1.0.

use crate::shared::atan2_bfp_divide::bfp_divide;

/// Length of the decoder's persistent noise ring (84 words); lives at
/// `decoder_state+0x64a`, immediately after the 16-bit LCG seed word at
/// `+0x648`.
pub(crate) const NOISE_RING_LEN: usize = 0x54; // 84

/// The sqrt-shaped amplitude window, 81 words spanning indices `0..=80`.
/// Byte-exact read from the DLL image.
///
/// Its shape is `32768*sqrt(i/80)` clamped to i16 -- but that is a
/// DESCRIPTION, not the generator: it reproduces every entry to within 1 LSB
/// and is off by one at index 27 (19037 vs 19036). The bytes are the
/// authority.
pub(crate) const SQRT_WINDOW_TABLE: [i16; 81] = [
    0, 3664, 5181, 6345, 7327, 8192, 8974, 9693, 10362, 10991, 11585, 12151, 12691, 13209, 13708,
    14189, 14654, 15105, 15543, 15969, 16384, 16789, 17184, 17570, 17948, 18318, 18681, 19037,
    19386, 19729, 20066, 20398, 20724, 21046, 21362, 21674, 21981, 22285, 22584, 22879, 23170,
    23458, 23743, 24024, 24301, 24576, 24848, 25116, 25382, 25645, 25905, 26163, 26418, 26671,
    26922, 27170, 27416, 27659, 27901, 28140, 28378, 28613, 28847, 29079, 29309, 29537, 29763,
    29988, 30211, 30432, 30652, 30870, 31086, 31302, 31515, 31727, 31938, 32148, 32356, 32563,
    32767,
];

/// The DLL's noise LCG (a 25-byte leaf):
///
/// ```text
///   mov eax,0xad / mov edx,0x3619 / imul ax,WORD PTR [ecx] / add ax,dx
///   mov WORD PTR [ecx],ax / ret
/// ```
///
/// i.e. `seed = 173*seed + 13849 (mod 2^16)`, returning the *new* seed. The
/// multiply is a 16-bit `imul`, so the whole recurrence is mod 2^16.
#[inline]
pub(crate) fn lcg_next(seed: &mut u16) -> i16 {
    *seed = seed.wrapping_mul(173).wrapping_add(13849);
    *seed as i16
}

/// The Q15 product used by both arms of the excitation builder:
/// `imul ecx,eax / add ecx,ecx / sar ecx,0x10`, then a 16-bit *truncating*
/// store (`mov WORD PTR [esi],cx`) -- not a saturating one.
#[inline]
fn q15_mul(a: i16, b: i16) -> i16 {
    let prod = (a as i32).wrapping_mul(b as i32);
    (prod.wrapping_add(prod) >> 16) as i16
}

/// Faithful port of the sqrt-window resampler: resample `tbl[0..=tlen]` onto `n1+1` points
/// via a Q16 accumulator and a Q15 lerp, writing `dst[0..=n1]`.
///
/// ```text
///   step   = (tlen / n1) in Q16, via the BFP divide + an exponent shift
///   dst[0] = tbl[0]
///   acc    = step
///   for i in 1..n1:  idx  = acc >> 16
///                    frac = (acc >> 1) & 0x7fff
///                    dst[i] = (2*((tbl[idx] << 15) + frac*(tbl[idx+1] - tbl[idx]))) >> 16
///                    acc += step
///   dst[n1] = tbl[tlen]
/// ```
///
/// `acc` is *accumulated*, not recomputed as `i*step`, so its rounding drift is
/// part of the contract.
///
/// See the module-level caveat: real frames only exercise the `n1 == tlen`
/// identity case; the interpolation path is pinned by a direct sweep.
pub(crate) fn window_from_table(dst: &mut [i16], n1: i16, tbl: &[i16], tlen: i16) {
    // --- normalize n1 --------------------------------------------------
    let n1_shifted: i32 = (n1 as i32) << 16;
    let shift1: i32 = if n1_shifted == 0 {
        0
    } else {
        let b = if n1_shifted >= 0 {
            n1_shifted as u32
        } else {
            !(n1_shifted as u32)
        };
        let highest = 31 - b.leading_zeros() as i32;
        (30 - highest) & 0xffff
    };
    // neg eax / movzx ebx,ax  -> the divide's exp_adjust, as an unsigned 16-bit
    let exp_adjust: i32 = (shift1.wrapping_neg() as u32 & 0xffff) as i32;
    let n1_norm: i32 = ((n1_shifted as u32) << (shift1 as u32 & 0x1f)) as i32;

    // --- normalize tlen ------------------------------------------------
    let tlen_shifted: i32 = (tlen as i32) << 16;
    let shift2: i32 = if tlen_shifted == 0 {
        0
    } else {
        let b = if tlen_shifted >= 0 {
            tlen_shifted as u32
        } else {
            !(tlen_shifted as u32)
        };
        let highest = 31 - b.leading_zeros() as i32;
        (30 - highest) & 0xffff
    };
    let tlen_norm: i32 = ((tlen_shifted as u32) << (shift2 as u32 & 0x1f)) as i32;

    // --- BFP divide(num, -shift2, den_mantissa, exp_adjust) ------------
    let den_mant: i32 = n1_norm >> 16; // sar edx,0x10
    let (mant, exp) = match bfp_divide(tlen_norm, shift2.wrapping_neg(), den_mant, exp_adjust) {
        Some(v) => v,
        // The DLL would divide by zero here; its callers never pass n1 = 0.
        None => panic!("0x10318960: divide-by-zero, n1={n1} normalizes to a zero mantissa"),
    };

    // --- scale the mantissa to a Q16 step ------------------------------
    let mut step: i32 = (mant as i32) << 16;
    let sh: i16 = exp.wrapping_sub(15); // sub cx,0xf, 16-bit
    if sh >= 0 {
        step = ((step as u32) << (sh as u32 & 0x1f)) as i32; // shl edi,cl
    } else if sh > -31 {
        step >>= (sh.wrapping_neg()) as u32 & 0x1f; // neg ecx / sar edi,cl
    } else {
        step >>= 31; // sar edi,0x1f
    }

    // --- endpoint + interpolation loop ---------------------------------
    dst[0] = tbl[0];
    if 1 < n1 {
        let mut count: u32 = ((n1 - 1) as u16) as u32; // dec eax / movzx eax,ax
        let mut acc: i32 = step;
        let mut dst_idx: usize = 1;
        loop {
            let idx16: u16 = ((acc >> 16) & 0xffff) as u16; // sar eax,0x10 / movzx ecx,ax
            let frac: i32 = (acc >> 1) & 0x7fff; // sar eax,1 / and eax,0x7fff
            acc = acc.wrapping_add(step); // add ebp,edi
            let i0 = ((idx16 as i16) as i32) as usize; // movsx eax,cx
            let i1 = ((idx16.wrapping_add(1) as i16) as i32) as usize; // lea eax,[ecx+1] / cwde
            let t0 = tbl[i0] as i32;
            let t1 = tbl[i1] as i32;
            let mut val = t1.wrapping_mul(frac); // imul ecx,edx
            let base = ((t0 as u32) << 15) as i32; // shl eax,0xf
            let t0f = t0.wrapping_mul(frac); // imul esi,edx
            val = val.wrapping_add(base);
            val = val.wrapping_add(val); // add ecx,ecx
            let t0f2 = t0f.wrapping_add(t0f); // add esi,esi
            val = val.wrapping_sub(t0f2); // sub ecx,esi
            dst[dst_idx] = (val >> 16) as i16; // sar ecx,0x10 / mov WORD [eax],cx
            dst_idx += 1;
            count -= 1;
            if count == 0 {
                break;
            }
        }
    }
    // --- tail: dst[n1] = tbl[tlen] -------------------------------------
    dst[n1 as usize] = tbl[tlen as usize];
}

/// Faithful port of the excitation builder: build the unvoiced excitation.
///
/// Writes `out[0..nout]`, advances the 84-word `ring` and the LCG `seed`.
/// `win` must hold `n1+1` words (indices `0..=n1`) -- the descending arm reads
/// `win[n1]`. See the module docs for the exact shape.
pub(crate) fn excitation(
    out: &mut [i16],
    seed: &mut u16,
    ring: &mut [i16],
    win: &[i16],
    n1: i16,
    nout: i16,
) {
    let n1i = n1 as i32;

    // --- ascending arm: ring x window ----------------------------------
    // `cmp ax,bp / jge` with ax = 0: skipped entirely unless n1 > 0.
    if 0 < n1 {
        let cnt = n1 as u16; // movzx ebx,ax
        for i in 0..cnt as usize {
            out[i] = q15_mul(ring[i], win[i]);
        }
    }

    // --- shift the ring down by n1 -------------------------------------
    let span = NOISE_RING_LEN as i32 - n1i; // mov ebx,0x54 / sub ebx,eax
    let mut written: i32 = 0; // xor edx,edx
    if span > 0 {
        let mut k: i32 = 0;
        loop {
            ring[k as usize] = ring[(k + n1i) as usize];
            written += 1;
            k = (written as i16) as i32; // movsx ecx,dx
            if k >= span {
                break;
            }
        }
    }

    // --- refill the tail with LCG draws --------------------------------
    // `cmp dx,0x54 / jge`: only when the shift left room.
    if (written as i16) < NOISE_RING_LEN as i16 {
        let cnt = ((NOISE_RING_LEN as i32 - written) as u16) as usize; // movzx ebp,ax
        let mut k = written as usize;
        for _ in 0..cnt {
            ring[k] = lcg_next(seed);
            k += 1;
        }
    }

    // --- descending arm: window x ring ---------------------------------
    if 0 < n1 {
        let cnt = n1 as u16; // movzx ebx,bp
        for t in 0..cnt as usize {
            out[n1i as usize + t] = q15_mul(ring[t], win[n1i as usize - t]);
        }
    }

    // --- zero the remainder --------------------------------------------
    // word-fill(dst = &out[2*n1], 0, count) -- count is read as a 16-bit signed
    // word count, and the fill is skipped when it is <= 0.
    let fill = ((nout as i32 - 2 * n1i) as i16) as i32;
    if fill > 0 {
        let start = 2 * n1i as usize;
        for v in out[start..start + fill as usize].iter_mut() {
            *v = 0;
        }
    }
}
