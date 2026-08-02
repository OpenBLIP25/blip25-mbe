//! Voicing's "array A stage 2": the ARRAY-mutating half of the stage routine,
//! run in its INVERSE-mode dispatch (`arg4=1`), reached from array A's stage-2
//! refine. The same function's SCALAR-only return is ported separately for the
//! loudness/`outer` side ([`super::outer_transform::combine_outer`]).
//!
//! The stage routine's forward-mode sibling dispatch (`arg4==0`) is the literal
//! sign-flipped mirror image of the inverse-mode kernel this module ports -- see
//! [`kernel_combine`].
//!
//! ## Structure
//!
//! Given a 256-i16-word array (`arg3=8` at every observed call site, i.e.
//! `n_pairs_words = 1 << arg3 = 256`, `n_complex = n_pairs_words/2 = 128`
//! complex taps):
//!
//! 1. **DC/Nyquist-boundary front-end** (`arg5!=0, arg4!=0` branch, the only one
//!    real captures exercise): combines `arr[0]` (tap 0's real part) with
//!    `arr[n_pairs_words]` -- **one word PAST the logical 256-word window** --
//!    into `arr[0]`/`arr[1]`. The "obvious" in-window index
//!    `arr[2*(n_complex-1)]` is wrong: it gives a small but real 2-3-unit
//!    mismatch on every non-silent frame.
//! 2. **Kernel combine** ([`kernel_combine`]): a front/back split-combine
//!    butterfly, front pointer walking forward from tap 1, back pointer walking
//!    backward from tap `n_complex-1`, for `(n_pairs_words>>2)-1` iterations (63
//!    for `n_pairs_words=256`).
//! 3. **Twiddle-multiply tail** ([`twiddle_multiply`]): a SECOND pass over the
//!    SAME front/back range, rotating the kernel's output by a twiddle factor
//!    indexed `scale*(iteration+1)` into the same `T1`/`T2` cosine/sine tables
//!    [`super::loudness_transform`]'s `fft_bfp_transform` uses
//!    (`scale = 0x200 >> arg3`).
//! 4. **Nyquist-tap post-processing** ([`nyquist_finish`]): tap `n_complex/2`
//!    (word index `n_pairs_words/2`) is touched by neither pass above -- its
//!    real part is halved, its imaginary part is negated-and-halved with an
//!    explicit `i16::MIN` saturate (the reference's `cmp eax,0x80000000`
//!    boundary check).
//! 5. **Scalar return**: `(arg2 as i32 + 1) as i16`. The scalar return is
//!    direction-independent -- it is the same formula
//!    [`super::outer_transform::combine_outer`] uses for the loudness side's
//!    `arg4=0` call -- only the array output differs between the two `arg4`
//!    values.
//!
//! ## Scope
//!
//! Assembling the stage-2 refine's full output needs this module's
//! [`inverse_fft_butterfly_stage`] wired to
//! [`super::loudness_transform::fft_bfp_transform`], in that order -- see
//! `Encoder::attempt_b1_fixed`.

include!("loudness_transform_tables.rs");

use crate::fixops::i16r::{i16t, s32};
#[inline]
fn cos_at(idx: i32) -> i32 {
    T1[idx as usize] as i32
}
#[inline]
fn sin_at(idx: i32) -> i32 {
    T2[idx as usize] as i32
}

/// Shared `(x<<16>>shift, y<<16>>shift)` sum/diff combine used by the
/// front-end DC/Nyquist-boundary step, matching the real DLL's own
/// literal `shl 0x10; sar shift; lea sum; sub diff; sar 0x10` instruction
/// sequence (real 32-bit wraparound, not an idealized wider-precision
/// shortcut).
fn combine_shift(x: i16, y: i16, shift: u32) -> (i16, i16) {
    let xs = (s32(x) << 16) >> shift;
    let ys = (s32(y) << 16) >> shift;
    let sum = xs.wrapping_add(ys);
    let diff = xs.wrapping_sub(ys);
    (i16t(sum >> 16), i16t(diff >> 16))
}

/// The INVERSE-mode kernel (dispatched when `arg4 != 0`): one front/back
/// butterfly combine. `(a_re,a_im)` is the front tap (BEFORE this call),
/// `(b_re,b_im)` the back/mirror tap (BEFORE this call). Returns
/// `(new_front_re, new_front_im, new_back_re, new_back_im)`.
///
/// This is the exact sign-flipped mirror image of the `arg4==0`/forward-mode
/// kernel (dispatched by the SAME stage wrapper for the loudness side's own
/// call, though loudness never reads this function's array output -- only
/// its scalar return, per [`super::outer_transform`]'s own doc): that
/// forward kernel computes `front_im=(a_im-b_im)>>1` and
/// `back_im=(b_re-a_re)>>1`, i.e. every "difference" term here has the
/// opposite sign.
fn kernel_combine(a_re: i16, a_im: i16, b_re: i16, b_im: i16) -> (i16, i16, i16, i16) {
    let a = (s32(a_re) << 16) >> 1;
    let b = (s32(a_im) << 16) >> 1;
    let c = (s32(b_re) << 16) >> 1;
    let d = (s32(b_im) << 16) >> 1;
    let front_re = i16t((a.wrapping_add(c)) >> 16);
    let front_im = i16t((d.wrapping_sub(b)) >> 16);
    let back_re = i16t((b.wrapping_add(d)) >> 16);
    let back_im = i16t((a.wrapping_sub(c)) >> 16);
    (front_re, front_im, back_re, back_im)
}

/// The twiddle-multiply tail, run over the SAME front/back range as
/// [`kernel_combine`] in a SECOND pass (operating on the kernel's own
/// output), rotating by `T1[idx]`/`T2[idx]` (`idx = scale*(iteration+1)`,
/// `scale = 0x200 >> arg3`). Returns `(new_front_re, new_front_im,
/// new_back_re, new_back_im)`.
///
/// **The back pair is not a mirror image of the front pair**, though a static
/// read makes it look like one. The front pair writes `(fr15 + d_re,
/// fi15 + d_im)`, but the back pair's imaginary half is `d_im - fi15`, NOT
/// `fi15 - d_im`. The mirrored form matches only 28 of 200 captures, all
/// silent or degenerate frames where the distinction does not bite.
fn twiddle_multiply(
    front_re: i16,
    front_im: i16,
    back_re: i16,
    back_im: i16,
    idx: i32,
) -> (i16, i16, i16, i16) {
    let t1v = cos_at(idx);
    let t2v = sin_at(idx);
    let br = s32(back_re);
    let bi = s32(back_im);
    let d_re = t1v.wrapping_mul(br).wrapping_sub(t2v.wrapping_mul(bi));
    let d_im = t2v.wrapping_mul(br).wrapping_add(t1v.wrapping_mul(bi));
    let fr15 = (s32(front_re) << 16) >> 1;
    let fi15 = (s32(front_im) << 16) >> 1;
    let new_front_re = i16t((fr15.wrapping_add(d_re)) >> 16);
    let new_back_re = i16t((fr15.wrapping_sub(d_re)) >> 16);
    let new_front_im = i16t((fi15.wrapping_add(d_im)) >> 16);
    let new_back_im = i16t((d_im.wrapping_sub(fi15)) >> 16);
    (new_front_re, new_front_im, new_back_re, new_back_im)
}

/// The Nyquist tap (word index `n_pairs_words/2`), touched by neither
/// [`kernel_combine`] nor [`twiddle_multiply`] (both stop one tap short of
/// it on each side): real part halved (`>>1` via the real DLL's own
/// `shl 0x10; sar 0x11` double-shift), imaginary part negated-and-halved
/// with an explicit `i16::MIN` saturate (the real DLL's own `cmp
/// eax,0x80000000` boundary check, since `-(-32768)` does not fit in
/// `i16`).
fn nyquist_finish(re: i16, im: i16) -> (i16, i16) {
    let new_re = i16t((s32(re) << 16) >> 17);
    let im_shifted = s32(im) << 16;
    let new_im = if im_shifted == i32::MIN {
        i16t(0x7fff_ffffi32 >> 17)
    } else {
        i16t((im_shifted.wrapping_neg()) >> 17)
    };
    (new_re, new_im)
}

/// The stage routine, INVERSE-mode dispatch only (`arg4 != 0`), the only branch
/// any capture exercises. The `arg4 == 0`/`arg5 == 0` branches are read from the
/// disassembly but not verified against capture, since voicing's call site (the
/// inverse-mode stage entry, reached from the stage-2 refine with a literal
/// `arg4=1`) never exercises them.
///
/// `arr` must be at least `(1 << arg3) + 2` i16 words long -- the `+2` is for the
/// one-past-the-window DC/Nyquist-boundary read the front-end performs, see the
/// module doc. Mutates `arr[0..1<<arg3]` in place and returns the scalar return
/// value.
pub(crate) fn inverse_fft_butterfly_stage(
    arr: &mut [i16],
    arg2: i16,
    arg3: u32,
    arg4: i16,
    arg5: i16,
) -> i16 {
    let n_pairs_words = 1usize << arg3;
    let n_complex = n_pairs_words / 2;
    let scale = 0x200i32 >> arg3;

    let re0 = arr[0];
    let im0 = arr[1];
    let re_last = arr[n_pairs_words];
    // let im_last = arr[n_pairs_words + 1]; // read by the real DLL too but unused by the arg5!=0/arg4!=0 branch

    if arg5 == 0 {
        let (a, b) = combine_shift(re0, im0, 1);
        arr[0] = a;
        arr[1] = b;
    } else if arg4 != 0 {
        let (a, b) = combine_shift(re0, re_last, 2);
        arr[0] = a;
        arr[1] = b;
    } else {
        // This branch (`arg5 != 0 && arg4 == 0`, the combination the LOUDNESS
        // call site uses) writes its second output word-pair to
        // `arr[n_pairs_words]`/`arr[n_pairs_words+1]` -- the SAME
        // one-past-the-256-word-window slot the `arg4!=0` branch above reads as
        // `re_last`. It is NOT `arr[2*(n_complex-1)]`/`arr[2*(n_complex-1)+1]`:
        // those alias the main combine loop's first "back" tap
        // (`back0 = 2*(n_complex-1)`, used below), so writing there silently
        // corrupts an in-window slot the main loop reads as input on its first
        // iteration.
        //
        // Confirmed by decompiler cross-check against the reference
        // disassembly, term for term, but NOT by per-call captures: voicing's
        // call site never exercises `arg4==0`, and the loudness call site has no
        // capture harness for the stage routine itself.
        let (a, b) = combine_shift(re0, im0, 1);
        arr[0] = a;
        arr[n_pairs_words] = b;
        arr[n_pairs_words + 1] = 0;
        arr[1] = 0;
    }

    let group_count = (n_pairs_words >> 2).saturating_sub(1);
    let front0 = 2usize;
    let back0 = 2 * (n_complex - 1);

    for it in 0..group_count {
        let fw = front0 + 2 * it;
        let bw = back0 - 2 * it;
        let (a_re, a_im, b_re, b_im) = (arr[fw], arr[fw + 1], arr[bw], arr[bw + 1]);
        let (fr, fi, br, bi) = if arg4 != 0 {
            kernel_combine(a_re, a_im, b_re, b_im)
        } else {
            // forward-mode sibling: structurally read, not
            // live-verified (see this fn's own doc).
            let a = (s32(a_re) << 16) >> 1;
            let b = (s32(a_im) << 16) >> 1;
            let c = (s32(b_re) << 16) >> 1;
            let d = (s32(b_im) << 16) >> 1;
            (
                i16t((a.wrapping_add(c)) >> 16),
                i16t((b.wrapping_sub(d)) >> 16),
                i16t((b.wrapping_add(d)) >> 16),
                i16t((c.wrapping_sub(a)) >> 16),
            )
        };
        arr[fw] = fr;
        arr[fw + 1] = fi;
        arr[bw] = br;
        arr[bw + 1] = bi;
    }

    for it in 0..group_count {
        let fw = front0 + 2 * it;
        let bw = back0 - 2 * it;
        let idx = scale.wrapping_mul((it as i32) + 1);
        let (fr, fi, br, bi) = twiddle_multiply(arr[fw], arr[fw + 1], arr[bw], arr[bw + 1], idx);
        arr[fw] = fr;
        arr[fw + 1] = fi;
        arr[bw] = br;
        arr[bw + 1] = bi;
    }

    let nyq = n_pairs_words / 2;
    let (nre, nim) = nyquist_finish(arr[nyq], arr[nyq + 1]);
    arr[nyq] = nre;
    arr[nyq + 1] = nim;

    i16t((arg2 as i32) + 1)
}
