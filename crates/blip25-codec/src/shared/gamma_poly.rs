//! The `ptr63c` BFP polynomial (`gamma_poly_bfp_eval`) and its per-word
//! block-float round/rescale primitive (`block_float_word_rescale`).
//!
//! Shared between the encoder's loudness assembler (`enc::loudness_fixed`) and
//! the decoder's unvoiced / OLA amplitude chains (`dec::unvoiced`, `dec::ola_gen`)
//! plus the site-A `omega0` geometric mean. Kept out of `enc/` so the decoder
//! reaches it without the encoder analysis pipeline.

// The small local helpers below duplicate raw register-level x86 semantics that
// `band_decompress.rs`/`voicing_fixed.rs` also keep private copies of, rather
// than sharing them across module-privacy boundaries.
#[inline]
fn gamma_poly_s32(x: u32) -> i32 {
    x as i32
}
#[inline]
fn gamma_poly_u32(x: i32) -> u32 {
    x as u32
}
#[inline]
fn gamma_poly_s16(x: u32) -> i16 {
    x as u16 as i16
}
#[inline]
fn gamma_poly_sar32(x: u32, n: u32) -> u32 {
    ((x as i32) >> (n & 0x1f)) as u32
}
#[inline]
fn gamma_poly_shl32(x: u32, n: u32) -> u32 {
    x.wrapping_shl(n & 0x1f)
}
#[inline]
fn gamma_poly_imul32(a: i32, b: i32) -> u32 {
    (a.wrapping_mul(b)) as u32
}

/// The 4 signed int16 coefficients of the polynomial table in the reference
/// vocoder's `.rdata`. **`COEF[3]` (`0x5a82`) is bit-identical to
/// this codebase's own already-closed `band_decompress`'s
/// `LOG2_COEF[0]`** (the same `1/sqrt(2)` Q15 constant, `23170`) -- a
/// strong structural signal the BFP polynomial is genuinely part of the same
/// BFP log/sqrt-family machinery as `band_decompress::log2_fn`,
/// not an unrelated function that happens to reuse a common constant.
const GAMMA_POLY_COEF: [i16; 4] = [
    0xe676u16 as i16,
    0x714bu16 as i16,
    0x283fu16 as i16,
    0x5a82u16 as i16,
];

/// The BFP polynomial, the word-rescale primitive's sole sub-call.
///
/// It is **not** "an exponent-difference computation". It evaluates a 3-4 term
/// Horner-style fixed-point polynomial of the NORMALIZED MANTISSA of `mantissa`
/// (`m16` below), gated by the parity of a separately-tracked, by-address
/// `exp_state` value (`x0`) that is also updated in place (`(x0+1)>>1`, a
/// halving -- the "handle an odd exponent when square-rooting in log-domain"
/// idiom, consistent with `COEF[3]` being the `1/sqrt(2)` constant). Returns
/// `(poly_result_masked, new_exp_state)`.
///
/// **The Horner polynomial is fed `m16`, not `x0`.** `edx` is reloaded from the
/// just-computed normalized mantissa `dx` (`movzx eax,dx` / `movsx edx,ax`),
/// overwriting the earlier `x0`-holding value that briefly lived in `eax`.
/// Wiring `x0` in instead produces a real-vs-predicted ratio that tracks
/// `sqrt(mantissa)` almost exactly -- the shape looks right, so it is easy to
/// mistake for a scaling issue rather than a swapped operand.
pub(crate) fn gamma_poly_bfp_eval(mantissa: i32, exp_state: i16) -> (u32, i16) {
    let mant = gamma_poly_u32(mantissa);
    if mant == 0 {
        return (0, exp_state);
    }
    let scan_in = if gamma_poly_s32(mant) < 0 { !mant } else { mant };
    let bitpos: i32 = if scan_in == 0 {
        0
    } else {
        31 - scan_in.leading_zeros() as i32
    };
    let shift_amt = gamma_poly_u32(0x1e - bitpos);
    let shift_amt_low16 = shift_amt & 0xffff;

    let mut norm = gamma_poly_shl32(mant, shift_amt & 0x1f);
    norm = gamma_poly_sar32(norm, 0x10);
    let m16 = gamma_poly_s16(norm) as i32;

    let e_in16 = exp_state as u16 as u32;
    let x0_16 = gamma_poly_s16(e_in16.wrapping_sub(shift_amt_low16));
    let x0u = x0_16 as u16 as u32;

    let [c0, c1, c2, c3] = GAMMA_POLY_COEF;

    let mut acc = gamma_poly_imul32(c0 as i32, m16);
    let mut coef_term = gamma_poly_shl32(c1 as i32 as u32, 0xf);
    acc = gamma_poly_u32(gamma_poly_s32(acc).wrapping_add(gamma_poly_s32(coef_term)));
    let mut round_tmp = gamma_poly_u32(gamma_poly_s32(acc).wrapping_mul(2).wrapping_add(0x8000));
    round_tmp = gamma_poly_sar32(round_tmp, 0x10);
    let x1 = gamma_poly_s16(round_tmp) as i32;

    acc = gamma_poly_imul32(x1, m16);
    coef_term = gamma_poly_shl32(c2 as i32 as u32, 0xf);
    acc = gamma_poly_u32(gamma_poly_s32(acc).wrapping_add(gamma_poly_s32(coef_term)));
    acc = gamma_poly_u32(gamma_poly_s32(acc).wrapping_mul(2).wrapping_add(0x8000));
    acc &= 0xffff0000;

    if x0u & 1 != 0 {
        let mut sqrt_tmp = gamma_poly_sar32(acc, 0x10);
        sqrt_tmp = gamma_poly_s16(sqrt_tmp) as u32;
        sqrt_tmp = gamma_poly_imul32(c3 as i32, gamma_poly_s32(sqrt_tmp));
        acc = gamma_poly_u32(gamma_poly_s32(sqrt_tmp).wrapping_add(gamma_poly_s32(sqrt_tmp)));
    }

    let exp_i32 = x0_16 as i32;
    acc = gamma_poly_u32(gamma_poly_s32(acc).wrapping_add(0x8000));
    let exp_plus1 = exp_i32 + 1;
    acc &= 0xffff0000;
    let exp_half = exp_plus1 >> 1; // arithmetic (sign-preserving) shift-right by 1, i.e. halving the exponent
    let new_exp_state = gamma_poly_s16(gamma_poly_u32(exp_half));
    (acc, new_exp_state)
}

/// `ptr63c`'s per-word block-float round/rescale primitive -- called exactly 15x
/// per outer pass, filling the 15 "content" words of each fresh 16-word
/// coefficient block that becomes part of `ptr63c`. Word 0 of each block is a
/// separately-written clamped magnitude bin, not this function's output. It has
/// a single call site.
///
/// Pinned bit-exact against 40000 captured `(arg1,arg2,arg3) -> return` tuples.
///
/// **Coverage caveat**: every observed `arg1` is `>= 0` (magnitude-squared-shaped
/// source data). The `arg1 < 0` branch inside [`gamma_poly_bfp_eval`] (the `not eax`
/// one's-complement-for-bsr idiom) is a literal translation matching the same
/// idiom verified in `log2_fn`, but is never exercised by real input.
pub(crate) fn block_float_word_rescale(arg1: i32, arg2: i16, arg3: i16) -> i16 {
    let (poly_result, mutated_low16) = gamma_poly_bfp_eval(arg1, arg2);
    let exp_u32 = mutated_low16 as u16 as u32;
    let sh = gamma_poly_s16(exp_u32.wrapping_sub(arg3 as u16 as u32));
    let mut val = poly_result;
    let shifted = if sh >= 0 {
        gamma_poly_shl32(val, sh as u32 & 0x1f)
    } else if sh > -31 {
        gamma_poly_sar32(val, (-(sh as i32)) as u32 & 0x1f)
    } else {
        gamma_poly_sar32(val, 31)
    };
    val = gamma_poly_u32(gamma_poly_s32(shifted).wrapping_add(0x8000));
    gamma_poly_s16(gamma_poly_sar32(val, 0x10))
}
