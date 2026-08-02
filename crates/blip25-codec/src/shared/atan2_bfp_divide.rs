//! Block-floating-point normalized divide from the reference codec, one
//! building block of the voicing "S" angle calculation (see
//! [`super::voicing_fixed`]). Instruction-by-instruction port, pinned bit-exact
//! on both mantissa and exponent.
//!
//! This does NOT close voicing's `S` formula: the surrounding range-reduction
//! dispatcher, the Horner polynomial and the final octant combine are open.

/// The block-floating-point divide. Returns `(mantissa, exponent)`, matching
/// the reference function's 16-bit return value and its 5th-argument output
/// write. `y` uses only its low 16 bits: the reference loads `y` in full but
/// then performs exclusively 16-bit-width operations on it. Returns `None` if
/// `y`'s absolute value is 0, where the reference would divide by zero; callers
/// never hit this for real audio.
pub(crate) fn bfp_divide(x: i32, exp_seed: i32, y: i32, exp_adjust: i32) -> Option<(i16, i16)> {
    let y16 = y as i16;

    // sign_xor: sign-extend y16 to 32 bits, then XOR with the full 32-bit x
    let sign_xor = (y16 as i32) ^ x;

    let x_abs: i32 = if x >= 0 {
        x
    } else if x == i32::MIN {
        i32::MAX
    } else {
        -x
    };

    // y_abs: abs of the 16-bit y, with -32768 saturating to 0x7fff, else
    // the low 16 bits of negating the full 32-bit value.
    let y_abs: i32 = if y16 >= 0 {
        y16 as i32
    } else if y16 == i16::MIN {
        0x7fff
    } else {
        (y as i64).wrapping_neg() as u32 as u16 as i32
    };
    if y_abs == 0 {
        return None;
    }

    let mut exp = exp_seed;
    let divisor = y_abs; // 0..=32767, safe to treat as signed
    let mut dividend = x_abs;

    let div_shl = divisor << 16; // divisor <= 32767, no i32 overflow
    if div_shl > dividend {
        // skip halving
    } else {
        dividend >>= 1;
        exp = exp.wrapping_add(1);
    }

    let mut mant: i32 = dividend / divisor; // dividend >= 0, truncating division
    mant >>= 1;
    mant &= 0xFFFF;
    if sign_xor < 0 {
        mant = if mant == 0 {
            0
        } else {
            (0x10000 - mant) & 0xFFFF
        };
    }

    let r16 = mant as i16;
    let mut norm: i32 = (r16 as i32) << 16;

    let mut sh: i32 = 0;
    if norm != 0 {
        let bsr_input: u32 = if norm >= 0 {
            norm as u32
        } else {
            !(norm as u32)
        };
        let highest_bit = 31 - bsr_input.leading_zeros() as i32;
        sh = 30 - highest_bit;
        sh &= 0xFFFF;
    }
    let shift_amt = (sh as u32) & 0x1F;
    norm = ((norm as i64) << shift_amt) as i32;
    norm >>= 16; // arithmetic
    let norm_lo = (norm as u32) & 0xFFFF;

    if norm_lo == 0 {
        return Some((0, 0));
    }

    exp = exp.wrapping_sub(sh);
    exp = exp.wrapping_sub(exp_adjust);
    let final_exp = exp as i16;
    let mantissa = norm_lo as i16;
    Some((mantissa, final_exp))
}
