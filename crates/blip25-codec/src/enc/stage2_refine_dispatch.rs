//! Array A's "stage 2" refine dispatcher -- the sole call its parent stage
//! makes -- composed from two pieces:
//!
//! 1. [`super::array_a_stage2::inverse_fft_butterfly_stage`] (INVERSE-mode,
//!    `arg4=1`).
//! 2. [`super::loudness_transform::fft_bfp_transform`].
//!
//! ## Wiring
//!
//! The dispatcher's call chain, read from the disassembly:
//!
//! ```text
//! push [esp+0x1c]      ; inverse_fft_butterfly_stage's arg2 = the wrapper's own arg2
//! push edi             ; inverse_fft_butterfly_stage's arg1 = ptr (the wrapper's own arg1)
//! call inverse_fft_butterfly_stage   ; inverse_fft_butterfly_stage(ptr, arg2, arg3=esi, arg4=1, arg5=1)
//! movzx ecx,ax         ; ecx = inverse_fft_butterfly_stage's own i16 return (r1)
//! lea eax,[esi-1]      ; eax = arg3 - 1 (esi = the wrapper's own arg3, untouched by the call)
//! push eax             ; fft_bfp_transform's arg3 = orig_arg3 - 1
//! push ecx             ; fft_bfp_transform's arg2 = r1
//! push edi             ; fft_bfp_transform's arg1 = SAME ptr, mutated in place by inverse_fft_butterfly_stage
//! call fft_bfp_transform   ; fft_bfp_transform(ptr, r1, orig_arg3-1) -> r2
//! add esp,0x20
//! sub eax,esi          ; eax = r2 - orig_arg3
//! inc eax              ; eax = r2 - orig_arg3 + 1  (the WHOLE wrapper's own return)
//! ret
//! ```
//!
//! A direct hook on the inner `fft_bfp_transform` call confirms this
//! independently of the reading above: across 200 captures the call site always
//! passes `arg2=5` (`inverse_fft_butterfly_stage`'s return) and `arg3=7`
//! (`orig_arg3(8) - 1`).
//!
//! ## Standing constraint
//!
//! `fft_bfp_transform`'s group loop fires on **every** stage in this call
//! context, not stage 1 only: each `fft_bfp_transform` call makes exactly 63
//! inner butterfly-combine calls (`63 = 32+16+8+4+2+1`, the radix-2 DIT pattern
//! for a full 6-stage pass over `n_pairs=128`, which `arg3=7` selects). A
//! "stage 1 only" reading holds for some loudness-path calls but is wrong here
//! -- do not narrow `fft_bfp_transform` back to it.

/// Array A's stage-2 refine dispatcher, composing
/// [`super::array_a_stage2::inverse_fft_butterfly_stage`] with
/// [`super::loudness_transform::fft_bfp_transform`] per the wiring in this
/// module's doc. Matches the reference on both the mutated array and the scalar
/// return.
pub(crate) fn array_a_stage2_refine_dispatch(arr: &mut [i16], arg2: i16, arg3: u32, arg4: i16) -> i16 {
    if arg4 == 0 {
        arr[1] = 0;
    }
    let r1 = super::array_a_stage2::inverse_fft_butterfly_stage(arr, arg2, arg3, 1, 1);
    let inner_arg3 = arg3 - 1;
    let r2 = super::loudness_transform::fft_bfp_transform(arr, 0, r1, inner_arg3);
    (r2 as i32 - arg3 as i32 + 1) as i16
}
