//! DVSI soft-decision decoder packet — the chip handoff format.
//!
//! Per the DVSI application note *"Soft Decision Error Decoding with the
//! AMBE+2™ Vocoder Chips"* (2022). When a user wants to hand our
//! soft-decision channel bits off to **official DVSI silicon**
//! (USB-3000 / AMBE-3000 / AMBE-2000/2020) rather than this crate's
//! clean-room soft FEC, the chip's decoder accepts each encoded
//! channel bit as a **4-bit soft-decision (SD) value** instead of a hard
//! bit. This module converts between our native `i8` soft bits (sign =
//! hard decision, magnitude = confidence — the convention used by
//! [`crate::imbe7200::frame::decode_frame_soft`] and
//! [`crate::rate33::frame::decode_frame_soft`]) and the DVSI
//! SD format / packet.
//!
//! ## The 4-bit soft value (offset-binary)
//!
//! The chip's SD value is *not* sign-magnitude — it is a 16-level
//! offset-binary ramp from "definitely 0" to "definitely 1", with the
//! MSB equal to the hard decision and the decision threshold between 7
//! and 8:
//!
//! | SD value | Interpretation     |
//! |----------|--------------------|
//! | `0b0000` | most confident `0` |
//! | `0b0111` | least confident `0`|
//! | `0b1000` | least confident `1`|
//! | `0b1111` | most confident `1` |
//!
//! Soft-decision is a **decoder-only** feature: "Enabling the
//! soft-decision does nothing to the encoder packet." The chip does the
//! Golay/Hamming FEC internally on these soft channel bits.
//!
//! ## The AMBE-2000/2020 host packet (Table 1)
//!
//! A 20 ms soft-decision decoder packet is 60 × 16-bit words
//! (120 bytes / 960 bits):
//!
//! * word 0      — header, always [`SD_HEADER`] (`0x13EC`)
//! * words 1..12 — 12 words (192 bits) of overhead (power/control, five
//!   rate-info words, DTMF, control words — see [`SdPacketHeader`])
//! * words 12..60 — 48 data words = **192 SD slots × 4 bits**
//!   (`SD0..SD191`), four nibbles per word, `SD0` in the high nibble.
//!
//! The container is sized for the chip's maximum rate (9600 bps,
//! 192 channel bits). The **active** slot count is set by the configured
//! rate: **Rate 33** (P25/DMR/NXDN half-rate, 3600 bps with FEC) uses
//! `SD0..SD71` ([`RATE_33_CHANNEL_BITS`]); full-rate IMBE uses
//! `SD0..SD143`. Unused slots are filled with `0b0000` (most-confident-0).
//!
//! ## Validation status (DVSI reference vectors, 2026-06-03)
//!
//! Validated against DVSI's USB-3000 vectors
//! (`tv-std/tv/p25/clean_e1_sd.bit` vs the clean hard `clean.bit`):
//!
//! * **SD encoding** — the offset-binary value with MSB = hard bit is
//!   confirmed: SD-nibble hard decisions agree **98.90%** with the clean
//!   reference (the 1.1% gap = the vector's injected 1% BER). The three
//!   other nibble/bit-order conventions score ~50% (random).
//! * **Nibble order** — `SD0` is the high nibble of each byte, confirmed
//!   by the same sweep ([`pack_nibble_stream`]).
//! * **Channel-bit ordering** — sequential and identical between DVSI's
//!   soft and hard files. Our `imbe7200` bit order was already validated
//!   against these same `tv/p25` vectors, so a 144-bit soft frame maps
//!   straight to `SD0..SD143`.
//! * **Rate-control words** — taken from the reference command file
//!   `cmpp25.txt`; see [`DVSI_P25_FULLRATE_FEC`] / [`DVSI_P25_FULLRATE_NOFEC`].
//!
//! Remaining caveat: the AMBE-2000/2020 **host packet**'s five
//! `rate_info` words ([`SdPacketHeader::rate_info`]) are a *different*
//! framing from the 6-word `-r` rate-control vector and are not exercised
//! by the available USB-3000 vectors — still caller-supplied. The raw
//! [`pack_nibble_stream`] format (what the vectors use) carries no rate
//! words at all.

/// Packet header word — always `0x13EC` (Table 1, word 0).
pub const SD_HEADER: u16 = 0x13EC;

/// Total 16-bit words in a soft-decision decoder packet (60 words =
/// 120 bytes = 960 bits).
pub const SD_PACKET_WORDS: usize = 60;

/// Number of 16-bit overhead words preceding the SD data (words 0..12).
pub const SD_OVERHEAD_WORDS: usize = 12;

/// Number of 16-bit SD data words (words 12..60), four nibbles each.
pub const SD_DATA_WORDS: usize = 48;

/// Maximum soft-decision slots in one packet (`SD0..SD191`), = the
/// chip's highest-rate channel-bit count (9600 bps).
pub const SD_SLOTS: usize = 192;

/// Active channel-bit count for **Rate 33** (P25/DMR/NXDN half-rate,
/// 3600 bps with FEC): `SD0..SD71`.
pub const RATE_33_CHANNEL_BITS: usize = 72;

/// Active channel-bit count for full-rate IMBE (7200 bps): `SD0..SD143`.
pub const IMBE_FULL_RATE_CHANNEL_BITS: usize = 144;

// ---------------------------------------------------------------------------
// 4-bit soft value <-> i8 LLR
// ---------------------------------------------------------------------------

/// Convert one native soft bit (`i8`: sign = hard decision with `> 0`
/// meaning `1`, magnitude = confidence in `0..=127`) into a DVSI 4-bit
/// soft-decision value in `0..=15` (offset-binary, MSB = hard bit).
///
/// The magnitude is quantized to one of 8 confidence levels
/// (`|llr| >> 4`). A `1`-leaning bit maps to `8 + level` (least→most
/// confident 1 across `8..=15`); a `0`-leaning bit (including `llr == 0`,
/// which is hard-`0` by the `> 0` convention) maps to `7 - level`
/// (least→most confident 0 across `7..=0`).
#[inline]
pub fn llr_to_sd_nibble(llr: i8) -> u8 {
    let level = (llr.unsigned_abs() >> 4).min(7); // 0..=7
    if llr > 0 {
        8 + level
    } else {
        7 - level
    }
}

/// Inverse of [`llr_to_sd_nibble`]: map a DVSI 4-bit SD value back to a
/// representative `i8` LLR. Lossy (4-bit → 8-bit) but exactly
/// round-trips every SD value: `llr_to_sd_nibble(sd_nibble_to_llr(n)) == n`
/// for all `n in 0..=15`.
///
/// Only the low nibble of `n` is used; higher bits are ignored.
#[inline]
pub fn sd_nibble_to_llr(n: u8) -> i8 {
    let n = n & 0xF;
    if n >= 8 {
        let level = i16::from(n - 8); // 0..=7 → least..most confident 1
                                      // max(1,..) so the least-confident-1 value stays strictly > 0
                                      // (hard `1`); level*127/7 gives 0,18,..,127 — distinct >>4 buckets.
        (level * 127 / 7).max(1) as i8
    } else {
        let level = i16::from(7 - n); // 0..=7 → least..most confident 0
        -(level * 127 / 7) as i8
    }
}

// ---------------------------------------------------------------------------
// Packet
// ---------------------------------------------------------------------------

/// The 12 overhead words (words 0..12) of an AMBE-2000/2020
/// soft-decision packet, minus the fixed header.
///
/// Field names follow Table 1. **The five `rate_info` words and the
/// control words are chip-/rate-specific and caller-supplied** — see the
/// module-level "Validation boundary" note. `Default` zeroes everything
/// except via [`SdPacketHeader::new`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SdPacketHeader {
    /// Word 1, high byte — Power Control (8 bits).
    pub power_control: u8,
    /// Word 1, low byte — Control Word 1 (8 bits).
    pub control_word1: u8,
    /// Words 2..7 — Rate info 0..4. Chip-/rate-specific; not derivable
    /// from the app note. Source from DVSI's rate-control spec.
    pub rate_info: [u16; 5],
    /// Word 10 — DTMF Control.
    pub dtmf_control: u16,
    /// Word 11 — Control Word 2.
    pub control_word2: u16,
}

impl SdPacketHeader {
    /// Construct a header carrying only the five rate-info words, with
    /// all other overhead fields zero. The common case: the rate words
    /// select the codec configuration (e.g. Rate 33) and nothing else
    /// is needed for a plain decode packet.
    pub fn new(rate_info: [u16; 5]) -> Self {
        Self {
            rate_info,
            ..Self::default()
        }
    }
}

/// Errors from soft-decision packet (un)packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftDecisionError {
    /// More than [`SD_SLOTS`] (192) channel bits were supplied.
    TooManyChannelBits(usize),
    /// Packet header word was not [`SD_HEADER`] (`0x13EC`).
    BadHeader(u16),
}

impl core::fmt::Display for SoftDecisionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyChannelBits(n) => {
                write!(
                    f,
                    "{n} channel bits exceeds the {SD_SLOTS}-slot DVSI SD packet"
                )
            }
            Self::BadHeader(w) => write!(
                f,
                "bad SD packet header 0x{w:04X}, expected 0x{SD_HEADER:04X}"
            ),
        }
    }
}

impl std::error::Error for SoftDecisionError {}

/// Pack soft channel bits (in **chip encoded-packet order** — see the
/// module "Validation boundary" note) into a DVSI AMBE-2000/2020
/// soft-decision decoder packet.
///
/// `channel_llrs` holds one native `i8` per channel bit (`SD0`
/// first). Fewer than [`SD_SLOTS`] bits is normal — e.g. 72 for
/// Rate 33, 144 for full-rate IMBE; the remaining slots are filled with
/// `0b0000` (most-confident-0). Returns the 60-word packet, or
/// [`SoftDecisionError::TooManyChannelBits`] if more than 192 bits are
/// given.
pub fn pack_channel_bits(
    channel_llrs: &[i8],
    header: &SdPacketHeader,
) -> Result<[u16; SD_PACKET_WORDS], SoftDecisionError> {
    if channel_llrs.len() > SD_SLOTS {
        return Err(SoftDecisionError::TooManyChannelBits(channel_llrs.len()));
    }

    let mut words = [0u16; SD_PACKET_WORDS];
    words[0] = SD_HEADER;
    words[1] = (u16::from(header.power_control) << 8) | u16::from(header.control_word1);
    words[2..7].copy_from_slice(&header.rate_info);
    // words 7,8,9 "Unused in Input" stay 0.
    words[10] = header.dtmf_control;
    words[11] = header.control_word2;

    // 48 data words: nibbles SD[4w+0..4w+3], SD0 in the high nibble.
    // Unused slots default to nibble 0 (most-confident-0).
    let mut sd = [0u8; SD_SLOTS];
    for (slot, &llr) in channel_llrs.iter().enumerate() {
        sd[slot] = llr_to_sd_nibble(llr);
    }
    for w in 0..SD_DATA_WORDS {
        let b = 4 * w;
        words[SD_OVERHEAD_WORDS + w] = (u16::from(sd[b]) << 12)
            | (u16::from(sd[b + 1]) << 8)
            | (u16::from(sd[b + 2]) << 4)
            | u16::from(sd[b + 3]);
    }
    Ok(words)
}

/// Unpack a DVSI soft-decision packet into its header and the full
/// 192-slot SD field as native `i8` LLRs (`SD0` first). Verifies the
/// `0x13EC` header. Inverse of [`pack_channel_bits`] (the SD field
/// round-trips exactly; unused trailing slots come back as the LLR for
/// nibble 0, i.e. most-confident-0).
pub fn unpack_packet(
    words: &[u16; SD_PACKET_WORDS],
) -> Result<(SdPacketHeader, [i8; SD_SLOTS]), SoftDecisionError> {
    if words[0] != SD_HEADER {
        return Err(SoftDecisionError::BadHeader(words[0]));
    }
    let mut rate_info = [0u16; 5];
    rate_info.copy_from_slice(&words[2..7]);
    let header = SdPacketHeader {
        power_control: (words[1] >> 8) as u8,
        control_word1: (words[1] & 0xFF) as u8,
        rate_info,
        dtmf_control: words[10],
        control_word2: words[11],
    };

    let mut llrs = [0i8; SD_SLOTS];
    for w in 0..SD_DATA_WORDS {
        let word = words[SD_OVERHEAD_WORDS + w];
        let b = 4 * w;
        llrs[b] = sd_nibble_to_llr((word >> 12) as u8);
        llrs[b + 1] = sd_nibble_to_llr((word >> 8) as u8);
        llrs[b + 2] = sd_nibble_to_llr((word >> 4) as u8);
        llrs[b + 3] = sd_nibble_to_llr(word as u8);
    }
    Ok((header, llrs))
}

/// Serialize a 60-word packet to 120 big-endian bytes (the host wire
/// order: high byte of each 16-bit word first).
pub fn packet_to_bytes(words: &[u16; SD_PACKET_WORDS]) -> [u8; SD_PACKET_WORDS * 2] {
    let mut out = [0u8; SD_PACKET_WORDS * 2];
    for (i, &w) in words.iter().enumerate() {
        out[2 * i] = (w >> 8) as u8;
        out[2 * i + 1] = (w & 0xFF) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// Raw SD nibble stream (the USB-3000 `*_sd.bit` file format)
// ---------------------------------------------------------------------------
//
// DVSI's USB-3000 reference vectors store soft-decision channel bits as a
// raw stream of 4-bit SD values, **two nibbles per byte, SD0 in the high
// nibble** — no 0x13EC header and no rate words (the rate is configured
// out-of-band via the chip's `-r` rate-control vector). This is the
// directly-useful handoff format for piping soft bits to `usb3k_client`
// or comparing against the `*_sd.bit` test vectors.
//
// Validated 2026-06-03 against `tv-std/tv/p25/clean_e1_sd.bit` vs the
// clean hard `clean.bit`: with this exact convention (hard byte MSB-first,
// SD0 = high nibble, SD-value MSB = hard bit) the SD-nibble hard decisions
// agree 98.90% with the clean reference — the residual 1.1% being the
// vector's injected 1% BER. The three other nibble/bit-order conventions
// score ~50% (random), pinning the convention unambiguously.

/// Pack soft channel bits into the raw USB-3000 SD nibble stream: two
/// 4-bit SD values per byte, `SD0` in the high nibble. Length is
/// `ceil(channel_llrs.len() / 2)` bytes; an odd final bit leaves the low
/// nibble `0` (most-confident-0). No header or rate words — see the
/// module note.
pub fn pack_nibble_stream(channel_llrs: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(channel_llrs.len().div_ceil(2));
    for pair in channel_llrs.chunks(2) {
        let hi = llr_to_sd_nibble(pair[0]);
        let lo = pair.get(1).map_or(0, |&l| llr_to_sd_nibble(l));
        out.push((hi << 4) | lo);
    }
    out
}

/// Unpack a raw USB-3000 SD nibble stream (`*_sd.bit` format) into native
/// `i8` LLRs, `SD0` first (high nibble of byte 0). Inverse of
/// [`pack_nibble_stream`]; yields `2 * bytes.len()` LLRs.
pub fn unpack_nibble_stream(bytes: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(sd_nibble_to_llr((b >> 4) & 0xF));
        out.push(sd_nibble_to_llr(b & 0xF));
    }
    out
}

// ---------------------------------------------------------------------------
// Verified DVSI rate-control vectors (the chip `-r` flag)
// ---------------------------------------------------------------------------
//
// Source: DVSI USB-3000 reference command file `tv-std/tv/cmpp25.txt`
// (`-c P25 -r <six u16 words>`). These are the 6-word rate-control vector
// passed to the chip out-of-band, NOT the AMBE-2000/2020 host packet's
// 5-word `rate_info` field (a different framing). Use these when driving a
// USB-3000 / AMBE-3000-series chip for P25 full-rate.

/// P25 full-rate (7200 bps IMBE) **with FEC** — DVSI `-r` rate control.
pub const DVSI_P25_FULLRATE_FEC: [u16; 6] = [0x0558, 0x086B, 0x1030, 0x0000, 0x0000, 0x0190];

/// P25 full-rate (7200 bps IMBE) **without FEC** — DVSI `-r` rate control.
pub const DVSI_P25_FULLRATE_NOFEC: [u16; 6] = [0x0558, 0x086B, 0x0000, 0x0000, 0x0000, 0x0158];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_truth_table_anchors() {
        // The four anchor rows from the app note's table.
        assert_eq!(llr_to_sd_nibble(-127), 0b0000, "most confident 0");
        assert_eq!(
            llr_to_sd_nibble(0),
            0b0111,
            "least confident 0 (llr 0 is hard-0)"
        );
        assert_eq!(llr_to_sd_nibble(1), 0b1000, "least confident 1");
        assert_eq!(llr_to_sd_nibble(127), 0b1111, "most confident 1");
    }

    #[test]
    fn msb_is_the_hard_decision() {
        // Offset-binary: bit 3 of the SD value equals the hard bit.
        for llr in i8::MIN..=i8::MAX {
            let n = llr_to_sd_nibble(llr);
            let hard = u8::from(llr > 0);
            assert_eq!(n >> 3, hard, "llr {llr} -> sd {n:04b}");
            assert!(n <= 15);
        }
    }

    #[test]
    fn nibble_increases_with_p_bit_is_one() {
        // Monotonic ramp: stronger 1 => larger value, stronger 0 =>
        // smaller value. Walk llr from -127..127 and check the SD value
        // is non-decreasing.
        let mut prev = 0u8;
        for llr in -127i8..=127 {
            let n = llr_to_sd_nibble(llr);
            assert!(n >= prev, "non-monotonic at llr {llr}: {n} < {prev}");
            prev = n;
        }
    }

    #[test]
    fn nibble_llr_roundtrips_every_value() {
        for n in 0u8..=15 {
            let llr = sd_nibble_to_llr(n);
            assert_eq!(
                llr_to_sd_nibble(llr),
                n,
                "sd {n} -> llr {llr} -> sd mismatch"
            );
        }
    }

    #[test]
    fn header_word_is_0x13ec() {
        let pkt = pack_channel_bits(&[], &SdPacketHeader::default()).unwrap();
        assert_eq!(pkt[0], SD_HEADER);
        assert_eq!(pkt[0], 0x13EC);
    }

    #[test]
    fn rate_info_lands_in_words_2_through_6() {
        let header = SdPacketHeader::new([0x1111, 0x2222, 0x3333, 0x4444, 0x5555]);
        let pkt = pack_channel_bits(&[], &header).unwrap();
        assert_eq!(&pkt[2..7], &[0x1111, 0x2222, 0x3333, 0x4444, 0x5555]);
    }

    #[test]
    fn sd0_is_high_nibble_of_first_data_word() {
        // SD0 = most-confident-1 (0xF), SD1..3 = most-confident-0 (0x0).
        let llrs = [127i8, -127, -127, -127];
        let pkt = pack_channel_bits(&llrs, &SdPacketHeader::default()).unwrap();
        assert_eq!(
            pkt[SD_OVERHEAD_WORDS], 0xF000,
            "SD0 must be the high nibble"
        );
    }

    #[test]
    fn rate_33_uses_first_72_slots_rest_filled_zero() {
        // 72 strong-1 bits; slots 72..192 must be most-confident-0 (0x0).
        let llrs = [100i8; RATE_33_CHANNEL_BITS];
        let pkt = pack_channel_bits(&llrs, &SdPacketHeader::default()).unwrap();
        let (_, sd) = unpack_packet(&pkt).unwrap();
        for (i, &s) in sd.iter().enumerate() {
            if i < RATE_33_CHANNEL_BITS {
                assert!(s > 0, "active slot {i} should be hard-1");
            } else {
                assert_eq!(
                    llr_to_sd_nibble(s),
                    0,
                    "unused slot {i} must be most-conf-0"
                );
            }
        }
    }

    #[test]
    fn pack_unpack_roundtrip_preserves_sd_field_and_header() {
        let header = SdPacketHeader {
            power_control: 0xAB,
            control_word1: 0xCD,
            rate_info: [0x0558, 0x086B, 0x1030, 0x0000, 0x0190],
            dtmf_control: 0x1234,
            control_word2: 0x5678,
        };
        // A varied 144-bit (full-rate IMBE) soft vector.
        let mut llrs = [0i8; IMBE_FULL_RATE_CHANNEL_BITS];
        let mut state = 0xC0FFEEu32;
        for s in llrs.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *s = (state >> 24) as i8;
        }
        let pkt = pack_channel_bits(&llrs, &header).unwrap();
        let (h2, sd) = unpack_packet(&pkt).unwrap();
        assert_eq!(h2, header, "header must round-trip");
        // SD nibbles round-trip exactly, so re-quantizing the recovered
        // LLRs must reproduce the original nibbles.
        for (i, &orig) in llrs.iter().enumerate() {
            assert_eq!(
                llr_to_sd_nibble(sd[i]),
                llr_to_sd_nibble(orig),
                "slot {i} SD nibble changed across round-trip"
            );
        }
    }

    #[test]
    fn too_many_channel_bits_errors() {
        let llrs = [0i8; SD_SLOTS + 1];
        assert_eq!(
            pack_channel_bits(&llrs, &SdPacketHeader::default()),
            Err(SoftDecisionError::TooManyChannelBits(SD_SLOTS + 1))
        );
    }

    #[test]
    fn bad_header_rejected_on_unpack() {
        let mut pkt = pack_channel_bits(&[], &SdPacketHeader::default()).unwrap();
        pkt[0] = 0x0000;
        assert_eq!(
            unpack_packet(&pkt),
            Err(SoftDecisionError::BadHeader(0x0000))
        );
    }

    #[test]
    fn packet_is_960_bits_and_header_bytes_are_0x13_0xec() {
        let pkt = pack_channel_bits(&[], &SdPacketHeader::default()).unwrap();
        let bytes = packet_to_bytes(&pkt);
        assert_eq!(bytes.len(), 120, "120 bytes = 960 bits");
        assert_eq!(bytes[0], 0x13);
        assert_eq!(bytes[1], 0xEC);
    }

    #[test]
    fn nibble_stream_sd0_is_high_nibble() {
        // Validated DVSI convention: SD0 in the high nibble of byte 0.
        let bytes = pack_nibble_stream(&[127i8, -127]); // most-conf-1, most-conf-0
        assert_eq!(bytes, vec![0xF0]);
    }

    #[test]
    fn nibble_stream_roundtrips() {
        let mut llrs = [0i8; IMBE_FULL_RATE_CHANNEL_BITS];
        let mut state = 0xBADC0DEu32;
        for s in llrs.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *s = (state >> 24) as i8;
        }
        let bytes = pack_nibble_stream(&llrs);
        assert_eq!(
            bytes.len(),
            IMBE_FULL_RATE_CHANNEL_BITS / 2,
            "2 nibbles/byte"
        );
        let back = unpack_nibble_stream(&bytes);
        for (i, &orig) in llrs.iter().enumerate() {
            assert_eq!(
                llr_to_sd_nibble(back[i]),
                llr_to_sd_nibble(orig),
                "slot {i} nibble changed across stream round-trip"
            );
        }
    }

    #[test]
    fn nibble_stream_odd_length_pads_low_nibble_zero() {
        let bytes = pack_nibble_stream(&[127i8]); // one bit -> hi nibble F, lo 0
        assert_eq!(bytes, vec![0xF0]);
    }

    #[test]
    fn nibble_stream_msb_is_hard_bit() {
        // The high bit of each SD nibble must equal the hard decision —
        // the property that scored 98.9% against the DVSI clean vector.
        let llrs: Vec<i8> = (-120..120).step_by(7).collect();
        let bytes = pack_nibble_stream(&llrs);
        let recovered = unpack_nibble_stream(&bytes);
        for (i, &llr) in llrs.iter().enumerate() {
            let hard = llr > 0;
            assert_eq!(recovered[i] > 0, hard, "slot {i} hard bit flipped");
        }
    }

    #[test]
    fn verified_rate_control_words_match_cmpp25() {
        assert_eq!(
            DVSI_P25_FULLRATE_FEC,
            [0x0558, 0x086B, 0x1030, 0x0000, 0x0000, 0x0190]
        );
        assert_eq!(
            DVSI_P25_FULLRATE_NOFEC,
            [0x0558, 0x086B, 0x0000, 0x0000, 0x0000, 0x0158]
        );
    }
}
