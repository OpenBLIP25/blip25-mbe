#!/usr/bin/env python3
"""DVSI USB-3000 / AMBE-3000R driver for blip25-mbe synthesis debugging.

This is the working Python driver derived from the AMBE-3000R Vocoder Chip
User's Manual (Version 1.4, March 2013) and empirically validated against
hardware on `pve` (Proxmox host, /dev/ttyUSB0, 460800 baud).

The chip is an AMBE-3000R (Rate Converter variant — version
`0AMBE3000R V120.E100.XXXX.C106.G514.R009.B0010411.C0020208`). It supports
P25 full-rate IMBE despite the "RC" name.

## Critical safety rule

NEVER send PKT_RESET (0x33). It hangs the chip until physical USB
unplug/replug. The Proxmox host's xhci_hcd root hub does NOT support
per-port power switching, so software cannot recover. Use PKT_INIT
(0x0B) for soft initialization instead — it returns a response so we
know it succeeded and doesn't hang the chip.

## Packet protocol (per manual §4.2 / §6.2)

    [0x61] [len_hi] [len_lo] [type] [...fields...] [0x2F] [parity]

- Length = bytes between length field and parity_byte (includes type
  and parity_marker).
- Parity = XOR of all bytes after start byte (excluding the parity
  byte itself).

## Channel packet for P25 full-rate IMBE decode

Per manual Table 33 + Table 107:
- PKT_CHANNEL0 (0x40): 1 byte total — selects vocoder channel 0.
- CHAND (0x01): variable — field_id + 1-byte bit count (40..192) + data
  packed 8 bits per byte. Chand[0] = most error-sensitive bits.

For P25 full-rate FEC (144 bits, 18 bytes):

    type=0x01 | 0x40 | 0x01 | 0x90 | <18 bytes>

## Speech packet (both directions, same wire shape)

Decode direction — chip emits:

    type=0x02 | 0x40 (CHANNEL0) | 0x00 (SPEECHD field id) | 0xA0 (160 samples) | <320 bytes PCM>

Encode direction — host sends:

    type=0x02 | 0x40 (CHANNEL0) | 0x00 (SPEECHD field id) | 0xA0 (160 samples) | <320 bytes PCM>

PCM is big-endian i16 per BABA-A convention (confirmed by DVSI's own
`usb3k_client.c` — `put_word` emits hi byte first).

## Init flags (PKT_INIT data byte)

From DVSI `a3kpacket.h`:

- PKT_INIT_ENCODER = 0x01
- PKT_INIT_DECODER = 0x02
- PKT_INIT_ECHO    = 0x04

Decode-only probes use 0x02. Encode probes need 0x01. Roundtrip probes
(encode-then-decode) need both → 0x03.

## P25 rate configuration (PKT_RATEP, field id 0x0A)

From cmpp25.txt + USB-3000 manual Table 8:
- Full-rate FEC (7200 bps): RCWs = 0x0558 0x086B 0x1030 0x0000 0x0000 0x0190
- Full-rate no-FEC (4400 bps): RCWs = 0x0558 0x086B 0x0000 0x0000 0x0000 0x0158
"""
import argparse
import math
import os
import struct
import sys
import time

import serial

# Set DVSI_DEBUG=1 in the env to log per-frame timing + parity audit
# from read_response. Useful when rerunning on the chip after changing
# the framing logic — confirms the new code reads exactly one packet's
# worth of bytes per frame instead of stalling on a phantom byte read.
DEBUG = bool(int(os.environ.get("DVSI_DEBUG", "0") or "0"))

# ----------------------------------------------------------------------------
# Protocol constants
# ----------------------------------------------------------------------------

PKT_HEADER       = 0x61
PKT_PARITYBYTE   = 0x2F

# Packet types
PKT_TYPE_CONTROL = 0x00
PKT_TYPE_CHANNEL = 0x01
PKT_TYPE_SPEECH  = 0x02

# Field identifiers
FIELD_SPEECHD    = 0x00
FIELD_CHAND      = 0x01
FIELD_CHAND4     = 0x17
FIELD_RATET      = 0x09
FIELD_RATEP      = 0x0A
FIELD_INIT       = 0x0B
FIELD_PRODID     = 0x30
FIELD_VERSTRING  = 0x31
FIELD_CHANNEL0   = 0x40

# PKT_INIT data-byte flags (DVSI a3kpacket.h)
INIT_ENCODER     = 0x01
INIT_DECODER     = 0x02
INIT_ECHO        = 0x04

# WARNING: do not send these
FIELD_RESET      = 0x33  # hangs chip — never use

# P25 rate config words (USB-3000 manual Table 8)
P25_FULLRATE_FEC_RCWS   = [0x0558, 0x086B, 0x1030, 0x0000, 0x0000, 0x0190]
P25_FULLRATE_NOFEC_RCWS = [0x0558, 0x086B, 0x0000, 0x0000, 0x0000, 0x0158]

# P25 half-rate AMBE+2 rate-table indices (Columbia DVSI api.h, AMBE-3003 §11)
P25_HALFRATE_FEC_INDEX    = 33  # 3600 / 2450 / 1150, 9 bytes = 72 bits per frame
P25_HALFRATE_NOFEC_INDEX  = 34  # 2450 / 2450 / 0, info-only path

# ----------------------------------------------------------------------------
# Packet construction / parsing
# ----------------------------------------------------------------------------

def parity(payload):
    p = 0
    for b in payload:
        p ^= b
    return p

def pack_packet(pkt_type, fields):
    """Build a complete packet including header, length, type, parity_marker, parity."""
    length = 1 + len(fields) + 1  # type(1) + fields + parity_marker(1)
    body = bytes([(length >> 8) & 0xFF, length & 0xFF, pkt_type]) + bytes(fields) + bytes([PKT_PARITYBYTE])
    return bytes([PKT_HEADER]) + body + bytes([parity(body)])

def read_response(s, timeout_s=2.0):
    """Read one complete response packet from the chip. Returns (type, body) or None.

    The chip's response wire format is **asymmetric** with `pack_packet`'s
    request format:

        [HEADER 0x61] [LEN_HI] [LEN_LO] [TYPE] [...fields...]

    `length` counts only the fields — no marker, no parity byte. Total
    wire bytes per response = `4 + length`. Empirically verified on the
    AMBE-3000R: a SPEECH response with 320 PCM bytes reports
    `length = 322` (= 2-byte SPEECHD/n_samples header + 320 PCM bytes),
    and the chip sends exactly 326 wire bytes — no trailing parity.

    The original driver's `s.read(1)` after the body was the source of
    the ~2 s/frame stall: it blocked the full timeout waiting for a
    phantom parity byte the chip never sends. Fix here: read exactly
    `plen` body bytes and stop. The `pkt_type` returned is the type
    byte from the header; `body` is the field bytes only.
    """
    s.timeout = timeout_s
    t0 = time.monotonic() if DEBUG else 0.0
    hdr = s.read(4)
    if len(hdr) < 4:
        if DEBUG:
            sys.stderr.write(
                f"[dvsi] read_response: header timeout ({len(hdr)}/4 bytes in {time.monotonic()-t0:.2f}s)\n"
            )
        return None
    if hdr[0] != PKT_HEADER:
        if DEBUG:
            sys.stderr.write(
                f"[dvsi] read_response: bad header byte 0x{hdr[0]:02x} (expected 0x{PKT_HEADER:02x}); framing desync\n"
            )
        return None
    plen = (hdr[1] << 8) | hdr[2]
    pkt_type = hdr[3]
    # Chip response format: `length` = len(fields). No trailing marker
    # or parity byte over the wire.
    body = b''
    while len(body) < plen:
        chunk = s.read(plen - len(body))
        if not chunk:
            if DEBUG:
                sys.stderr.write(
                    f"[dvsi] read_response: body timeout ({len(body)}/{plen} bytes)\n"
                )
            return None
        body += chunk
    if DEBUG:
        elapsed = time.monotonic() - t0
        sys.stderr.write(
            f"[dvsi] read_response: type=0x{pkt_type:02x} fields={plen}B took {elapsed:.3f}s\n"
        )
    return (pkt_type, body)

def make_channel_packet(frame_bytes, n_bits=144, channel=0):
    """Build a P25 channel packet for the chip's decoder.

    Format (per manual §6.9.1, Table 33 + Table 107):
        type=PKT_CHANNEL | CHANNEL0(0x40) | CHAND(0x01) | n_bits_1byte | data
    """
    if channel != 0:
        raise NotImplementedError("only channel 0 supported (CHANNEL0 field is 1-byte, no data)")
    fields = [FIELD_CHANNEL0, FIELD_CHAND, n_bits & 0xFF] + list(frame_bytes)
    return pack_packet(PKT_TYPE_CHANNEL, fields)

def make_speech_packet(samples, channel=0):
    """Build a speech packet to feed the chip's encoder.

    Format mirrors the decode response (per DVSI usb3k_client.c
    `encode_speech_packet`):
        type=PKT_SPEECH | CHANNEL0(0x40) | SPEECHD(0x00) | n_samples | <samples × BE i16>

    `samples` is a list of i16 PCM values in [-32768, 32767]. Length is
    typically 160 for P25 full-rate (20 ms @ 8 kHz).
    """
    if channel != 0:
        raise NotImplementedError("only channel 0 supported")
    n = len(samples)
    if n > 0xFF:
        raise ValueError(f"sample count {n} exceeds 1-byte field width")
    fields = [FIELD_CHANNEL0, FIELD_SPEECHD, n & 0xFF]
    for v in samples:
        v = int(v)
        if v < -32768 or v > 32767:
            raise ValueError(f"PCM sample {v} out of i16 range")
        if v < 0:
            v += 0x10000
        fields.append((v >> 8) & 0xFF)  # big-endian per put_word
        fields.append(v & 0xFF)
    return pack_packet(PKT_TYPE_SPEECH, fields)

def extract_channel_bits(channel_body):
    """Extract `(n_bits, frame_bytes)` from a channel packet body.

    Body layout (per manual §6.9.1, mirror of make_channel_packet):
        [0x40 (CHANNEL0)] [0x01 (CHAND)] [n_bits_1byte] [ceil(n_bits/8) bytes]
        OR (when chip omits CHANNEL0 field):
        [0x01 (CHAND)] [n_bits_1byte] [ceil(n_bits/8) bytes]
    """
    if len(channel_body) >= 3 and channel_body[0] == FIELD_CHAND:
        n_bits = channel_body[1]
        data = channel_body[2:]
    elif len(channel_body) >= 4 and channel_body[0] == FIELD_CHANNEL0 and channel_body[1] == FIELD_CHAND:
        n_bits = channel_body[2]
        data = channel_body[3:]
    else:
        return None
    n_bytes = (n_bits + 7) // 8
    if len(data) < n_bytes:
        return None
    return (n_bits, bytes(data[:n_bytes]))

def extract_pcm(speech_body):
    """Extract 160 i16 samples from a speech packet body.

    Body layout (per manual §6.11.1, response shape verified empirically):
        [0x40 (CHANNEL0)] [0x00 (SPEECHD)] [0xA0 (160)] [320 bytes PCM big-endian i16]
        OR (when chip omits CHANNEL0 field):
        [0x00 (SPEECHD)] [0xA0 (160)] [320 bytes PCM big-endian i16]
    """
    if len(speech_body) >= 322 and speech_body[0] == 0x00 and speech_body[1] == 0xA0:
        pcm_bytes = speech_body[2:322]
    elif len(speech_body) >= 323 and speech_body[0] == 0x40 and speech_body[1] == 0x00 and speech_body[2] == 0xA0:
        pcm_bytes = speech_body[3:323]
    else:
        return None
    samples = []
    for i in range(160):
        hi = pcm_bytes[2*i]
        lo = pcm_bytes[2*i+1]
        v = (hi << 8) | lo
        if v >= 0x8000:
            v -= 0x10000
        samples.append(v)
    return samples

# ----------------------------------------------------------------------------
# High-level chip operations
# ----------------------------------------------------------------------------

def get_version(s):
    """Query chip version string."""
    pkt = pack_packet(PKT_TYPE_CONTROL, [FIELD_PRODID, FIELD_VERSTRING])
    s.write(pkt); s.flush()
    resp = read_response(s)
    if resp is None:
        return None
    pkt_type, body = resp
    return body.decode('ascii', errors='replace').rstrip('\x00')

def init_codec(s, flags=INIT_DECODER):
    """PKT_INIT — initialize the requested codec halves.

    `flags` is a bitwise OR of INIT_ENCODER / INIT_DECODER / INIT_ECHO.
    Decode-only: 0x02. Encode-only: 0x01. Roundtrip: 0x03. Safe (returns
    a response), unlike PKT_RESET (0x33) which permanently hangs the chip.
    """
    pkt = pack_packet(PKT_TYPE_CONTROL, [FIELD_INIT, flags & 0xFF])
    s.write(pkt); s.flush()
    return read_response(s)

def init_decoder(s):
    """Back-compat alias: initialize decoder only."""
    return init_codec(s, INIT_DECODER)

def init_encoder(s):
    """Initialize encoder only."""
    return init_codec(s, INIT_ENCODER)

def init_both(s):
    """Initialize encoder and decoder for roundtrip probes."""
    return init_codec(s, INIT_ENCODER | INIT_DECODER)

def set_p25_fullrate(s, with_fec=True):
    """Configure chip for P25 full-rate IMBE decode/encode."""
    rcws = P25_FULLRATE_FEC_RCWS if with_fec else P25_FULLRATE_NOFEC_RCWS
    fields = [FIELD_RATEP]
    for w in rcws:
        fields.append((w >> 8) & 0xFF)
        fields.append(w & 0xFF)
    pkt = pack_packet(PKT_TYPE_CONTROL, fields)
    s.write(pkt); s.flush()
    return read_response(s)

def set_p25_halfrate(s, with_fec=True):
    """Configure chip for P25 half-rate AMBE+2 decode/encode via PKT_RATET.

    Uses the AMBE-3000 rate-table indices (33 / 34) instead of explicit
    rate-config words — the by-index path is simpler for half-rate and
    matches the Columbia DVSI API documentation
    (`AMBE-3003 ratet(channel, 33)` for FEC, 34 for no-FEC).

    Channel packets after this call carry 72-bit (9-byte) FEC frames
    when `with_fec=True`, or 49-bit (7-byte) info-only payloads when
    `with_fec=False`. The driver's `decode_halfrate_frame` /
    `encode_halfrate_frame` helpers default to the FEC variant.
    """
    idx = P25_HALFRATE_FEC_INDEX if with_fec else P25_HALFRATE_NOFEC_INDEX
    fields = [FIELD_RATET, idx & 0xFF]
    pkt = pack_packet(PKT_TYPE_CONTROL, fields)
    s.write(pkt); s.flush()
    return read_response(s)

def decode_halfrate_frame(s, frame_bytes, with_fec=True):
    """Send one P25 half-rate AMBE+2 frame, return 160 PCM samples (i16 list).

    `frame_bytes` is 9 bytes (72 bits) with FEC, or 7 bytes (49 bits)
    info-only. The chip must already be in the matching
    `set_p25_halfrate(with_fec=...)` mode.
    """
    if with_fec:
        if len(frame_bytes) != 9:
            raise ValueError(f"expected 9 bytes, got {len(frame_bytes)}")
        n_bits = 72
    else:
        if len(frame_bytes) != 7:
            raise ValueError(f"expected 7 bytes, got {len(frame_bytes)}")
        n_bits = 49
    pkt = make_channel_packet(frame_bytes, n_bits=n_bits)
    s.write(pkt); s.flush()
    resp = read_response(s, timeout_s=2.0)
    if resp is None:
        return None
    pkt_type, body = resp
    if pkt_type != PKT_TYPE_SPEECH:
        return None
    return extract_pcm(body)

def encode_halfrate_frame(s, pcm_160_samples, with_fec=True):
    """Send 160 PCM samples, return half-rate frame bytes.

    Chip must be in half-rate mode (set_p25_halfrate). Returns 9 bytes
    (FEC) or 7 bytes (info-only).
    """
    if len(pcm_160_samples) != 160:
        raise ValueError(f"expected 160 samples, got {len(pcm_160_samples)}")
    expected_n_bits = 72 if with_fec else 49
    expected_n_bytes = (expected_n_bits + 7) // 8
    pkt = make_speech_packet(pcm_160_samples)
    s.write(pkt); s.flush()
    resp = read_response(s, timeout_s=2.0)
    if resp is None:
        return None
    pkt_type, body = resp
    if pkt_type != PKT_TYPE_CHANNEL:
        return None
    parsed = extract_channel_bits(body)
    if parsed is None:
        return None
    n_bits, data = parsed
    if n_bits != expected_n_bits or len(data) != expected_n_bytes:
        return None
    return data

def decode_halfrate_file(s, ambe9_path, out_path, n_frames=None, with_fec=True):
    """Decode an .ambe9 file (concatenated 9-byte AMBE+2 frames) to PCM."""
    fb = 9 if with_fec else 7
    with open(ambe9_path, 'rb') as f:
        fec = f.read()
    total = len(fec) // fb
    if n_frames is None:
        n_frames = total
    n_frames = min(n_frames, total)
    pcm_out = bytearray()
    errors = 0
    for f_idx in range(n_frames):
        frame = fec[f_idx*fb:(f_idx+1)*fb]
        samples = decode_halfrate_frame(s, frame, with_fec=with_fec)
        if samples is None:
            errors += 1
            pcm_out += bytes(320)
            continue
        for v in samples:
            pcm_out += struct.pack('<h', v)
    with open(out_path, 'wb') as f:
        f.write(pcm_out)
    return n_frames, errors

def decode_frame(s, frame_18_bytes):
    """Send one P25 full-rate FEC frame, return 160 PCM samples (i16 list)."""
    if len(frame_18_bytes) != 18:
        raise ValueError(f"expected 18 bytes, got {len(frame_18_bytes)}")
    pkt = make_channel_packet(frame_18_bytes, n_bits=144)
    s.write(pkt); s.flush()
    resp = read_response(s, timeout_s=2.0)
    if resp is None:
        return None
    pkt_type, body = resp
    if pkt_type != PKT_TYPE_SPEECH:
        return None
    return extract_pcm(body)

def encode_frame(s, pcm_160_samples):
    """Send 160 PCM samples to the chip's encoder, return 18 frame bytes.

    For P25 full-rate FEC the response carries 144 bits / 18 bytes. The
    chip is in `set_p25_fullrate` mode; the host is responsible for
    setting the rate before calling this.
    """
    if len(pcm_160_samples) != 160:
        raise ValueError(f"expected 160 samples, got {len(pcm_160_samples)}")
    pkt = make_speech_packet(pcm_160_samples)
    s.write(pkt); s.flush()
    resp = read_response(s, timeout_s=2.0)
    if resp is None:
        return None
    pkt_type, body = resp
    if pkt_type != PKT_TYPE_CHANNEL:
        return None
    parsed = extract_channel_bits(body)
    if parsed is None:
        return None
    n_bits, data = parsed
    if n_bits != 144 or len(data) != 18:
        return None
    return data

def encode_file(s, pcm_path, bit_out, n_frames=None):
    """Encode a raw PCM file (LE i16, 8 kHz mono) to a P25 full-rate .bit file."""
    with open(pcm_path, 'rb') as f:
        pcm = f.read()
    samples = struct.unpack(f'<{len(pcm)//2}h', pcm[:(len(pcm)//2)*2])
    total = len(samples) // 160
    if n_frames is None:
        n_frames = total
    n_frames = min(n_frames, total)
    bit_buf = bytearray()
    errors = 0
    for f_idx in range(n_frames):
        frame_pcm = list(samples[f_idx*160:(f_idx+1)*160])
        bits = encode_frame(s, frame_pcm)
        if bits is None:
            errors += 1
            bit_buf += bytes(18)
            continue
        bit_buf += bits
    with open(bit_out, 'wb') as f:
        f.write(bit_buf)
    return n_frames, errors

def decode_file(s, fec_path, out_path, n_frames=None):
    """Decode an entire .bit file (P25 full-rate FEC) to PCM."""
    with open(fec_path, 'rb') as f:
        fec = f.read()
    total = len(fec) // 18
    if n_frames is None:
        n_frames = total
    n_frames = min(n_frames, total)
    pcm_out = bytearray()
    errors = 0
    for f_idx in range(n_frames):
        frame = fec[f_idx*18:(f_idx+1)*18]
        samples = decode_frame(s, frame)
        if samples is None:
            errors += 1
            pcm_out += bytes(320)
            continue
        for v in samples:
            pcm_out += struct.pack('<h', v)
    with open(out_path, 'wb') as f:
        f.write(pcm_out)
    return n_frames, errors

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=460800)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('version')

    p_decode = sub.add_parser('decode', help='Decode P25 full-rate .bit file via chip')
    p_decode.add_argument('bit_path')
    p_decode.add_argument('pcm_out')
    p_decode.add_argument('--frames', type=int, default=None)
    p_decode.add_argument('--no-fec', action='store_true')

    p_compare = sub.add_parser('compare', help='Decode and compare against a reference .pcm')
    p_compare.add_argument('bit_path')
    p_compare.add_argument('ref_pcm')
    p_compare.add_argument('--frames', type=int, default=60)
    p_compare.add_argument('--no-fec', action='store_true')

    p_encode = sub.add_parser('encode', help='Encode a raw PCM file (LE i16 8 kHz mono) to a P25 full-rate .bit file')
    p_encode.add_argument('pcm_path')
    p_encode.add_argument('bit_out')
    p_encode.add_argument('--frames', type=int, default=None)
    p_encode.add_argument('--no-fec', action='store_true')

    p_round = sub.add_parser('roundtrip', help='Encode PCM via chip, then decode the chip bits back to PCM via chip')
    p_round.add_argument('pcm_in')
    p_round.add_argument('pcm_out')
    p_round.add_argument('--frames', type=int, default=None)
    p_round.add_argument('--no-fec', action='store_true')
    p_round.add_argument('--bit-out', default=None, help='Optional path to also save the intermediate .bit stream')

    p_hd = sub.add_parser('decode-halfrate', help='Decode a P25 half-rate AMBE+2 .ambe9 file (9-byte frames) via chip')
    p_hd.add_argument('ambe9_path')
    p_hd.add_argument('pcm_out')
    p_hd.add_argument('--frames', type=int, default=None)
    p_hd.add_argument('--no-fec', action='store_true', help='Treat input as 7-byte info-only frames')

    args = ap.parse_args()
    with serial.Serial(args.port, args.baud, timeout=2.0) as s:
        s.reset_input_buffer()

        if args.cmd == 'version':
            v = get_version(s)
            print(v if v else "(no response)")
            return

        # Encode-capable commands need both halves initialized.
        init_flags = INIT_DECODER
        if args.cmd in ('encode', 'roundtrip'):
            init_flags = INIT_ENCODER | INIT_DECODER
        init_codec(s, init_flags)
        # Rate config: half-rate commands need PKT_RATET 33/34, all
        # others stay on the full-rate PKT_RATEP path.
        if args.cmd in ('decode-halfrate',):
            set_p25_halfrate(s, with_fec=not args.no_fec)
        else:
            set_p25_fullrate(s, with_fec=not args.no_fec)
        time.sleep(0.1)

        if args.cmd == 'decode-halfrate':
            n, errors = decode_halfrate_file(
                s, args.ambe9_path, args.pcm_out, args.frames,
                with_fec=not args.no_fec,
            )
            print(f"decoded {n - errors}/{n} half-rate frames to {args.pcm_out}")
            if errors:
                print(f"errors: {errors}")
            return

        if args.cmd == 'decode':
            n, errors = decode_file(s, args.bit_path, args.pcm_out, args.frames)
            print(f"decoded {n - errors}/{n} frames to {args.pcm_out}")
            if errors:
                print(f"errors: {errors}")
            return

        if args.cmd == 'encode':
            n, errors = encode_file(s, args.pcm_path, args.bit_out, args.frames)
            print(f"encoded {n - errors}/{n} frames to {args.bit_out}")
            if errors:
                print(f"errors: {errors}")
            return

        if args.cmd == 'roundtrip':
            bit_path = args.bit_out if args.bit_out else '/tmp/_chip_round.bit'
            n_e, err_e = encode_file(s, args.pcm_in, bit_path, args.frames)
            n_d, err_d = decode_file(s, bit_path, args.pcm_out, n_e)
            print(f"encoded {n_e - err_e}/{n_e}, decoded {n_d - err_d}/{n_d} → {args.pcm_out}")
            if err_e or err_d:
                print(f"encode errors: {err_e}; decode errors: {err_d}")
            return

        if args.cmd == 'compare':
            n, errors = decode_file(s, args.bit_path, '/tmp/_chip_out.pcm', args.frames)
            with open('/tmp/_chip_out.pcm', 'rb') as f:
                our = f.read()
            with open(args.ref_pcm, 'rb') as f:
                ref = f.read(args.frames * 320)
            n_samp = min(len(our), len(ref)) // 2
            our_s = struct.unpack(f'<{n_samp}h', our[:n_samp*2])
            ref_s = struct.unpack(f'<{n_samp}h', ref[:n_samp*2])
            sse = sum((a-b)**2 for a, b in zip(our_s, ref_s))
            sref = sum(b*b for b in ref_s)
            be = sum(1 for a, b in zip(our_s, ref_s) if a == b)
            snr = 10*math.log10(sref/sse) if (sse > 0 and sref > 0) else 0.0
            print(f"frames decoded: {n - errors}/{n}")
            print(f"bit-exact:      {be}/{n_samp} ({100*be/n_samp:.2f}%)")
            print(f"SNR vs ref:     {snr:+.2f} dB")
            return

if __name__ == '__main__':
    main()
