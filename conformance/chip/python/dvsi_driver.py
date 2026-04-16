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

## Speech packet response

Per manual §6.11.1:

    type=0x02 | 0x40 (CHANNEL0) | 0x00 (SPEECHD field id) | 0xA0 (160 samples) | <320 bytes PCM>

PCM is big-endian i16 per BABA-A convention.

## P25 rate configuration (PKT_RATEP, field id 0x0A)

From cmpp25.txt + USB-3000 manual Table 8:
- Full-rate FEC (7200 bps): RCWs = 0x0558 0x086B 0x1030 0x0000 0x0000 0x0190
- Full-rate no-FEC (4400 bps): RCWs = 0x0558 0x086B 0x0000 0x0000 0x0000 0x0158
"""
import argparse
import math
import struct
import sys
import time

import serial

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

# WARNING: do not send these
FIELD_RESET      = 0x33  # hangs chip — never use

# P25 rate config words (USB-3000 manual Table 8)
P25_FULLRATE_FEC_RCWS   = [0x0558, 0x086B, 0x1030, 0x0000, 0x0000, 0x0190]
P25_FULLRATE_NOFEC_RCWS = [0x0558, 0x086B, 0x0000, 0x0000, 0x0000, 0x0158]

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
    """Read one complete response packet from the chip. Returns (type, body) or None."""
    s.timeout = timeout_s
    hdr = s.read(4)
    if len(hdr) < 4:
        return None
    if hdr[0] != PKT_HEADER:
        return None
    plen = (hdr[1] << 8) | hdr[2]
    pkt_type = hdr[3]
    body = b''
    while len(body) < plen:
        chunk = s.read(plen - len(body))
        if not chunk:
            return None
        body += chunk
    # Discard the trailing parity byte
    s.read(1)
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

def init_decoder(s):
    """PKT_INIT with data=0x02 — initialize decoder. Safe (returns response)."""
    pkt = pack_packet(PKT_TYPE_CONTROL, [FIELD_INIT, 0x02])
    s.write(pkt); s.flush()
    return read_response(s)

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

    args = ap.parse_args()
    with serial.Serial(args.port, args.baud, timeout=2.0) as s:
        s.reset_input_buffer()

        if args.cmd == 'version':
            v = get_version(s)
            print(v if v else "(no response)")
            return

        # All other commands need decoder init + rate set
        init_decoder(s)
        set_p25_fullrate(s, with_fec=not args.no_fec)
        time.sleep(0.1)

        if args.cmd == 'decode':
            n, errors = decode_file(s, args.bit_path, args.pcm_out, args.frames)
            print(f"decoded {n - errors}/{n} frames to {args.pcm_out}")
            if errors:
                print(f"errors: {errors}")
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
