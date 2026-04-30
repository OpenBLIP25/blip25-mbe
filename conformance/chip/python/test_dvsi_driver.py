#!/usr/bin/env python3
"""Byte-accounting tests for `dvsi_driver.py`'s `read_response`.

Runs against an in-memory mock serial — no chip required. Validates
the chip's **response** wire format (asymmetric with the host-side
`pack_packet` request format): chip responses are
`[HEADER][LEN_HI][LEN_LO][TYPE][...fields...]` with `length = len(fields)`
and no trailing marker or parity byte.

Empirically verified on AMBE-3000R: a SPEECH response with 320 PCM
bytes reports `length = 322` (2-byte SPEECHD/n_samples header + 320
PCM) and chip sends exactly 326 wire bytes total. The pre-fix driver
issued an extra `s.read(1)` after the body that blocked for the full
2-second serial timeout waiting for a phantom byte — dropping
throughput from ~50 fps to ~0.5 fps. The
`back_to_back_chip_responses_stay_in_frame` test guards against any
future regression that would consume more or fewer bytes than the
chip actually sends.

Run:

    python3 conformance/chip/python/test_dvsi_driver.py
"""
from __future__ import annotations

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dvsi_driver as d


class MockSerial:
    """Minimal pyserial-shaped object backed by a BytesIO stream.

    Only the surface `dvsi_driver.read_response` touches: `read(n)`,
    settable `timeout`. Reads return up to `n` bytes from the buffer;
    if fewer are available, returns whatever is there (mirrors
    pyserial's behavior on timeout).
    """

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)
        self.timeout = 0.0

    def read(self, n: int) -> bytes:
        return self._buf.read(n)

    def remaining(self) -> int:
        cur = self._buf.tell()
        end = self._buf.seek(0, os.SEEK_END)
        self._buf.seek(cur)
        return end - cur


def chip_response(pkt_type: int, fields: bytes) -> bytes:
    """Build a chip-shaped response packet for use as a mock-serial input.

    Mirrors the AMBE-3000R wire format observed empirically: header,
    big-endian 16-bit `length = len(fields)`, type byte, then the field
    bytes. No trailing marker or parity — that's the asymmetry with
    `pack_packet`'s host→chip request format.
    """
    plen = len(fields)
    return bytes([d.PKT_HEADER, (plen >> 8) & 0xFF, plen & 0xFF, pkt_type]) + bytes(fields)


class ChipResponseParsing(unittest.TestCase):
    def test_control_response_parsed(self):
        # 2-byte INIT ack — chip's actual minimum response shape.
        wire = chip_response(d.PKT_TYPE_CONTROL, bytes([0x0B, 0x02]))
        ms = MockSerial(wire)
        ms.timeout = 1.0
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_CONTROL)
        self.assertEqual(body, bytes([0x0B, 0x02]))
        self.assertEqual(ms.remaining(), 0)

    def test_channel_response_parsed_into_18_byte_frame(self):
        frame = bytes(range(18))
        # Chip emits CHANNEL0 + CHAND + n_bits + 18 bytes = 21 fields.
        fields = bytes([d.FIELD_CHANNEL0, d.FIELD_CHAND, 144]) + frame
        wire = chip_response(d.PKT_TYPE_CHANNEL, fields)
        ms = MockSerial(wire)
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_CHANNEL)
        n_bits, data = d.extract_channel_bits(body)
        self.assertEqual(n_bits, 144)
        self.assertEqual(data, frame)
        self.assertEqual(ms.remaining(), 0)

    def test_speech_response_parsed_into_160_samples(self):
        # Chip's typical SPEECH response: SPEECHD (no leading CHANNEL0)
        # + n_samples + 320 PCM bytes = 322 fields.
        samples = [(i - 80) * 256 for i in range(160)]
        pcm_bytes = bytearray()
        for v in samples:
            if v < 0:
                v += 0x10000
            pcm_bytes.append((v >> 8) & 0xFF)
            pcm_bytes.append(v & 0xFF)
        fields = bytes([d.FIELD_SPEECHD, 0xA0]) + bytes(pcm_bytes)
        wire = chip_response(d.PKT_TYPE_SPEECH, fields)
        ms = MockSerial(wire)
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_SPEECH)
        recovered = d.extract_pcm(body)
        self.assertEqual(recovered, samples)
        self.assertEqual(ms.remaining(), 0)

    def test_back_to_back_chip_responses_stay_in_frame(self):
        # Guard against the byte-accounting bug returning. If
        # read_response consumes more or fewer bytes than the chip
        # actually sends, the next call mis-parses the next response's
        # header. Concatenate three different responses; all must parse.
        a = chip_response(d.PKT_TYPE_CONTROL, bytes([0x0B, 0x02]))
        frame = bytes(range(18))
        b_fields = bytes([d.FIELD_CHANNEL0, d.FIELD_CHAND, 144]) + frame
        b = chip_response(d.PKT_TYPE_CHANNEL, b_fields)
        c_fields = bytes([d.FIELD_SPEECHD, 0xA0]) + bytes(320)
        c = chip_response(d.PKT_TYPE_SPEECH, c_fields)
        ms = MockSerial(a + b + c)
        ra = d.read_response(ms)
        rb = d.read_response(ms)
        rc = d.read_response(ms)
        self.assertIsNotNone(ra)
        self.assertIsNotNone(rb)
        self.assertIsNotNone(rc)
        self.assertEqual(ra[0], d.PKT_TYPE_CONTROL)
        self.assertEqual(rb[0], d.PKT_TYPE_CHANNEL)
        self.assertEqual(rc[0], d.PKT_TYPE_SPEECH)
        self.assertEqual(ms.remaining(), 0)

    def test_host_request_pack_packet_parity_correct(self):
        # The host→chip pack_packet uses a different wire layout (with
        # marker + parity). Sanity-check the parity computation is
        # still right per the protocol spec.
        fields = [d.FIELD_CHANNEL0, d.FIELD_CHAND, 144] + list(range(18))
        wire = d.pack_packet(d.PKT_TYPE_CHANNEL, fields)
        expected = 0
        for byte in wire[1:-1]:
            expected ^= byte
        self.assertEqual(wire[-1], expected)

    def test_truncated_response_returns_none(self):
        # Chip cut off mid-body — read_response must return None
        # instead of blocking or raising.
        wire = chip_response(d.PKT_TYPE_SPEECH, bytes(322))
        truncated = wire[:50]
        ms = MockSerial(truncated)
        out = d.read_response(ms)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
