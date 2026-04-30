#!/usr/bin/env python3
"""Round-trip + byte-accounting tests for `dvsi_driver.py`.

Runs against an in-memory mock serial — no chip required. Validates
that `pack_packet` / `read_response` are inverses across the full set
of packet shapes the driver emits (control / channel / speech), and
that `read_response` consumes exactly the number of wire bytes it
should so back-to-back response decoding stays in frame.

The original driver had a 1-byte over-read in `read_response`: the
body loop swallowed the parity byte that the trailing `s.read(1)`
was meant to consume. On a real chip — which only sends a response
in reply to a request — that extra `s.read(1)` blocked for the full
2-second timeout each frame, dropping throughput from ~50 fps to
~0.5 fps. This test guards against the bug regressing.

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


class PackParseRoundTrip(unittest.TestCase):
    def test_control_packet_round_trip(self):
        # PKT_INIT shape: [type=0x00] [FIELD_INIT 0x0B] [flags 0x02]
        wire = d.pack_packet(d.PKT_TYPE_CONTROL, [d.FIELD_INIT, d.INIT_DECODER])
        ms = MockSerial(wire)
        ms.timeout = 1.0
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_CONTROL)
        self.assertEqual(body, bytes([d.FIELD_INIT, d.INIT_DECODER]))
        self.assertEqual(ms.remaining(), 0, "must consume exactly one packet")

    def test_channel_packet_round_trip(self):
        # P25 full-rate FEC channel packet: 18 bytes of frame data.
        frame = bytes(range(18))
        wire = d.make_channel_packet(frame, n_bits=144)
        ms = MockSerial(wire)
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_CHANNEL)
        # Body should be exactly the field bytes (no marker / parity).
        self.assertEqual(
            body,
            bytes([d.FIELD_CHANNEL0, d.FIELD_CHAND, 144]) + frame,
        )
        # And `extract_channel_bits` should round-trip cleanly.
        n_bits, data = d.extract_channel_bits(body)
        self.assertEqual(n_bits, 144)
        self.assertEqual(data, frame)
        self.assertEqual(ms.remaining(), 0)

    def test_speech_packet_round_trip(self):
        # Speech response: 160 i16 samples. Use a recognisable ramp.
        samples = [(i - 80) * 256 for i in range(160)]
        wire = d.make_speech_packet(samples)
        ms = MockSerial(wire)
        out = d.read_response(ms)
        self.assertIsNotNone(out)
        pkt_type, body = out
        self.assertEqual(pkt_type, d.PKT_TYPE_SPEECH)
        # extract_pcm must recover the ramp exactly.
        recovered = d.extract_pcm(body)
        self.assertEqual(recovered, samples)
        self.assertEqual(ms.remaining(), 0)

    def test_back_to_back_packets_stay_in_frame(self):
        # The bug we're guarding against: read_response consumes more
        # than one packet's worth of bytes, so the next call mis-parses
        # the header. Concatenate three different responses and read
        # them in order.
        a = d.pack_packet(d.PKT_TYPE_CONTROL, [d.FIELD_INIT, d.INIT_DECODER])
        b = d.make_channel_packet(bytes(range(18)), n_bits=144)
        c = d.make_speech_packet([0] * 160)
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
        # Buffer must be exactly empty.
        self.assertEqual(ms.remaining(), 0)

    def test_parity_byte_value_is_correct(self):
        # Sanity-check pack_packet's parity computation. Per protocol:
        # parity = XOR of all bytes between the header and the parity
        # byte itself.
        fields = [d.FIELD_CHANNEL0, d.FIELD_CHAND, 144] + list(range(18))
        wire = d.pack_packet(d.PKT_TYPE_CHANNEL, fields)
        # wire[-1] is the parity byte; XOR over wire[1:-1] should equal it.
        expected = 0
        for byte in wire[1:-1]:
            expected ^= byte
        self.assertEqual(wire[-1], expected)

    def test_timeout_on_short_packet_returns_none(self):
        # Truncated packet (only header + half of body). read_response
        # should return None rather than blocking or raising.
        wire = d.make_speech_packet([0] * 160)
        truncated = wire[:50]
        ms = MockSerial(truncated)
        out = d.read_response(ms)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
