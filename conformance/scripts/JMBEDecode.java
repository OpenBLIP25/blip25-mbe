/*
 * Black-box IMBE decoder driver for the chip-bit-format validation probe.
 *
 * Uses JMBE (Dennis Sheirer) as a third-party P25 Phase 1 IMBE decoder to
 * sanity-check whether chip.bit is genuinely non-IMBE-standard. Build the
 * JMBE jar once via the JMBE Creator (produces ~/jmbe-1.0.9.jar).
 *
 * Usage:
 *   javac -cp ~/jmbe-1.0.9.jar JMBEDecode.java
 *   java  -cp ~/jmbe-1.0.9.jar:. JMBEDecode <in.bit> <out.wav>
 *
 * Input format: concatenated 18-byte full-rate FEC frames (same layout as
 * DVSI tv-std .bit files). 144 bits per frame, MSB-first.
 */

import jmbe.JMBEAudioLibrary;
import jmbe.iface.IAudioCodec;
import jmbe.iface.IAudioCodecLibrary;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

public class JMBEDecode {
    private static final int FRAME_BYTES = 18;
    private static final int SAMPLES_PER_FRAME = 160;
    private static final int SAMPLE_RATE = 8000;

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: JMBEDecode <in.bit> <out.wav>");
            System.exit(2);
        }
        byte[] bits = Files.readAllBytes(Paths.get(args[0]));
        int nFrames = bits.length / FRAME_BYTES;
        System.err.printf("input: %s (%d bytes, %d frames)%n",
                args[0], bits.length, nFrames);

        IAudioCodecLibrary lib = new JMBEAudioLibrary();
        System.err.printf("JMBE version: %s%n", lib.getVersion());
        if (!lib.supports("IMBE")) {
            System.err.println("JMBE does not advertise IMBE support");
            System.exit(3);
        }
        IAudioCodec codec = lib.getAudioConverter("IMBE");
        System.err.printf("codec: %s%n", codec.getCodecName());

        ByteBuffer out = ByteBuffer.allocate(nFrames * SAMPLES_PER_FRAME * 2);
        out.order(ByteOrder.LITTLE_ENDIAN);
        int decoded = 0, empty = 0, shortFrames = 0;
        float peakAbs = 0, sumSq = 0;
        long sampleCount = 0;
        for (int f = 0; f < nFrames; f++) {
            byte[] frame = new byte[FRAME_BYTES];
            System.arraycopy(bits, f * FRAME_BYTES, frame, 0, FRAME_BYTES);
            float[] pcm = codec.getAudio(frame);
            if (pcm == null || pcm.length == 0) {
                empty++;
                for (int i = 0; i < SAMPLES_PER_FRAME; i++) out.putShort((short) 0);
                continue;
            }
            if (pcm.length != SAMPLES_PER_FRAME) {
                shortFrames++;
            }
            int n = Math.min(pcm.length, SAMPLES_PER_FRAME);
            for (int i = 0; i < n; i++) {
                float v = pcm[i] * 32767.0f;
                if (v > 32767) v = 32767;
                if (v < -32768) v = -32768;
                short s = (short) v;
                out.putShort(s);
                float a = Math.abs(pcm[i]);
                if (a > peakAbs) peakAbs = a;
                sumSq += pcm[i] * pcm[i];
                sampleCount++;
            }
            for (int i = n; i < SAMPLES_PER_FRAME; i++) out.putShort((short) 0);
            decoded++;
        }

        // Write WAV.
        byte[] raw = out.array();
        AudioFormat fmt = new AudioFormat(SAMPLE_RATE, 16, 1, true, false);
        AudioInputStream ais = new AudioInputStream(
                new ByteArrayInputStream(raw), fmt, raw.length / 2);
        AudioSystem.write(ais, AudioFileFormat.Type.WAVE, new File(args[1]));

        double rms = sampleCount > 0 ? Math.sqrt(sumSq / sampleCount) : 0;
        System.err.printf("decoded=%d empty=%d short=%d  peak=%.3f rms=%.4f%n",
                decoded, empty, shortFrames, peakAbs, rms);
        System.err.printf("wrote %s%n", args[1]);
    }
}
