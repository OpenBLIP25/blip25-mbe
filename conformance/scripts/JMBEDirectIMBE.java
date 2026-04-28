/*
 * Direct IMBE decoder using JMBE's codec module only (no iface jar
 * required). Reads 18-byte IMBE frames from a .bit file, decodes via
 * jmbe.codec.imbe.IMBEFrame + IMBESynthesizer, writes 8 kHz / 16-bit
 * mono WAV.
 *
 * Compiles against jmbe-1.0.9.jar (codec-only).
 *
 * Usage:
 *   javac -cp ~/jmbe-1.0.9.jar JMBEDirectIMBE.java
 *   java  -cp ~/jmbe-1.0.9.jar:. JMBEDirectIMBE <in.bit> <out.wav>
 */

import jmbe.codec.imbe.IMBEFrame;
import jmbe.codec.imbe.IMBESynthesizer;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

public class JMBEDirectIMBE {
    private static final int FRAME_BYTES = 18;
    private static final int SAMPLES_PER_FRAME = 160;
    private static final int SAMPLE_RATE = 8000;

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: JMBEDirectIMBE <in.bit> <out.wav>");
            System.exit(2);
        }
        byte[] bits = Files.readAllBytes(Paths.get(args[0]));
        if (bits.length % FRAME_BYTES != 0) {
            System.err.printf("input length %d not a multiple of %d%n", bits.length, FRAME_BYTES);
            System.exit(3);
        }
        int nFrames = bits.length / FRAME_BYTES;

        IMBESynthesizer synth = new IMBESynthesizer();
        ByteBuffer out = ByteBuffer.allocate(nFrames * SAMPLES_PER_FRAME * 2);
        out.order(ByteOrder.LITTLE_ENDIAN);

        for (int f = 0; f < nFrames; f++) {
            byte[] frame = new byte[FRAME_BYTES];
            System.arraycopy(bits, f * FRAME_BYTES, frame, 0, FRAME_BYTES);
            IMBEFrame imbe = new IMBEFrame(frame);
            float[] pcm = synth.getAudio(imbe);
            // JMBE returns float in [-1.0, 1.0]; scale to i16 range.
            for (int s = 0; s < SAMPLES_PER_FRAME; s++) {
                int v = Math.round(pcm[s] * 32767.0f);
                if (v > Short.MAX_VALUE) v = Short.MAX_VALUE;
                if (v < Short.MIN_VALUE) v = Short.MIN_VALUE;
                out.putShort((short) v);
            }
        }

        AudioFormat fmt = new AudioFormat(
            AudioFormat.Encoding.PCM_SIGNED,
            SAMPLE_RATE, 16, 1, 2, SAMPLE_RATE, false);
        try (AudioInputStream ais = new AudioInputStream(
                new ByteArrayInputStream(out.array()), fmt,
                nFrames * SAMPLES_PER_FRAME)) {
            AudioSystem.write(ais, AudioFileFormat.Type.WAVE, new File(args[1]));
        }
        System.err.printf("decoded %d frames -> %s%n", nFrames, args[1]);
    }
}
