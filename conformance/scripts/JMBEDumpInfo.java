/*
 * A/B probe for gap 0021 — print the post-FEC u0..u7 info vectors that
 * JMBE recovers from a single chip.bit frame, so we can compare against
 * our spec-derived decoder's info[] for the same frame.
 *
 * Usage:
 *   javac -cp <JMBE_CP> JMBEDumpInfo.java
 *   java  -cp <JMBE_CP>:. JMBEDumpInfo <in.bit> <frame_idx>
 *
 * Output: one line "u0..u7 = 0xXXX 0xXXX ..." for the requested frame,
 * plus the per-vector error counts JMBE itself reports.
 *
 * Field widths per BABA-A §8.1: u0..u3 = 12 bits, u4..u6 = 11 bits,
 * u7 = 7 bits. After JMBE deinterleave, c_j bit (N-1) (MSB) is at
 * position offset_j + 0; u_j occupies the high width_j bits of c_j.
 */

import jmbe.codec.imbe.IMBEFrame;
import jmbe.binary.BinaryFrame;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;

public class JMBEDumpInfo {
    private static final int FRAME_BYTES = 18;
    private static final int[] OFFSETS = {0, 23, 46, 69, 92, 107, 122, 137};
    private static final int[] WIDTHS  = {12, 12, 12, 12, 11, 11, 11, 7};

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: JMBEDumpInfo <in.bit> <frame_idx>");
            System.exit(2);
        }
        byte[] all = Files.readAllBytes(Paths.get(args[0]));
        int frameIdx = Integer.parseInt(args[1]);
        byte[] frame = new byte[FRAME_BYTES];
        System.arraycopy(all, frameIdx * FRAME_BYTES, frame, 0, FRAME_BYTES);
        IMBEFrame imbe = new IMBEFrame(frame);
        BinaryFrame bf = imbe.getFrame();

        // Reflectively pull mErrors from IMBEFrame so we can compare error
        // counts. (mErrors is private but we just want the int array.)
        Field errF = IMBEFrame.class.getDeclaredField("mErrors");
        errF.setAccessible(true);
        int[] errs = (int[]) errF.get(imbe);

        System.out.print("frame " + frameIdx + ": u0..u7 = ");
        for (int j = 0; j < 8; j++) {
            int v = 0;
            for (int b = 0; b < WIDTHS[j]; b++) {
                if (bf.get(OFFSETS[j] + b)) {
                    // Position OFFSETS[j] + 0 is the MSB of c_j; build the
                    // info value MSB-first to match the BABA-A §8.1
                    // convention.
                    v |= 1 << (WIDTHS[j] - 1 - b);
                }
            }
            System.out.printf("0x%03x ", v);
        }
        System.out.print(" errors=");
        for (int e : errs) System.out.print(e + " ");
        System.out.println();

        // Also dump the full 144-bit deinterleaved+derandomized+corrected
        // frame for byte-level inspection.
        StringBuilder sb = new StringBuilder("post_fec_bits=");
        for (int i = 0; i < 144; i++) {
            sb.append(bf.get(i) ? '1' : '0');
            if (i % 8 == 7) sb.append(' ');
        }
        System.out.println(sb.toString());
    }
}
