use blip25_mbe::rate33::{fields_from_natural, fields_from_no_fec};
use blip25_mbe::vocoder::{LiveEncoder, Rate};
use std::fs;
fn main(){
    for (nm,src) in [("mark","/mnt/share/Blip25/Research/DVSI Vectors/tv-rc/mark.pcm"),
                     ("clean","/mnt/share/Blip25/Research/DVSI Vectors/tv-std/tv/clean.pcm"),
                     ("dam","/mnt/share/Blip25/Research/DVSI Vectors/tv-std/tv/dam.pcm"),
                     ("noisy","/mnt/share/Blip25/Research/DVSI Vectors/tv-std/tv/noisy.pcm")] {
        let pcm: Vec<i16> = fs::read(src).unwrap().chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0],c[1]])).collect();
        let orc: Vec<[u16;9]> = fs::read(format!("/mnt/share/Blip25/oracle_bits/{nm}.ambe2.bits"))
            .unwrap().chunks_exact(7).map(fields_from_natural).collect();
        let mut le = LiveEncoder::new(Rate::AmbePlus2_2450x2450);
        let mut ours: Vec<[u16;9]> = Vec::new();
        for ch in pcm.chunks(160) { for r in le.push(ch) { ours.push(fields_from_no_fec(&r.unwrap())); } }
        for f in le.flush().unwrap() { ours.push(fields_from_no_fec(&f)); }
        let fam = |b:u16| (18..=31).contains(&b);
        let o_fam = orc.iter().filter(|f| fam(f[1])).count();
        let u_fam = ours.iter().filter(|f| fam(f[1])).count();
        println!("{nm:6} frames={:5}  oracle family={:4}  ours family={:4}", ours.len(), o_fam, u_fam);
    }
}
