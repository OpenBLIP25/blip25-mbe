//! Live DVSI USB-3000 chip conformance harness.
//!
//! Drives the AMBE-3000 silicon via the USB-3000 dongle and compares its
//! output against the software decoder/encoder for the same inputs.
//! Intended to run on the project CI host with a chip attached, and
//! (later) to cache `(input_hash → chip_output)` pairs so this harness
//! can degrade gracefully to replay mode if the chip is unavailable.
//!
//! Not published to crates.io.

use anyhow::Result;
use clap::Parser;

/// Run blip25-mbe against live DVSI USB-3000 hardware.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("blip25-mbe chip-conformance harness — not yet implemented");
    std::process::exit(2);
}
