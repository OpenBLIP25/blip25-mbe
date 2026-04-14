//! Speech-quality conformance harness.
//!
//! Runs blip25-mbe output through PESQ / POLQA / ViSQOL scoring per the
//! BABG methodology: encoder and decoder tested independently against a
//! reference side, across the 15 public-safety noise environments from
//! BABG, with LQO ≥ 2.0 as the pass threshold.
//!
//! Later milestone. Not published to crates.io.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("blip25-mbe speech-quality harness — scheduled for a later milestone");
    std::process::exit(2);
}
