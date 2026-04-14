//! Vector-conformance harness — runs DVSI published test vectors through
//! blip25-mbe and reports per-vector pass/fail.
//!
//! Not published to crates.io. Expects `DVSI/Vectors/` at the workspace
//! root (symlink is fine) with the standard `tv-std/` and `tv-rc/` layouts.

use anyhow::Result;
use clap::Parser;

/// Run DVSI test vectors through blip25-mbe and report conformance.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the DVSI vectors directory.
    #[arg(long, default_value = "DVSI/Vectors")]
    vectors: String,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("blip25-mbe vector-conformance harness — not yet implemented");
    std::process::exit(2);
}
