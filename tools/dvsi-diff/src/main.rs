//! dvsi-diff — run one DVSI vector through blip25-mbe and show exactly
//! where our output diverges from the reference.
//!
//! Unlike the conformance harness (pass / fail, regression-oriented),
//! this tool is built for active debugging: dump MBE parameters at each
//! pipeline stage, show bit-level diffs against the expected output,
//! pinpoint the first stage at which divergence appears.

use anyhow::Result;
use clap::Parser;

/// Diagnose a single DVSI-vector failure stage by stage.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Vector identifier (e.g. `tv-std/voice_0042`).
    #[arg(long)]
    vector: Option<String>,

    /// Stage to dump (`wire`, `params`, `synthesis`, `all`).
    #[arg(long, default_value = "all")]
    stage: String,

    /// Verbose per-harmonic output.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("dvsi-diff — not yet implemented");
    std::process::exit(2);
}
