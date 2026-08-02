# Releasing

This workspace publishes **three crates in lockstep** at one version:

```
blip25-codebooks  ->  blip25-codec  ->  blip25-mbe
```

They must be uploaded in that order — `cargo publish` resolves each path
dependency's `version` against the registry, so a crate has to be live before
the one that depends on it goes up.

Only `blip25-mbe` is a supported dependency. The other two are published solely
because crates.io cannot resolve a path dependency; they carry no API stability
promise, and both READMEs say so.

> **Publishing is irreversible.** A version can be yanked but never deleted, and
> a crate name can never be reclaimed. This crate ships a patent-encumbered,
> reverse-engineered codec — re-read [`PATENT_NOTICE.md`](./PATENT_NOTICE.md)
> and [`ATTRIBUTION.md`](./ATTRIBUTION.md) before the first upload of any new
> crate name.

## One-time setup

1. A crates.io API token in repo secrets as `CARGO_REGISTRY_TOKEN`
   (Settings → Secrets and variables → Actions).
2. For the **first** publish of `blip25-codebooks` and `blip25-codec`, the token
   needs the **publish-new** scope, not just publish-update. Afterwards
   publish-update is enough.

## Release steps

1. **Pick the version.** Cargo 0.x rules: `0.x.y -> 0.x.(y+1)` must be
   backward compatible; `0.x -> 0.(x+1)` may break. The `semver` CI job checks
   this against the last published release and fails if the bump is too small.

2. **Bump it in two places** in the root `Cargo.toml` — they must match:
   - `[workspace.package] version`
   - the three `blip25-*` entries under `[workspace.dependencies]`

3. **Update `CHANGELOG.md`**: move `[Unreleased]` content into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and add the compare link at the bottom.
   If the release changes codec output, provenance, or the legal posture, say
   so at the top of the section — that is what an existing user needs first.

4. **Verify locally** (all of these run in CI, but the publish is one-way):

   ```bash
   cargo test --workspace --release
   cargo test -p blip25-mbe --all-features --release
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p blip25-mbe --all-features
   cargo deny check advisories bans licenses sources
   cargo +1.85 build -p blip25-mbe --all-features     # declared MSRV
   python3 .github/scripts/check_publishable.py       # publishability pre-flight
   cargo package -p blip25-codebooks                  # full verify build
   ```

   Only `blip25-codebooks` can be verify-built before release: the other two
   depend on workspace siblings at the version being released, which is not on
   crates.io yet.

   Note this is *not* a `cargo package --no-verify` on the other two —
   `--no-verify` skips the build but still resolves dependencies, so it fails
   for the same reason. That is what `check_publishable.py` is for: it uses
   `cargo metadata` and `cargo package --list`, neither of which touches the
   registry, to confirm every path dep declares a matching version and every
   `build.rs` input is inside `include`. Those are the two ways a release
   breaks that nothing else catches until upload.

5. **Commit to `main`**, wait for CI green.

6. **Tag and push**:

   ```bash
   git tag -a vX.Y.Z -m "…"
   git push origin vX.Y.Z
   ```

   `.github/workflows/publish.yml` fires on `v*`, re-verifies that the tag
   matches all three crate versions, tests, then publishes bottom-up.

## If a publish half-fails

The workflow publishes sequentially, so a failure can leave some crates up and
others not. Versions are immutable — you cannot re-upload the same number.

- **`blip25-codebooks` succeeded, `blip25-codec` failed.** Fix the problem, bump
  to the next patch version everywhere, re-tag. The orphaned codebooks version
  is harmless; nothing depends on it yet. Yank it if you prefer a clean index.
- **All but `blip25-mbe` succeeded.** Same: bump, re-tag. The engine crates at
  the skipped version are unreferenced.
- **Never** try to work around this by publishing by hand out of order.

## What is deliberately not published

`conformance/roundtrip` is `publish = false` — its examples read the
non-redistributable reference corpus, so the crate is unrunnable for anyone else.
`fuzz/` is outside the workspace entirely (it needs nightly). Neither ships.
