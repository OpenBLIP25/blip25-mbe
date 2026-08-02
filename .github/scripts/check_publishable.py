#!/usr/bin/env python3
"""Pre-flight checks for a crates.io release.

This exists because the 0.3.0 publish broke silently: `blip25-mbe` gained a
path dependency on `blip25-codec` with no `version`, which `cargo publish`
rejects — and nothing noticed until a release was attempted, because no CI job
exercised publishability.

A full `cargo package` cannot verify the two dependent crates before release:
their path deps point at workspace siblings at the *upcoming* version, which by
definition is not on crates.io yet, so dependency resolution fails. What we can
check without the registry:

  1. every path dependency of a publishable crate declares a version
  2. those versions agree with the version actually being published
  3. each crate's `include` list covers everything its build.rs reads
  4. `cargo package --list` succeeds (manifest parses, globs resolve)
  5. each crate ships a LICENSE matching the root copy

Run: python3 .github/scripts/check_publishable.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def cargo_metadata() -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def main() -> int:
    meta = cargo_metadata()
    # cargo reports publish=None for "publishable anywhere", publish=[] for
    # `publish = false`, or a list of allowed registries.
    publishable = [p for p in meta["packages"] if p.get("publish") is None]
    names = {p["name"] for p in publishable}

    print(f"publishable crates: {sorted(names)}\n")
    if not names:
        print("no publishable crates found — is this the right workspace?")
        return 1

    print("[1/5] path dependencies declare a version")
    for p in publishable:
        for d in p["dependencies"]:
            if not d.get("path"):
                continue
            if d["req"] == "*":
                fail(f"{p['name']} -> {d['name']}: path dep has no version; "
                     f"`cargo publish` will reject this")
            else:
                ok(f"{p['name']} -> {d['name']} = {d['req']}")

    print("\n[2/5] internal dependency versions match the release version")
    versions = {p["name"]: p["version"] for p in publishable}
    for p in publishable:
        for d in p["dependencies"]:
            if d["name"] not in names or not d.get("path"):
                continue
            want = versions[d["name"]]
            # req looks like "^0.3.0"; compare on the bare version.
            if want not in d["req"]:
                fail(f"{p['name']} requires {d['name']} {d['req']} but that "
                     f"crate is at {want} — lockstep versions must agree")
            else:
                ok(f"{p['name']} requires {d['name']} {d['req']} (is {want})")

    print("\n[3/5] `include` covers build.rs inputs")
    for p in publishable:
        manifest = Path(p["manifest_path"])
        crate_dir = manifest.parent
        build_rs = crate_dir / "build.rs"
        if not build_rs.exists():
            ok(f"{p['name']}: no build.rs")
            continue
        listed = package_list(p["name"])
        if listed is None:
            continue
        # Every directory the build script reads from must appear in the
        # packaged file list, or the published crate fails to build.
        src = build_rs.read_text(encoding="utf8")
        missing = []
        for token in ("spec_tables", "tables"):
            if f'"{token}/' in src or f"'{token}/" in src or f'join("{token}")' in src:
                if not any(f.startswith(f"{token}/") for f in listed):
                    missing.append(token)
        if missing:
            fail(f"{p['name']}: build.rs reads {missing} but no such files are "
                 f"packaged — the published crate will not build")
        else:
            ok(f"{p['name']}: build.rs inputs are packaged")

    print("\n[4/5] `cargo package --list` succeeds")
    for p in publishable:
        if package_list(p["name"]) is not None:
            ok(f"{p['name']}: manifest parses, include globs resolve")

    # Cargo cannot package files above the crate root, so anything shipped in
    # the tarball has to exist as a per-crate copy and can silently drift from
    # the root original. Only LICENSE is duplicated that way. PATENT_NOTICE.md
    # and ATTRIBUTION.md live at the repository root only — the crate
    # descriptions point readers there rather than into the tarball.
    print("\n[5/5] the license is shipped and matches the root copy")
    notices = ["LICENSE"]
    for p in publishable:
        crate_dir = Path(p["manifest_path"]).parent
        listed = package_list(p["name"])
        if listed is None:
            continue
        for n in notices:
            if n not in listed:
                fail(f"{p['name']}: {n} is not in the package — every "
                     f"published crate must carry its license")
                continue
            root, local = ROOT / n, crate_dir / n
            if not root.exists():
                fail(f"{n} missing at repository root")
            elif root.read_bytes() != local.read_bytes():
                fail(f"{p['name']}/{n} has drifted from the root copy — "
                     f"re-copy it (`cp {n} {crate_dir.relative_to(ROOT)}/`)")
            else:
                ok(f"{p['name']}: {n} shipped, matches root")

    print()
    if failures:
        print(f"{len(failures)} publishability problem(s) found")
        return 1
    print("all publishability checks passed")
    return 0


_list_cache: dict[str, list[str] | None] = {}


def package_list(name: str) -> list[str] | None:
    if name in _list_cache:
        return _list_cache[name]
    r = subprocess.run(
        ["cargo", "package", "-p", name, "--allow-dirty", "--no-verify", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        fail(f"{name}: `cargo package --list` failed:\n{r.stderr.strip()[:400]}")
        _list_cache[name] = None
    else:
        _list_cache[name] = r.stdout.split()
    return _list_cache[name]


if __name__ == "__main__":
    sys.exit(main())
