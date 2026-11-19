"""Gap C observation runner - THEIA_SHIFT_MASK differential on the hybrid_k3 fixture.

Implements Theia/PLAN.md's recorded Gap-C contract (verification plan ae00516,
"order C -> B -> A"), verbatim:

    "Observation is parameter-driven via the existing THEIA_SHIFT_MASK step-through
     (bit0 = k=2, bit1 = k>=3): evidence = mask differential renders (k3-only vs
     k2-only vs all) on the fixture must differ AND the k3-only arm must stay
     gate-clean vs Hyperion - proving the k=3 body runs and is correct."

Mask semantics (Application.cpp L96-99 / gi.comp.slang L56-59):
    1 = k=2 only, 2 = k>=3 only, 3 = all (default).

The four checks and what each proves:
    A. mask1 vs mask3 DIFFER  -> the k=3 body actually fires (turning k3 off
       changes the image). If identical, the fixture does not exercise k=3 and
       Gap C's premise FAILS (distinct exit code, never reported as a pass).
    B. mask2 vs mask3 DIFFER  -> the k=2 body contributes (arms are separable).
    C. mask2 vs Hyperion PASS -> the k3-only arm is gate-clean (correctness).
    D. masks vs Hyperion reported for context.

No invented thresholds: "differs" is byte-identity (sha256) plus mean_diff, and
correctness uses the family gate verbatim - compare_renders.py --gate all
(mean_diff<=120, rel_mse<=250, SSIM>=0.6, lum_hist>=0.8), whose exit code IS the
gate (return 0 if passed else 1). Run-to-run pixel reproducibility of multi-frame
captures is already on record (Theia/PLAN.md L299), so a byte difference between
arms cannot be Monte-Carlo noise; same-seed renders are reproducible.

Budget = the gate's calibration budget (AGENTS.md: 320x240 / 256 spp / 256
frames) - the thresholds are budget-specific, so the arms must render at it.

GPU discipline: all renders run SEQUENTIALLY in this one process (one GPU app).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

GITHUB = Path(r"C:\Development\GitHub")
STAGED = (GITHUB / "Theia" / "build" / "sms-qualification" / "scenes" / "HYBRID"
          / "hybrid_k3.scene.toml")
THEIA = GITHUB / "Theia" / "build" / "theia.exe"
HYPERION = GITHUB / "Hyperion" / "build" / "hyperion.exe"
COMPARE = GITHUB / "Harmonia" / "tools" / "compare_renders.py"

MASKS = {"all": 3, "k2_only": 1, "k3_only": 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_exe(label: str, exe: Path, args: list[str], out: Path,
            extra_env: dict[str, str] | None = None) -> tuple[int, float]:
    """qualify_sms.run_exe's contract: log to file, fail on GPU/validation noise."""
    log_path = out.with_suffix(".log")
    if out.exists():
        out.unlink()
    if log_path.exists():
        log_path.unlink()
    env = dict(os.environ)
    env["HARMONIA_FRAME_TIMING"] = "1"
    if extra_env:
        env.update(extra_env)
    started = time.perf_counter()
    proc = subprocess.run([str(exe), *args], capture_output=True, text=True,
                          check=False, env=env)
    elapsed = time.perf_counter() - started
    text = proc.stdout + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    if "GPU error bits" in text or "VUID-" in text:
        raise SystemExit(f"{label}: GPU/validation failure; see {log_path.name}")
    if proc.returncode != 0 or not out.is_file():
        raise SystemExit(f"{label}: render failed (code {proc.returncode}); "
                         f"see {log_path.name}")
    print(f"  rendered {label:24s} {elapsed:7.1f}s -> {out.name}", flush=True)
    return proc.returncode, elapsed


def render_hyperion(scene: Path, out: Path, args: argparse.Namespace) -> None:
    run_exe("hyperion reference", HYPERION, [
        "--scene", str(scene), "--sms-mode", "off",
        "--width", str(args.width), "--height", str(args.height),
        "--spp", str(args.spp), "--rng-seed", str(args.seed),
        "--validation", "--output", str(out),
    ], out)


def render_theia(scene: Path, out: Path, mask: int, args: argparse.Namespace) -> None:
    run_exe(f"theia mask={mask}", THEIA, [
        "--scene", str(scene), "--sms-mode", args.sms_mode,
        "--width", str(args.width), "--height", str(args.height),
        "--offscreen-frames", str(args.frames), "--rng-seed", str(args.seed),
        "--validation", "--output", str(out),
    ], out, extra_env={"THEIA_SHIFT_MASK": str(mask)})


def gate(reference: Path, candidate: Path) -> tuple[bool, str]:
    """The family strict-AND gate, verbatim (exit code is the verdict)."""
    proc = subprocess.run(
        [sys.executable, str(COMPARE), str(reference), str(candidate),
         "--no-heatmap", "--gate", "all"],
        capture_output=True, text=True, check=False)
    tail = [line for line in (proc.stdout + proc.stderr).splitlines()
            if "RESULT" in line or "mean_diff" in line or "SSIM" in line]
    return proc.returncode == 0, " | ".join(line.strip() for line in tail[-3:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=GITHUB / "Theia" / "build" / "sms-qualification" / "gapc")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sms-mode", default="local")
    ap.add_argument("--skip-render", action="store_true",
                    help="re-analyze EXRs already on disk")
    args = ap.parse_args()

    for path, what in ((STAGED, "staged fixture"), (THEIA, "theia.exe"),
                       (HYPERION, "hyperion.exe"), (COMPARE, "compare_renders.py")):
        if not path.is_file():
            print(f"missing {what}: {path}")
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    ref = args.out / "hyperion.exr"
    arms = {name: args.out / f"theia_mask{mask}.exr" for name, mask in MASKS.items()}

    if not args.skip_render:
        print(f"Gap C renders (budget {args.width}x{args.height}, "
              f"{args.spp}spp / {args.frames}f, seed {args.seed}) - SEQUENTIAL\n")
        render_hyperion(STAGED, ref, args)
        # Order matters for cost: 'all' first so A/B are available if a later
        # render dies; every render is independent and uses the same seed.
        for name in ("all", "k2_only", "k3_only"):
            render_theia(STAGED, arms[name], MASKS[name], args)
        print()

    for path in (ref, *arms.values()):
        if not path.is_file():
            print(f"missing render: {path}")
            return 2

    print("=" * 72)
    print("Gap C checks (Theia/PLAN.md verification plan ae00516)")
    print("=" * 72)

    h_ref, h_all = sha256(ref), sha256(arms["all"])
    h_k2, h_k3 = sha256(arms["k2_only"]), sha256(arms["k3_only"])

    ok_a = h_k2 != h_all
    ok_b = h_k3 != h_all
    print(f"A. k3 body FIRES   : mask1(k2-only) vs mask3(all) byte-identical = "
          f"{h_k2 == h_all}  -> {'FIRES' if ok_a else 'NEVER FIRES (Gap C premise fails)'}")
    print(f"   sha256 k2-only={h_k2[:16]}  all={h_all[:16]}")
    print(f"B. arms separable  : mask2(k3-only) vs mask3(all) byte-identical = "
          f"{h_k3 == h_all}  -> {'DIFFER' if ok_b else 'IDENTICAL'}")
    print(f"   sha256 k3-only={h_k3[:16]}  all={h_all[:16]}")
    print(f"   reference       = {h_ref[:16]}")

    print("\nC. k3-only arm gate-clean vs Hyperion (family strict-AND gate):")
    c_pass, c_detail = gate(ref, arms["k3_only"])
    print(f"   {'PASS' if c_pass else 'FAIL'}  {c_detail}")

    print("\nD. context - other arms vs Hyperion:")
    for name in ("all", "k2_only"):
        p, detail = gate(ref, arms[name])
        print(f"   {name:8s} {'PASS' if p else 'FAIL'}  {detail}")

    # Numeric magnitudes for the differential (reported, not thresholded: the
    # plan asks for "must differ", which byte-identity already settles).
    print("\nPairwise mean_diff (1/255):")
    for a, b in (("k2_only", "all"), ("k3_only", "all"), ("k3_only", "k2_only")):
        p, detail = gate(arms[a], arms[b])
        print(f"   {a:8s} vs {b:8s}  {detail}")

    ok = ok_a and ok_b and c_pass
    print()
    if not ok_a:
        print("GAP C RESULT: FAIL - the fixture never reaches the k=3 body; the")
        print("capture-gate premise (pair 1 rejected, pair 2 qualifying) did not hold.")
        return 3
    print(f"GAP C RESULT: {'PASS' if ok else 'FAIL'}"
          f"  (k3 fires={ok_a}, arms differ={ok_b}, k3-only gate={c_pass})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
