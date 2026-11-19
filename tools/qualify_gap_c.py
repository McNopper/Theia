"""Gap C observation runner - THEIA_SHIFT_MASK differential on the hybrid_k3 fixture.

Implements Theia/PLAN.md's recorded Gap-C contract (verification plan ae00516,
"order C -> B -> A"):

    "evidence = mask differential renders (k3-only vs k2-only vs all) on the
     fixture must differ AND the k3-only arm must stay gate-clean vs Hyperion -
     proving the k=3 body runs and is correct."

TRUE mask semantics, read from the shader (this is what the contract maps onto):
  gi.comp.slang reads `reconnectShiftEnabled` in exactly two ways -
      L1573  hybridMode = (reconnectShiftEnabled & 2u) != 0u
      L2456/2463/2474  if (reconnectShiftEnabled != 0u)
  `& 1u` is NEVER tested, and the shift BODY is chosen by the captured kLevel
  (L2464-2467: kLevel==1 -> tryReconnectionShift, >=2 -> tryReconnectionShiftK3),
  deliberately NOT by the mask (L2457-2462). Therefore:
      mask 0 = replay-only (no shift at all)
      mask 1 = RECONNECTION mode (bit1 clear): unconditional capture at the first
               secondary hit -> the k=2 body only
      mask 2 = HYBRID mode (bit1 set): first pair passing alpha_min/d_min ->
               adaptive k. Bit0 is not independently consumed, so mask 2 and
               mask 3 are FUNCTIONALLY IDENTICAL - there is no "k3-only" mask.
  The plan's arm names map onto reality as:  k2-only = mask 1,
  "k3/all" = mask 3 (on this fixture the pair gates force captureTarget = 1 ->
  kLevel 2 -> the k=3 body), and mask 0 = replay baseline.

Checks and what each proves:
  0. CONTENT PRECONDITION (exit 5): every frame - reference and all arms - must be
     non-degenerate (max > 0 and std > 0). The family strict-AND gate CANNOT catch
     two identical black images (0 vs 0 passes every threshold), so degenerate
     frames invalidate the whole observation. This exists because a first run
     produced five all-zero EXRs and reported a vacuous PASS.
  A. mask1 vs mask3 DIFFER  -> the k=3 body fires. The fixture's whole design is
     that hybrid must capture at pair 2 (k=3) where reconnection unconditionally
     captures at k=2; if the arms are identical, the criteria never advanced past
     pair 1 and the fixture never reached k=3 -> exit 3 (premise failure, never
     reported as a pass).
  B. mask0 vs mask1 DIFFER  -> the mask genuinely reaches the estimator (instrument
     sanity: without this, a degenerate A could mean a dead knob, not a dead k=3).
  C. mask3 vs Hyperion PASS -> the k=3-capable arm is gate-clean (correctness).
  D. mask2 vs mask3 reported -> expected IDENTICAL (documents that bit0 is not
     independently read); any difference would CONTRADICT this reading and is
     flagged loudly.

No invented thresholds: "differs" = PIXEL max|d| > 0 - never whole-file hashes.
EXR headers carry metadata, so pixel-identical renders hash differently: a sha256
basis reported LIVE/FIRES/UNEXPECTED on frames that were all exactly zero. The
decision rests on the recorded run-to-run pixel reproducibility of multi-frame
captures (Theia/PLAN.md), and correctness uses the family gate verbatim -
compare_renders.py --gate all (mean_diff<=120, rel_mse<=250, SSIM>=0.6,
lum_hist>=0.8), whose exit code IS the gate (return 0 if passed else 1).
Exit codes: 0 pass, 1 gate fail, 2 missing input/render, 3 k=3 never fires,
4 mask dead, 5 degenerate (all-zero/uniform) frame content.

Budget = the gate's own calibration budget (AGENTS.md: 320x240 / 256 spp / 256
frames) - the thresholds are budget-specific, so every arm renders at it.

GPU discipline: all renders run SEQUENTIALLY inside this one process.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Development\GitHub") / "Harmonia" / "tools"))
from compare_renders import load_exr

import numpy as np

GITHUB = Path(r"C:\Development\GitHub")
STAGED = (GITHUB / "Theia" / "build" / "sms-qualification" / "scenes" / "HYBRID"
          / "hybrid_k3.scene.toml")
THEIA = GITHUB / "Theia" / "build" / "theia.exe"
HYPERION = GITHUB / "Hyperion" / "build" / "hyperion.exe"
COMPARE = GITHUB / "Harmonia" / "tools" / "compare_renders.py"

# arm label -> THEIA_SHIFT_MASK value (render order = cheapest risk first: the
# two arms check A and B depend on come before the redundant mask-2 probe).
ARMS: dict[str, int] = {"reconnection_k2": 1, "hybrid_k3": 3, "replay": 0, "hybrid_bit0": 2}


def pixels(path: Path) -> np.ndarray:
    """Pixel content of an EXR - the ONLY basis for a differencing decision.

    Whole-file sha256 must never decide a differential: EXR headers carry
    metadata, so two pixel-identical renders hash differently and would be
    reported as 'differing'. (That bug reported LIVE/FIRES/UNEXPECTED on
    frames that were all exactly zero.)
    """
    return load_exr(path)


def content_state(arr: np.ndarray) -> tuple[bool, str]:
    """Reject degenerate frames BEFORE any comparison.

    The family strict-AND gate cannot catch two identical black images (0 vs 0
    passes every threshold), so a non-degenerate-content assertion is the
    instrument's own precondition.
    """
    lo, hi = float(arr.min()), float(arr.max())
    spread = float(arr.std())
    ok = hi > 0.0 and spread > 0.0
    return ok, f"min {lo:.6f} max {hi:.6f} std {spread:.6f}"


def pixel_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def run_exe(label: str, exe: Path, args: list[str], out: Path,
            extra_env: dict[str, str] | None = None) -> float:
    """qualify_sms.run_exe's contract: log to file, fail on GPU/validation noise."""
    log_path = out.with_suffix(".log")
    for stale in (out, log_path):
        if stale.exists():
            stale.unlink()
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
    print(f"  rendered {label:22s} {elapsed:7.1f}s -> {out.name}", flush=True)
    return elapsed


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
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines()
             if "RESULT" in line or "mean_diff" in line or "SSIM" in line]
    return proc.returncode == 0, " | ".join(lines[-3:])


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
    arms = {name: args.out / f"theia_mask{mask}.exr" for name, mask in ARMS.items()}

    if not args.skip_render:
        print(f"Gap C renders (budget {args.width}x{args.height}, "
              f"{args.spp}spp / {args.frames}f, seed {args.seed}) - SEQUENTIAL\n")
        render_hyperion(STAGED, ref, args)
        for name, mask in ARMS.items():
            render_theia(STAGED, arms[name], mask, args)
        print()

    for path in (ref, *arms.values()):
        if not path.is_file():
            print(f"missing render: {path}")
            return 2

    print("=" * 74)
    print("Gap C checks (Theia/PLAN.md verification plan ae00516)")
    print("=" * 74)

    # Content precondition: the strict-AND gate cannot catch two identical black
    # images (0 vs 0 passes every threshold), so degenerate frames make every
    # downstream number meaningless. Assert non-degenerate BEFORE any diff.
    frames: dict[str, np.ndarray] = {"reference": pixels(ref)}
    frames.update({name: pixels(path) for name, path in arms.items()})

    print(f"{'frame':16s} {'min':>10s} {'max':>10s} {'std':>10s}  state")
    degenerate = []
    for name, arr in frames.items():
        ok_c, detail = content_state(arr)
        print(f"{name:16s} {detail}  {'OK' if ok_c else 'DEGENERATE'}")
        if not ok_c:
            degenerate.append(name)
    if degenerate:
        print("\nGAP C RESULT: FAIL - degenerate frame(s): "
              + ", ".join(degenerate))
        print("The renders produced no image content, so no differential and no")
        print("gate result derived from them is meaningful (0 vs 0 passes every")
        print("threshold). Fix the fixture/render before observing anything.")
        return 5

    # PIXEL-based differencing (never file hashes: EXR headers carry metadata).
    # B first: without a live knob, A's outcome is uninterpretable.
    d_b = pixel_delta(frames["replay"], frames["reconnection_k2"])
    ok_b = d_b > 0.0
    print(f"\nB. mask REACHES estimator : replay(mask0) vs reconnection(mask1) "
          f"max|d| = {d_b:.6f}  -> {'LIVE' if ok_b else 'DEAD KNOB (differential uninterpretable)'}")

    d_a = pixel_delta(frames["reconnection_k2"], frames["hybrid_k3"])
    ok_a = d_a > 0.0
    print(f"A. k3 body FIRES          : reconnection(mask1) vs hybrid(mask3) "
          f"max|d| = {d_a:.6f}  -> {'FIRES' if ok_a else 'NEVER FIRES (Gap C premise fails)'}")

    # D: mask2 must equal mask3 (bit0 is not independently read) - a difference
    # would contradict the shader reading above and must be surfaced, not buried.
    d_d = pixel_delta(frames["hybrid_bit0"], frames["hybrid_k3"])
    bit0_same = d_d == 0.0
    print(f"D. mask2 vs mask3 (bit0)  : max|d| = {d_d:.6f}  -> "
          f"{'as predicted (bit0 not independently read)' if bit0_same else 'UNEXPECTED - contradicts the shader reading'}")

    print("\nC. hybrid(k3-capable) arm gate-clean vs Hyperion (family strict-AND gate):")
    c_pass, c_detail = gate(ref, arms["hybrid_k3"])
    print(f"   {'PASS' if c_pass else 'FAIL'}  {c_detail}")

    print("\n   context - other arms vs Hyperion:")
    for name in ("reconnection_k2", "replay"):
        p, detail = gate(ref, arms[name])
        print(f"   {name:16s} {'PASS' if p else 'FAIL'}  {detail}")

    print("\n   pairwise pixel max|d| (1/255-scale linear EXR):")
    for a, b in (("reconnection_k2", "hybrid_k3"), ("replay", "reconnection_k2"),
                 ("hybrid_bit0", "hybrid_k3")):
        print(f"   {a:16s} vs {b:14s} {pixel_delta(frames[a], frames[b]):.6f}")

    print()
    if not ok_b:
        print("GAP C RESULT: FAIL - THEIA_SHIFT_MASK does not reach the estimator,")
        print("so no differential observation is possible.")
        return 4
    if not ok_a:
        print("GAP C RESULT: FAIL - the fixture never reaches the k=3 body; the")
        print("capture-gate premise (pair 1 rejected, pair 2 qualifying) did not hold.")
        return 3
    ok = c_pass
    print(f"GAP C RESULT: {'PASS' if ok else 'FAIL'}"
          f"  (mask live={ok_b}, k3 fires={ok_a}, hybrid arm gate={c_pass})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
