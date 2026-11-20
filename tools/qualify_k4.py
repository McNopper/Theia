"""k=4 correctness gate - THEIA_SHIFT_MASK differential + noise envelope on hybrid_k4.

Closes the honest limit the k=4 OBSERVED receipt recorded (Theia/PLAN.md): "this proves
the k=4 BODY RUNS and is step-through observable; it does NOT yet prove the k=4 result
is correct (no Hyperion gate was applied to this fixture - its own noise envelope has
not been measured)".

Fixture: `hybrid_k4.scene.toml` - pair 1 (plate, slab) fails on sampled-lobe alpha,
pair 2 (slab, wall) fails on the delta transmission lobe, pair 3 (wall, floor)
qualifies => captureTarget 2 => kLevel 3 => k = 4, the body stage 2b built.

Arm semantics (read from gi.comp.slang dispatch: `kLevel == 2u || (kLevel >= 3u &&
(mask & 4u) != 0u)`):
  replay          mask 0 - no shift at all
  reconnection_k2 mask 1 - hybrid mode, bit2 clear: k=4 records fall back to replay
                          and k<=2 records take the k=2 body
  k4_fallback     mask 3 - hybrid + bit1, bit2 STILL clear: k=4 -> unbiased replay
  k4              mask 7 - hybrid + bit2: the N-hop body actually runs for k=4

Checks and what each proves:
  0. CONTENT PRECONDITION (exit 5): every frame non-degenerate - the strict-AND gate
     cannot catch two identical black images.
  B. mask REACHES estimator: replay(0) vs reconnection_k2(1) must differ - without a
     live knob the A outcome is uninterpretable (exit 4).
  A. k=4 FIRES: k4_fallback(3) vs k4(7) must differ - bit2 gates ONLY kLevel >= 3, so
     a difference proves a k=4 record was captured AND the N-hop body changed the
     result (exit 3 = premise failure, never a pass).
  C. k=4 arm CORRECT: mask 7 vs Hyperion within the OWNER's noise-envelope criterion
     (PLAN 0fbcb0b - the family gate thresholds sit below this fixture family's own
     noise; recorded there for hybrid_k3, applied here with a LIVE envelope measured
     in this run: ref seed17 vs ref seed18). All four metrics must hold, SSIM
     inverted (cross >= envelope). The mechanical strict-AND gate is reported as
     context only (expected to fail for every arm incl. ref-vs-ref, as recorded for
     hybrid_k3). The k4_fallback(3) arm is gated by the same criterion too: it is
     what a bit2-clear run produces, and if it were outside the envelope the mask7
     pass would be uninterpretable (exit 1).
  D. pairwise pixel deltas reported (no invented thresholds: differ = max|d| > 0,
     never file hashes - EXR headers carry metadata).

No new thresholds are introduced here: "differs" = PIXEL max|d| > 0; correctness =
the owner's recorded noise-envelope criterion verbatim with an envelope measured at
run time; the mechanical gate = family verbatim, context.

GPU discipline: all renders run SEQUENTIALLY inside this one process.
Exit codes: 0 pass, 1 envelope fail, 2 missing input/render, 3 k=4 never fires,
4 mask dead, 5 degenerate frame content.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Development\GitHub") / "Harmonia" / "tools"))
from compare_renders import load_exr

import numpy as np

GITHUB = Path(r"C:\Development\GitHub")
STAGED = (
    GITHUB
    / "Theia"
    / "build"
    / "sms-qualification"
    / "scenes"
    / "HYBRID_K4"
    / "hybrid_k4.scene.toml"
)
THEIA = GITHUB / "Theia" / "build" / "theia.exe"
HYPERION = GITHUB / "Hyperion" / "build" / "hyperion.exe"
COMPARE = GITHUB / "Harmonia" / "tools" / "compare_renders.py"

# arm label -> THEIA_SHIFT_MASK value (render order: cheapest risk first - the two
# arms checks B and A depend on come before the context arms).
ARMS: dict[str, int] = {"replay": 0, "reconnection_k2": 1, "k4_fallback": 3, "k4": 7}

# The four metrics of the owner's noise-envelope criterion (PLAN 0fbcb0b tables).
FOUR = ("mean_diff", "rel_mse", "ssim", "tm_mean")


def content_state(arr: np.ndarray) -> tuple[bool, str]:
    # float64: HDR outliers overflow float32 in std's square.
    a64 = arr.astype(np.float64)
    lo, hi, spread = float(a64.min()), float(a64.max()), float(a64.std())
    return hi > 0.0 and spread > 0.0, f"min {lo:.6f} max {hi:.6f} std {spread:.6f}"


def run_exe(
    label: str,
    exe: Path,
    args: list[str],
    out: Path,
    extra_env: dict[str, str] | None = None,
) -> float:
    log_path = out.with_suffix(".log")
    for stale in (out, log_path):
        if stale.exists():
            stale.unlink()
    env = dict(os.environ)
    env["HARMONIA_FRAME_TIMING"] = "1"
    if extra_env:
        env.update(extra_env)
    started = time.perf_counter()
    proc = subprocess.run(
        [str(exe), *args], capture_output=True, text=True, check=False, env=env
    )
    elapsed = time.perf_counter() - started
    text = proc.stdout + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    if "GPU error bits" in text or "VUID-" in text:
        raise SystemExit(f"{label}: GPU/validation failure; see {log_path.name}")
    if proc.returncode != 0 or not out.is_file():
        raise SystemExit(
            f"{label}: render failed (code {proc.returncode}); see {log_path.name}"
        )
    print(f"  rendered {label:22s} {elapsed:7.1f}s -> {out.name}", flush=True)
    return elapsed


def hyperion_args(seed: int, args: argparse.Namespace, out: Path) -> list[str]:
    return [
        "--scene",
        str(STAGED),
        "--sms-mode",
        "off",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--spp",
        str(args.spp),
        "--rng-seed",
        str(seed),
        "--validation",
        "--output",
        str(out),
    ]


def metrics(reference: Path, candidate: Path) -> dict[str, float | bool]:
    """The family comparator's four gated metrics + gate verdict, parsed verbatim."""
    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            str(reference),
            str(candidate),
            "--no-heatmap",
            "--gate",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    pats = {
        "mean_diff": r"(?m)^\s*mean_diff\s*:\s*([0-9.eE+-]+)",
        "rel_mse": r"(?m)^\s*Rel-MSE\s*:\s*([0-9.eE+-]+)",
        "ssim": r"(?m)^\s*SSIM\s*:\s*([0-9.eE+-]+)",
        "tm_mean": r"(?m)^\s*tm_mean_diff\s*:\s*([0-9.eE+-]+)",
    }
    vals: dict[str, float | bool] = {"gate": proc.returncode == 0}
    for key, pat in pats.items():
        m = re.search(pat, out)
        vals[key] = float(m.group(1)) if m else float("nan")
    return vals


def within_envelope(
    cross: dict[str, float | bool], env: dict[str, float | bool]
) -> bool:
    """Owner's criterion: cross <= ref self-noise on 4/4, SSIM inverted (higher better)."""
    return bool(
        cross["mean_diff"] <= env["mean_diff"]
        and cross["rel_mse"] <= env["rel_mse"]
        and cross["ssim"] >= env["ssim"]
        and cross["tm_mean"] <= env["tm_mean"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=GITHUB / "Theia" / "build" / "sms-qualification" / "gapk4gate",
    )
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--ref2-seed",
        type=int,
        default=18,
        help="second reference seed for the live self-noise envelope",
    )
    ap.add_argument("--sms-mode", default="local")
    ap.add_argument(
        "--skip-render", action="store_true", help="re-analyze EXRs already on disk"
    )
    args = ap.parse_args()

    for path, what in (
        (STAGED, "staged hybrid_k4 fixture"),
        (THEIA, "theia.exe"),
        (HYPERION, "hyperion.exe"),
        (COMPARE, "compare_renders.py"),
    ):
        if not path.is_file():
            print(f"missing {what}: {path}")
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    ref = args.out / "hyperion.exr"
    ref2 = args.out / f"hyperion_seed{args.ref2_seed}.exr"
    arms = {name: args.out / f"theia_mask{mask}.exr" for name, mask in ARMS.items()}

    if not args.skip_render:
        print(
            f"k=4 gate on hybrid_k4 (budget {args.width}x{args.height}, "
            f"{args.spp}spp / {args.frames}f, seed {args.seed}) - SEQUENTIAL\n"
        )
        run_exe(
            "hyperion reference", HYPERION, hyperion_args(args.seed, args, ref), ref
        )
        run_exe(
            f"hyperion ref2 seed{args.ref2_seed}",
            HYPERION,
            hyperion_args(args.ref2_seed, args, ref2),
            ref2,
        )
        for name, mask in ARMS.items():
            run_exe(
                f"theia mask={mask} {name}",
                THEIA,
                [
                    "--scene",
                    str(STAGED),
                    "--sms-mode",
                    args.sms_mode,
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--offscreen-frames",
                    str(args.frames),
                    "--rng-seed",
                    str(args.seed),
                    "--validation",
                    "--output",
                    str(arms[name]),
                ],
                arms[name],
                extra_env={"THEIA_SHIFT_MASK": str(mask)},
            )
        print()

    for path in (ref, ref2, *arms.values()):
        if not path.is_file():
            print(f"missing render: {path}")
            return 2

    print("=" * 74)
    print("k=4 correctness gate (hybrid_k4; closes the k=4 OBSERVED honest limit)")
    print("=" * 74)

    frames: dict[str, np.ndarray] = {
        "reference": load_exr(ref),
        "reference2": load_exr(ref2),
    }
    frames.update({name: load_exr(path) for name, path in arms.items()})

    print(f"{'frame':16s} {'min':>10s} {'max':>10s} {'std':>10s}  state")
    degenerate = []
    for name, arr in frames.items():
        ok_c, detail = content_state(arr)
        print(f"{name:16s} {detail}  {'OK' if ok_c else 'DEGENERATE'}")
        if not ok_c:
            degenerate.append(name)
    if degenerate:
        print(f"\nRESULT: FAIL - degenerate frame(s): {', '.join(degenerate)}")
        return 5

    def delta(a: str, b: str) -> float:
        return float(np.abs(frames[a] - frames[b]).max())

    # B first: without a live knob, A's outcome is uninterpretable.
    d_b = delta("replay", "reconnection_k2")
    ok_b = d_b > 0.0
    print(
        f"\nB. mask REACHES estimator : replay(0) vs reconnection_k2(1) "
        f"max|d| = {d_b:.6f}  -> "
        f"{'LIVE' if ok_b else 'DEAD KNOB (differential uninterpretable)'}"
    )

    d_a = delta("k4_fallback", "k4")
    ok_a = d_a > 0.0
    print(
        f"A. k=4 body FIRES         : k4_fallback(3) vs k4(7) "
        f"max|d| = {d_a:.6f}  -> "
        f"{'FIRES (bit2-gated N-hop body changed the result)' if ok_a else 'NEVER FIRES (Gap A premise fails)'}"
    )

    # C. owner's noise-envelope criterion (PLAN 0fbcb0b), envelope measured LIVE.
    env = metrics(ref, ref2)
    cross = {name: metrics(ref, path) for name, path in arms.items()}

    print("\nC. owner's noise-envelope criterion (PLAN 0fbcb0b): cross <= Hyperion's")
    print("   own self-noise on all four metrics (SSIM inverted). Envelope measured")
    print(f"   live: ref(seed {args.seed}) vs ref(seed {args.ref2_seed}).\n")
    print(
        f"   {'metric':10s} {'envelope':>12s} {'k4':>12s} {'k4_fallback':>12s}"
        f" {'k2_arm':>12s} {'replay':>12s}"
    )
    for key in FOUR:
        print(
            f"   {key:10s} {float(env[key]):12.4f} "
            + " ".join(
                f"{float(cross[n][key]):12.4f}"
                for n in ("k4", "k4_fallback", "reconnection_k2", "replay")
            )
        )
    k4_ok = within_envelope(cross["k4"], env)
    fb_ok = within_envelope(cross["k4_fallback"], env)
    print(f"\n   k4 arm within envelope on 4/4      : {'YES' if k4_ok else 'NO'}")
    print(
        f"   k4_fallback arm within envelope 4/4: {'YES' if fb_ok else 'NO'}"
        f"  (gated too - if it failed, the k4 pass would be uninterpretable)"
    )
    print(
        "   context - other arms: "
        + ", ".join(
            f"{n}={'YES' if within_envelope(cross[n], env) else 'NO'}"
            for n in ("reconnection_k2", "replay")
        )
    )

    print("\n   context - family strict-AND gate (mechanical, expected FAIL for every")
    print("   arm incl. ref-vs-ref, as recorded for hybrid_k3: condition C):")
    for name in ARMS:
        print(f"   {name:16s} {'PASS' if bool(cross[name]['gate']) else 'FAIL'}")

    print("\n   pairwise pixel max|d| (1/255-scale linear EXR):")
    pairs = (
        ("replay", "reconnection_k2"),
        ("k4_fallback", "k4"),
        ("reconnection_k2", "k4"),
        ("replay", "k4"),
    )
    for a, b in pairs:
        print(f"   {a:16s} vs {b:14s} {delta(a, b):.6f}")

    print()
    if not ok_b:
        print("RESULT: FAIL - THEIA_SHIFT_MASK does not reach the estimator, so no")
        print("k=4 differential observation is possible.")
        return 4
    if not ok_a:
        print("RESULT: FAIL - bit2 made no difference: no k=4 record was captured on")
        print("this fixture, or the dispatched body is dead. Not a pass.")
        return 3
    ok = k4_ok and fb_ok
    print(
        f"GAP A k=4 GATE: {'PASS' if ok else 'FAIL'}"
        f"  (mask live={ok_b}, k4 fires={ok_a}, k4 envelope={k4_ok},"
        f" fallback envelope={fb_ok})"
    )
    if ok:
        print("The k=4 N-hop body runs AND its result sits inside Hyperion's own")
        print("sampling noise on all four metrics - the k=4 OBSERVED honest limit")
        print("(no correctness gate applied) is now closed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
