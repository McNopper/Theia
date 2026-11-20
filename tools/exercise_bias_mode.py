"""Exercise biasCorrectionMode levels 0/1/2 via THEIA_BIAS_CORRECTION renders.

Closes Theia/PLAN.md's recorded gap on the biasCorrectionMode receipt (700e479):
"Level 2 (ray_traced) is NOT yet exercised - it needs THEIA_BIAS_CORRECTION=2
renders and is not claimed as verified."

THE THREE RTXDI MODES, AS READ FROM THE SOURCE while building this instrument
(NVIDIA-RTX/RTXDI commit 0cfd439: rtxdi-sdk/include/rtxdi/DIResamplingFunctions.hlsli
and DIReservoir.hlsli) - not from prose:
  OFF        : the merge ALWAYS streams with the donated sample's target pdf
               re-evaluated at the CURRENT surface (weightAtCurrent is computed
               unconditionally, in every mode - plain RIS). Only the FINALIZE
               differs: FinalizeResampling(state, 1.0, state.M), and since
               FinalizeResampling divides by reservoir.targetPdf * normalizationDenominator,
               OFF gives W = w_sum / (p_hat * M) - ReSTIR Eq (6), "explicitly biased".
  BASIC      : same merge; finalize (pi, piSum) where piSum = p_hat(sel)@current*M_cur
               + p_hat(sel)@source*M_prev - the GRIS normalization.
  RAY_TRACED : BASIC + one conservative-visibility ray per source; a source that could
               not have generated the sample gets temporalP = 0.

WHAT OUR SHIPPED KNOB ACTUALLY COMPUTES (gi.comp restirDiDirect):
  level 1 (default): merge w = p_hat_cur * W_src * M_src, finalize W = w_sum/(M*p_hat)
               -> this IS RTXDI's OFF math (correspondence above), i.e. today's
               behaviour is the OFF formula, not the BASIC piSum one (the receipt's
               "BASIC-equivalent" wording is corrected in PLAN).
  level 2     : level 1 + sourceConservativeVisible() per merged temporal/spatial
               source - the VISIBILITY half of RAY_TRACED (the piSum half is absent;
               recorded as the second gap when this run was taken).
  level 0     : pMerge forced to 1.0 - drops p_hat from the MERGE weight, which no
               reference algorithm does (RTXDI keeps weightAtCurrent in OFF too) and
               which compounds w_sum_prev/p_hat_prev every temporal frame. MEASURED
               EXPLODED (see FINDING below) - recorded in PLAN + owner decision
               pending; does NOT gate this instrument's level-2 receipt.

Checks and what each proves:
  0. CONTENT PRECONDITION (exit 5): every frame non-degenerate - the strict-AND
     gate cannot catch two identical black images.
  A. PARSE REACHED (exit 4): each render's own log line THEIA_BIAS_CORRECTION = <n>
     proves the env var parsed and clamped, per level.
  B. CONTROL: default (env unset) vs explicit 1 must be pixel BIT-IDENTICAL -
     proves the shipped default is level 1 and that this instrument can see a
     zero when it means zero (exit 6 on failure; everything downstream of a
     failed control is uninterpretable).
  C. LEVEL 2 FIRES (exit 3): =1 vs =2 must give pixel max|d| > 0 - the visibility
     half actually rejects non-generatable donors on this fixture. Identical would
     mean compiled-and-dispatched-but-unrun (the k=4 situation), reported as such.
  D. LEVEL 2 WITHIN THE NOISE ENVELOPE (exit 1): the owner's recorded acceptance
     basis for THIS fixture (PLAN 0fbcb0b) - the family gate thresholds sit below
     Hyperion's OWN sampling noise here (recorded ref-vs-ref mean_diff 2342 > 120
     at 256spp, gate unmeetable for ANY arm incl. the shift-free replay), so the
     criterion is: the cross gap (ref vs arm) must be <= Hyperion's self-noise on
     all four metrics, SSIM inverted (cross >= envelope). The envelope is measured
     LIVE in this run: ref(seed) vs ref(seed2). The mechanical strict-AND gate is
     still reported per arm as context - it is expected to FAIL for every arm and
     that failure is the recorded condition-C property of the fixture, not a level-2
     property. Head-to-head level2 vs level1 is reported, not gated (the visibility
     gate can trade bias for variance; no invented expectation).

No invented thresholds: "differs" = PIXEL max|d| > 0 (never file hashes - EXR
headers carry metadata); correctness = the owner's noise-envelope criterion with a
live-measured envelope; the mechanical gate = family verbatim, context only.

GPU discipline: all renders run SEQUENTIALLY inside this one process.
Exit codes: 0 level-2 receipt complete, 1 level 2 outside the noise envelope,
2 missing input/render, 3 level 2 never fires, 4 env parse missing from a log,
5 degenerate frame content, 6 control failed (default != level 1).
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
    / "HYBRID"
    / "hybrid_k3.scene.toml"
)
THEIA = GITHUB / "Theia" / "build" / "theia.exe"
HYPERION = GITHUB / "Hyperion" / "build" / "hyperion.exe"
COMPARE = GITHUB / "Harmonia" / "tools" / "compare_renders.py"

# arm label -> THEIA_BIAS_CORRECTION value; None = env UNSET (the shipped default path).
ARMS: dict[str, int | None] = {
    "default_unset": None,
    "level1": 1,
    "level2": 2,
    "level0": 0,
}

# The four metrics of the owner's noise-envelope criterion (PLAN 0fbcb0b tables).
FOUR = ("mean_diff", "rel_mse", "ssim", "tm_mean")


def content_state(arr: np.ndarray) -> tuple[bool, str]:
    # float64: level 0's exploded values (1e34) overflow float32 in std's square.
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
        default=GITHUB / "Theia" / "build" / "sms-qualification" / "biasmode",
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
        (STAGED, "staged fixture"),
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
    arms = {name: args.out / f"theia_bias_{name}.exr" for name in ARMS}

    if not args.skip_render:
        print(
            f"biasCorrectionMode exercise (budget {args.width}x{args.height}, "
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
        for name, level in ARMS.items():
            env = {} if level is None else {"THEIA_BIAS_CORRECTION": str(level)}
            run_exe(
                f"theia {name}",
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
                extra_env=env,
            )
        print()

    for path in (ref, ref2, *arms.values()):
        if not path.is_file():
            print(f"missing render: {path}")
            return 2

    print("=" * 74)
    print("biasCorrectionMode exercise (PLAN.md receipt owed since 700e479)")
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

    # A. parse reached: each explicit level must appear in its own render log.
    print("\nA. env parse per level (from each render's own log):")
    parse_ok = True
    for name, level in ARMS.items():
        if level is None:
            print("   default_unset    (no env, no log line expected)")
            continue
        log = arms[name].with_suffix(".log").read_text(encoding="utf-8")
        want = f"THEIA_BIAS_CORRECTION = {level}"
        hit = want in log
        parse_ok = parse_ok and hit
        print(f"   {name:16s} {'FOUND' if hit else 'MISSING'}: {want!r}")
    if not parse_ok:
        print("\nRESULT: FAIL - THEIA_BIAS_CORRECTION did not reach the binary")
        return 4

    # B. control: default must equal explicit 1 bit-for-bit (a zero that means zero).
    def delta(a: str, b: str) -> float:
        return float(np.abs(frames[a] - frames[b]).max())

    d_b = delta("default_unset", "level1")
    default_ok = d_b == 0.0
    print(
        f"\nB. control default == level1 : max|d| = {d_b:.7f}  -> "
        f"{'bit-identical (default neutrality re-proven on this build)' if default_ok else 'DIVERGES - default is NOT level 1'}"
    )
    if not default_ok:
        print(
            "\nRESULT: FAIL - control broken; downstream observations uninterpretable"
        )
        return 6

    # C. the receipt itself: level 2 must actually change something.
    d_c = delta("level1", "level2")
    fires = d_c > 0.0
    print(
        f"C. level 2 FIRES            : level1 vs level2 max|d| = {d_c:.7f}  -> "
        f"{'LIVE (visibility half rejects non-generatable donors)' if fires else 'NOT-FIRED (premise failure, not a pass)'}"
    )
    d_0 = delta("level1", "level0")
    print(
        f"   context: level0 vs level1 max|d| = {d_0:.7f}  "
        f"({'differs' if d_0 > 0 else 'IDENTICAL'})"
    )

    # D. owner's noise-envelope criterion (PLAN 0fbcb0b), envelope measured LIVE.
    env = metrics(ref, ref2)
    cross = {name: metrics(ref, path) for name, path in arms.items()}

    print("\nD. owner's noise-envelope criterion (PLAN 0fbcb0b): cross <= Hyperion's")
    print("   own self-noise on all four metrics (SSIM inverted). Envelope measured")
    print(f"   live: ref(seed {args.seed}) vs ref(seed {args.ref2_seed}).\n")
    print(
        f"   {'metric':10s} {'envelope':>12s} {'level2':>12s} {'level1':>12s}"
        f" {'default':>12s} {'level0':>12s}"
    )
    for key in FOUR:
        print(
            f"   {key:10s} {float(env[key]):12.4f} "
            + " ".join(
                f"{float(cross[n][key]):12.4f}"
                for n in ("level2", "level1", "default_unset", "level0")
            )
        )
    d_pass = within_envelope(cross["level2"], env)
    print(
        f"\n   level2 within envelope on 4/4: {'YES' if d_pass else 'NO'}   "
        f"(level1 reference point: {'YES' if within_envelope(cross['level1'], env) else 'NO'}, "
        f"default: {'YES' if within_envelope(cross['default_unset'], env) else 'NO'})"
    )
    print("   level2 vs level1 head-to-head: reported above, NOT gated (the visibility")
    print("   gate can trade bias for variance; no invented expectation).")

    print("\n   context - family strict-AND gate (mechanical, expected FAIL for every")
    print("   arm on this fixture incl. the shipped default: recorded condition C,")
    print("   ref-vs-ref also fails it):")
    for name in (*ARMS,):
        print(f"   {name:18s} {'PASS' if bool(cross[name]['gate']) else 'FAIL'}")

    # FINDING: level 0's structural explosion (reported loudly, not gated here).
    l0 = frames["level0"].astype(np.float64)
    l1 = frames["level1"].astype(np.float64)
    rf = frames["reference"].astype(np.float64)
    per_frame = (float(l0.max()) / max(float(l1.max()), 1e-30)) ** (1.0 / args.frames)
    print("\nFINDING - level 0 output is EXPLODED (structural, not a threshold call):")
    print(
        f"   max level0 {float(l0.max()):.6e}  vs level1 {float(l1.max()):.6e}"
        f"  vs ref {float(rf.max()):.6e}"
    )
    print(
        f"   implied per-frame growth over {args.frames}f: {per_frame:.3f}x"
        f"  (level0 vs level1 differing pixels: "
        f"{int((np.abs(l0 - l1).max(axis=2) > 0).sum())}/{l1.shape[0] * l1.shape[1]})"
    )
    print(
        "   mechanism: pMerge=1.0 drops p_hat from the merge weight, so each temporal"
    )
    print(
        "   hop adds w_sum_prev/p_hat_prev with nothing to cancel it; RTXDI's OFF mode"
    )
    print(
        "   KEEPS weightAtCurrent in the merge (source read, see module header). This"
    )
    print("   mode matches no reference algorithm. NOT gated here (level-2 receipt):")
    print("   recorded in PLAN + put to the owner for a knob-semantics decision.")

    print()
    if not fires:
        print("RESULT: FAIL - level 2 never fires on this fixture, so the ray_traced")
        print(
            "branch is compiled and dispatched but UNOBSERVED. Not claimed as exercised."
        )
        return 3
    if not d_pass:
        print(
            "RESULT: FAIL - level 2 cross gap exceeds Hyperion's own self-noise on at"
        )
        print("least one of the four metrics (owner's criterion, PLAN 0fbcb0b).")
        return 1
    print(
        f"RESULT: LEVEL-2 RECEIPT PASS  (parse={parse_ok}, control=default==level1, "
        f"fires max|d|={d_c:.7f}, envelope 4/4={d_pass})"
    )
    print("       LEVEL-0 FINDING: EXPLODED - structural defect recorded, owner")
    print("       decision pending (does not gate this receipt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
