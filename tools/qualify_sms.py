"""GI-SMS #2 qualification runner (machine-local; builds are NOT modified).

Stages per Harmonia/PLAN.md "GI-SMS #2 — bounded finish plan":
  smoke        real-Application mode checks on C1 (off/local/reuse + Hyperion off)
  reference    Hyperion ordinary-PT (SMS-off) convergence ladder with split-seed
               uncertainty gate (advance only when ROI uncertainty <= 0.5%)
  correctness  Theia local/reuse at 512/2048 frames x 4 seeds; pooled ROI RGB
               energy vs reference within 2%; 2048-frame pooled MSE < 512
  performance  equal-time quality at matched elapsed time (timing pilots first;
               never stop sample-dependent work at a time limit)
  stability    determinism (same seed twice -> identical EXR), seed variance

Artifacts: Theia/build/sms-qualification/ (git-ignored). Every run records
commit SHAs, input hashes, options, timings and exit codes in run_manifest.json.
This runner is a qualification harness only — it never edits source or shaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent.parent / "Harmonia" / "tools"))
from compare_renders import load_exr  # noqa: E402  (family parity reader)

GITHUB = TOOLS.parent.parent
ASSETS = GITHUB / "Harmonia" / "tests" / "assets" / "sms"
OUT = GITHUB / "Theia" / "build" / "sms-qualification"
MATERIAL_LINE = 'material_libraries = ["sms_qualification.materials.toml"]'

SEEDS = [17, 91, 1234, 2026]
REFERENCE_SPP_LADDER = [4096, 16384, 65536]
CANDIDATE_FRAMES = [512, 2048]
REF_UNCERTAINTY = 0.005
ENERGY_TOLERANCE = 0.02
TIMING_PILOTS = 3
TIMING_PILOT_FRAMES = 64
TARGET_SECONDS = [2.0, 8.0]
RESOLUTION = ["320", "240"]

CASES = {
    "C1": {
        "scene": "c1_curved_glass.scene.toml",
        "roi": [0.38, 0.70, 0.62, 0.84],  # frozen: Hyperion local-vs-off caustic bbox + margin
    },
    "C2": {
        "scene": "c2_absorption_prefix.scene.toml",
        "roi": [0.38, 0.70, 0.62, 0.84],
    },
    "C3": {
        "scene": "c3_boundary_mixed.scene.toml",
        "roi": [0.25, 0.60, 0.75, 0.85],
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def prepare_run_dir(case: str) -> Path:
    """Copy scenes + OBJs into the run dir; rewrite asset references absolute.

    Scene-relative paths are NOT used by the loader (material libraries and mesh
    paths resolve against the application's assetsDir), so the runner copies the
    whole fixture set and rewrites the exact committed reference lines to the
    copied absolute paths.
    """
    run_dir = OUT / "scenes" / case
    run_dir.mkdir(parents=True, exist_ok=True)
    materials = ASSETS / "sms_qualification.materials.toml"
    shutil.copy2(materials, run_dir / materials.name)
    obj_targets: dict[str, str] = {}
    for obj in sorted(ASSETS.glob("*.obj")):
        shutil.copy2(obj, run_dir / obj.name)
        obj_targets[obj.name] = (run_dir / obj.name).as_posix()
    source = ASSETS / CASES[case]["scene"]
    text = source.read_text(encoding="utf-8")
    if MATERIAL_LINE not in text:
        raise SystemExit(f"{source}: material line missing; runner rewrite contract broken")
    text = text.replace(MATERIAL_LINE, f'material_libraries = ["{(run_dir / materials.name).as_posix()}"]')
    for name, absolute in obj_targets.items():
        needle = f'path = "{name}"'
        if needle in text:
            text = text.replace(needle, f'path = "{absolute}"')
    target = run_dir / source.name
    target.write_text(text, encoding="utf-8")
    return target


def run_exe(label: str, exe: Path, args: list[str]) -> tuple[int, Path, float]:
    exe_log = OUT / f"{label}.log"
    started = time.perf_counter()
    process = subprocess.run([str(exe), *args], capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    exe_log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        log(f"{label}: EXIT {process.returncode} (see {exe_log.name})")
    if "GPU error bits" in process.stdout + process.stderr:
        raise SystemExit(f"{label}: logical frame failed (GPU error bits)")
    if "VUID-" in process.stdout + process.stderr:
        raise SystemExit(f"{label}: validation diagnostics present")
    return process.returncode, exe_log, elapsed


def render_hyperion(label: str, scene: Path, spp: int, seed: int) -> tuple[Path, float]:
    out = OUT / f"{label}.exr"
    exe = GITHUB / "Hyperion" / "build" / "hyperion.exe"
    code, _, elapsed = run_exe(label, exe, [
        "--scene", str(scene), "--sms-mode", "off",
        "--width", RESOLUTION[0], "--height", RESOLUTION[1],
        "--spp", str(spp), "--rng-seed", str(seed),
        "--validation", "--output", str(out)])
    if code != 0 or not out.exists():
        raise SystemExit(f"{label}: Hyperion reference render failed")
    return out, elapsed


def render_theia(label: str, scene: Path, mode: str, frames: int, seed: int) -> tuple[Path, float]:
    out = OUT / f"{label}.exr"
    exe = GITHUB / "Theia" / "build" / "theia.exe"
    code, _, elapsed = run_exe(label, exe, [
        "--scene", str(scene), "--sms-mode", mode,
        "--width", RESOLUTION[0], "--height", RESOLUTION[1],
        "--offscreen-frames", str(frames), "--rng-seed", str(seed),
        "--validation", "--output", str(out)])
    if code != 0 or not out.exists():
        raise SystemExit(f"{label}: Theia {mode} render failed")
    return out, elapsed


def roi_energy(path: Path, roi: list[float]) -> np.ndarray:
    image = load_exr(path)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = roi
    box = image[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)]
    return box.reshape(-1, 3).sum(axis=0, dtype=np.float64)


def roi_mse(reference: Path, candidate: Path, roi: list[float]) -> float:
    ref = load_exr(reference)
    cand = load_exr(candidate)
    height, width = ref.shape[:2]
    x0, y0, x1, y1 = roi
    box = (slice(int(y0 * height), int(y1 * height)), slice(int(x0 * width), int(x1 * width)))
    difference = ref[box].astype(np.float64) - cand[box].astype(np.float64)
    return float(np.mean(difference * difference))


def record(entry: dict) -> None:
    manifest = OUT / "run_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    entries.append(entry)
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def stage_smoke(case: str) -> None:
    scene = prepare_run_dir(case)
    for mode in ("off", "local", "reuse"):
        out, _ = render_theia(f"smoke_theia_{case}_{mode}", scene, mode, 2, SEEDS[0])
        png = out.with_suffix(".png")
        if not png.exists():
            raise SystemExit(f"smoke {mode}: PNG missing")
        log(f"smoke theia {case} {mode}: ok")
    out, _ = render_hyperion(f"smoke_hyperion_{case}", scene, 2, SEEDS[0])
    log(f"smoke hyperion {case}: ok")
    record({"stage": "smoke", "case": case, "theia_head": git_head(GITHUB / "Theia"),
            "harmonia_head": git_head(GITHUB / "Harmonia"),
            "hyperion_head": git_head(GITHUB / "Hyperion"),
            "scene_sha": sha256(scene), "materials_sha": sha256(ASSETS / "sms_qualification.materials.toml")})


def stage_reference(case: str) -> Path:
    scene = prepare_run_dir(case)
    roi = CASES[case]["roi"]
    for spp in REFERENCE_SPP_LADDER:
        halves = []
        for index, seed in enumerate(SEEDS[:2] if spp == REFERENCE_SPP_LADDER[0] else SEEDS[2:]):
            path, _ = render_hyperion(f"ref_{case}_{spp}_{seed}", scene, spp, seed)
            halves.append(roi_energy(path, roi))
        relative = float(np.max(np.abs(halves[0] - halves[1]) / np.maximum(0.5 * (halves[0] + halves[1]), 1e-30)))
        log(f"reference {case} spp={spp}: split-seed ROI uncertainty {relative:.5f}")
        if relative <= REF_UNCERTAINTY:
            final, elapsed = render_hyperion(f"ref_{case}_final_{spp}", scene, spp, SEEDS[0])
            log(f"reference {case}: accepted at spp={spp} ({elapsed:.1f}s)")
            record({"stage": "reference", "case": case, "spp": spp, "uncertainty": relative})
            return final
    raise SystemExit(f"reference {case}: ladder exhausted above {REF_UNCERTAINTY:.3f}; reference-budget blocker")


def stage_correctness(case: str, reference: Path) -> None:
    scene = prepare_run_dir(case)
    roi = CASES[case]["roi"]
    expected = roi_energy(reference, roi)
    results: dict[tuple[str, int], list[float]] = {}
    for mode in ("local", "reuse"):
        for frames in CANDIDATE_FRAMES:
            pooled = np.zeros(3, dtype=np.float64)
            for seed in SEEDS:
                path, _ = render_theia(f"cand_{case}_{mode}_{frames}_{seed}", scene, mode, frames, seed)
                pooled += roi_energy(path, roi)
            results[(mode, frames)] = (pooled / len(SEEDS) / np.maximum(expected, 1e-30)).tolist()
            ratios = results[(mode, frames)]
            log(f"correctness {case} {mode} @{frames}f: ROI RGB energy ratios "
                + " ".join(f"{ratio:.5f}" for ratio in ratios))
            if max(abs(ratio - 1.0) for ratio in ratios) > ENERGY_TOLERANCE:
                raise SystemExit(f"correctness {case} {mode} @{frames}f: RGB energy outside 2%")
    mse_512 = roi_mse(reference, OUT / f"cand_{case}_local_512_{SEEDS[0]}.exr", roi)
    mse_2048 = roi_mse(reference, OUT / f"cand_{case}_local_2048_{SEEDS[0]}.exr", roi)
    log(f"correctness {case}: pooled local MSE 512={mse_512:.6g} 2048={mse_2048:.6g}")
    if mse_2048 >= mse_512:
        raise SystemExit("correctness: 2048-frame error did not improve over 512 frames")
    record({"stage": "correctness", "case": case, "ratios": {str(k): v for k, v in results.items()},
            "mse": {"512": mse_512, "2048": mse_2048}})


def stage_performance(case: str, reference: Path) -> None:
    scene = prepare_run_dir(case)
    timings: dict[str, float] = {}
    for mode in ("local", "reuse"):
        samples = []
        for pilot in range(TIMING_PILOTS):
            _, _, elapsed = render_theia(f"pilot_{case}_{mode}_{pilot}", scene, mode, TIMING_PILOT_FRAMES, SEEDS[pilot])
            samples.append(elapsed)
        timings[mode] = float(np.median(samples))
        log(f"pilot {case} {mode}: median {timings[mode]:.2f}s for {TIMING_PILOT_FRAMES}f ({samples})")
    chosen: dict[str, dict[str, int]] = {}
    for mode in ("local", "reuse"):
        per_frame = timings[mode] / TIMING_PILOT_FRAMES
        chosen[mode] = {f"{target:.0f}s": max(1, round(target / per_frame)) for target in TARGET_SECONDS}
        log(f"chosen frames {case} {mode}: {chosen[mode]}")
    mse: dict[str, dict[str, float]] = {mode: {} for mode in ("local", "reuse")}
    roi = CASES[case]["roi"]
    for mode in ("local", "reuse"):
        for target, frames in chosen[mode].items():
            elapsed_total = 0.0
            pooled = np.zeros(3, dtype=np.float64)
            for seed in SEEDS:
                path, elapsed = render_theia(f"eqt_{case}_{mode}_{target}_{seed}", scene, mode, frames, seed)
                elapsed_total += elapsed
                pooled += roi_energy(path, roi)
            expected = roi_energy(reference, roi)
            mse[mode][target] = roi_mse(reference, OUT / f"eqt_{case}_{mode}_{target}_{SEEDS[0]}.exr", roi)
            ratios = (pooled / len(SEEDS) / np.maximum(expected, 1e-30)).tolist()
            log(f"equal-time {case} {mode} @{target}: ratios " + " ".join(f"{r:.5f}" for r in ratios)
                + f" roiMSE={mse[mode][target]:.6g} total={elapsed_total:.1f}s")
    for target in chosen["local"]:
        ratio = mse["reuse"][target] / max(mse["local"][target], 1e-30)
        log(f"equal-time {case} @{target}: reuse/local ROI MSE ratio {ratio:.4f}")
    record({"stage": "performance", "case": case, "pilots": timings, "frames": chosen, "mse": mse})


def stage_stability(case: str) -> None:
    scene = prepare_run_dir(case)
    first, _ = render_theia(f"stab_{case}_a", scene, "reuse", 64, SEEDS[0])
    second, _ = render_theia(f"stab_{case}_b", scene, "reuse", 64, SEEDS[0])
    if sha256(first) != sha256(second):
        raise SystemExit("stability: same-seed captures are not byte-identical")
    log(f"stability {case}: deterministic (identical EXR bytes)")
    energies = []
    for seed in SEEDS:
        path, _ = render_theia(f"stab_{case}_{seed}", scene, "reuse", 64, seed)
        energies.append(roi_energy(path, CASES[case]["roi"]))
    spread = float(np.max(np.abs(np.array(energies) - np.mean(energies, axis=0)) /
                          np.maximum(np.mean(energies, axis=0), 1e-30)))
    log(f"stability {case}: per-seed ROI energy spread {spread:.4f}")
    record({"stage": "stability", "case": case, "seed_spread": spread})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["smoke", "reference", "correctness", "performance", "stability"])
    parser.add_argument("--case", default="C1", choices=sorted(CASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        stage_smoke(args.case)
        return 0
    if args.stage == "reference":
        stage_reference(args.case)
        return 0
    manifest = OUT / "run_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    references = {entry["case"]: OUT / f"ref_{entry['case']}_final_{entry['spp']}.exr"
                  for entry in entries if entry["stage"] == "reference"}
    if args.case not in references:
        raise SystemExit(f"run --stage reference --case {args.case} first")
    if args.stage == "correctness":
        stage_correctness(args.case, references[args.case])
    elif args.stage == "performance":
        stage_performance(args.case, references[args.case])
    elif args.stage == "stability":
        stage_stability(args.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
