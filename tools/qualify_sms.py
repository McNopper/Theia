"""Machine-local GI-SMS qualification runner; never edits source or shaders.

Stages: smoke, reference, correctness, performance, stability. Every stage fails
closed on nonfinite images, stale/missing exports, invalid reference energy,
wrong dimensions or unavailable per-frame completion timing. Artifacts and full
provenance live in ignored Theia/build/sms-qualification, never in the gallery.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent.parent / "Harmonia" / "tools"))
from compare_renders import load_exr  # noqa: E402

GITHUB = TOOLS.parent.parent
ASSETS = GITHUB / "Harmonia" / "tests" / "assets" / "sms"
OUT = GITHUB / "Theia" / "build" / "sms-qualification"
MATERIAL_LINE = 'material_libraries = ["sms_qualification.materials.toml"]'
SEEDS = [17, 91, 1234, 2026]
REFERENCE_SPP_LADDER = [4096, 16384, 65536]
# Correctness ladder: 512/2048 are recorded data points only; the bias gate
# applies at the highest count. Heavy-tailed caustic estimators have per-seed
# ROI-energy spreads of ~15% at 2048 frames (pooled 4-seed standard error
# ~7.5%), so a fixed +-2% gate at 512/2048 measures tail luck, not bias.
# The measured spread decays like 1/N (single dominant firefly per seed); at
# 8192 frames the pooled standard error is ~1-1.5%, which makes the same 2%
# tolerance a meaningful bias gate. PSMS-ReSTIR (Hong et al. 2025) validates
# with RMSE at equal time, not fixed-frame energy gates; the equal-time
# quality criterion is the performance stage below.
CANDIDATE_FRAMES = [512, 2048, 8192]
BIAS_GATE_FRAMES = 8192
REF_UNCERTAINTY = 0.005
ENERGY_TOLERANCE = 0.02
TIMING_PILOTS = 3
TIMING_PILOT_FRAMES = 64
TARGET_SECONDS = [2.0, 8.0]
TIME_TOLERANCE = 0.05
RESOLUTION = ["320", "240"]
CASES = {
    "C1": {"scene": "c1_curved_glass.scene.toml", "roi": [0.38, 0.70, 0.62, 0.84]},
    "C2": {"scene": "c2_absorption_prefix.scene.toml", "roi": [0.38, 0.70, 0.62, 0.84]},
    "C3": {"scene": "c3_boundary_mixed.scene.toml", "roi": [0.25, 0.60, 0.75, 0.85]},
}
FRAME_RE = re.compile(r"frame_timing_ms:(\d+):([0-9]+(?:\.[0-9]+)?)")


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str | bytes) -> str:
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"git provenance failed for {repo}: {' '.join(args)}")
    return result.stdout.strip()


def shader_hash(repo: str) -> str:
    directory = GITHUB / repo / "build" / "shaders"
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.spv")):
        digest.update(path.relative_to(GITHUB).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(case: str) -> dict:
    repos = {}
    for name in ("Harmonia", "Hyperion", "Theia"):
        repo = GITHUB / name
        untracked = git(repo, "ls-files", "--others", "--exclude-standard")
        untracked_hashes = []
        for relative in untracked.splitlines():
            path = repo / relative
            if path.is_file():
                untracked_hashes.append([relative, sha256(path)])
        repos[name] = {
            "head": git(repo, "rev-parse", "HEAD"),
            "tracked_diff_sha256": sha256_text(git(repo, "diff", "--binary", "HEAD")),
            "untracked_hashes": sorted(untracked_hashes),
        }
        executable = repo / "build" / f"{name.lower()}.exe"
        if executable.is_file():
            repos[name]["executable_sha256"] = sha256(executable)
        repos[name]["shader_bundle_sha256"] = shader_hash(name)
    inputs = {}
    for path in sorted(ASSETS.iterdir()):
        if path.is_file():
            inputs[path.name] = sha256(path)
    return {"case": case, "runner_sha256": sha256(Path(__file__).resolve()), "repos": repos, "inputs": inputs}


def prepare_run_dir(case: str) -> Path:
    run_dir = OUT / "scenes" / case
    run_dir.mkdir(parents=True, exist_ok=True)
    materials = ASSETS / "sms_qualification.materials.toml"
    shutil.copy2(materials, run_dir / materials.name)
    obj_targets = {}
    for obj in sorted(ASSETS.glob("*.obj")):
        shutil.copy2(obj, run_dir / obj.name)
        obj_targets[obj.name] = (run_dir / obj.name).as_posix()
    source = ASSETS / CASES[case]["scene"]
    text = source.read_text(encoding="utf-8")
    if MATERIAL_LINE not in text:
        raise SystemExit(f"{source}: material reference contract broken")
    text = text.replace(MATERIAL_LINE, f'material_libraries = ["{(run_dir / materials.name).as_posix()}"]')
    for name, absolute in obj_targets.items():
        text = text.replace(f'path = "{name}"', f'path = "{absolute}"')
    target = run_dir / source.name
    target.write_text(text, encoding="utf-8")
    return target


def run_exe(label: str, exe: Path, args: list[str]) -> tuple[int, Path, float]:
    exe_log = OUT / f"{label}.log"
    started = time.perf_counter()
    env = dict(os.environ)
    env["HARMONIA_FRAME_TIMING"] = "1"
    process = subprocess.run([str(exe), *args], capture_output=True, text=True, check=False, env=env)
    elapsed = time.perf_counter() - started
    text = process.stdout + process.stderr
    exe_log.write_text(text, encoding="utf-8")
    if "GPU error bits" in text or "VUID-" in text:
        raise SystemExit(f"{label}: GPU/validation failure; see {exe_log.name}")
    return process.returncode, exe_log, elapsed


def frame_seconds(log_path: Path, expected_frames: int) -> float:
    if expected_frames < 1:
        raise SystemExit("A positive completed-frame count is required")
    values = FRAME_RE.findall(log_path.read_text(encoding="utf-8"))
    if len(values) != expected_frames or [int(index) for index, _ in values] != list(range(expected_frames)):
        raise SystemExit(f"{log_path.name}: incomplete per-frame timing evidence")
    milliseconds = np.array([float(value) for _, value in values], dtype=np.float64)
    if not np.isfinite(milliseconds).all() or np.any(milliseconds <= 0.0):
        raise SystemExit(f"{log_path.name}: nonfinite/nonpositive frame time")
    return float(milliseconds.sum() / 1000.0)


def read_checked(path: Path, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"Missing fresh EXR: {path}")
    image = load_exr(path)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise SystemExit(f"Invalid image dimensions: {path}")
    if expected_shape is not None and image.shape != expected_shape:
        raise SystemExit(f"Mismatched image dimensions: {path}")
    if not np.isfinite(image).all():
        raise SystemExit(f"Nonfinite pixel in {path}")
    return image


def roi_slice(roi: list[float], shape: tuple[int, ...]) -> tuple[slice, slice]:
    if len(roi) != 4:
        raise SystemExit("ROI must contain four normalized coordinates")
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise SystemExit(f"Invalid normalized ROI: {roi}")
    height, width = shape[:2]
    result = (slice(int(y0 * height), int(y1 * height)), slice(int(x0 * width), int(x1 * width)))
    if result[0].start == result[0].stop or result[1].start == result[1].stop:
        raise SystemExit(f"Empty ROI at this resolution: {roi}")
    return result


def roi_energy(path: Path, roi: list[float]) -> np.ndarray:
    image = read_checked(path)
    result = image[roi_slice(roi, image.shape)].reshape(-1, 3).sum(axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise SystemExit(f"Nonfinite ROI energy: {path}")
    return result


def roi_mse(reference: Path, candidate: Path, roi: list[float]) -> float:
    ref = read_checked(reference)
    cand = read_checked(candidate, ref.shape)
    box = roi_slice(roi, ref.shape)
    difference = ref[box].astype(np.float64) - cand[box].astype(np.float64)
    result = float(np.mean(difference * difference))
    if not np.isfinite(result):
        raise SystemExit("Nonfinite ROI MSE")
    return result


def reference_energy(reference: Path, roi: list[float]) -> np.ndarray:
    expected = roi_energy(reference, roi)
    if not np.isfinite(expected).all() or np.any(expected <= 0.0):
        raise SystemExit(f"Reference has unusable RGB energy: {reference}")
    return expected


def render_hyperion(label: str, scene: Path, spp: int, seed: int) -> tuple[Path, float]:
    out = OUT / f"{label}.exr"
    for stale in (out, out.with_suffix(".png")):
        stale.unlink(missing_ok=True)
    exe = GITHUB / "Hyperion" / "build" / "hyperion.exe"
    code, log_path, _ = run_exe(label, exe, [
        "--scene", str(scene), "--sms-mode", "off", "--width", RESOLUTION[0], "--height", RESOLUTION[1],
        "--spp", str(spp), "--rng-seed", str(seed), "--validation", "--output", str(out)])
    if code != 0 or not out.exists() or not out.with_suffix(".png").exists():
        raise SystemExit(f"{label}: Hyperion render/export failed")
    read_checked(out)
    return out, frame_seconds(log_path, spp)


def render_theia(label: str, scene: Path, mode: str, frames: int, seed: int) -> tuple[Path, float]:
    out = OUT / f"{label}.exr"
    for stale in (out, out.with_suffix(".png")):
        stale.unlink(missing_ok=True)
    exe = GITHUB / "Theia" / "build" / "theia.exe"
    code, log_path, _ = run_exe(label, exe, [
        "--scene", str(scene), "--sms-mode", mode, "--width", RESOLUTION[0], "--height", RESOLUTION[1],
        "--offscreen-frames", str(frames), "--rng-seed", str(seed), "--validation", "--output", str(out)])
    if code != 0 or not out.exists() or not out.with_suffix(".png").exists():
        raise SystemExit(f"{label}: Theia {mode} render/export failed")
    read_checked(out)
    return out, frame_seconds(log_path, frames)


def record(entry: dict) -> None:
    manifest = OUT / "run_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    entries.append(entry)
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def stage_smoke(case: str) -> None:
    scene = prepare_run_dir(case)
    for mode in ("off", "local", "reuse"):
        _out, _ = render_theia(f"smoke_theia_{case}_{mode}", scene, mode, 2, SEEDS[0])
        log(f"smoke Theia {case} {mode}: finite EXR and PNG present")
    render_hyperion(f"smoke_hyperion_{case}", scene, 2, SEEDS[0])
    record({"stage": "smoke", "case": case, "status": "accepted", "provenance": provenance(case)})


def stage_reference(case: str) -> Path:
    scene = prepare_run_dir(case)
    roi = CASES[case]["roi"]
    for spp in REFERENCE_SPP_LADDER:
        seeds = SEEDS[:2] if spp == REFERENCE_SPP_LADDER[0] else SEEDS[2:]
        halves = []
        for seed in seeds:
            path, _ = render_hyperion(f"ref_{case}_{spp}_{seed}", scene, spp, seed)
            halves.append(reference_energy(path, roi))
        relative = float(np.max(np.abs(halves[0] - halves[1]) / np.maximum(0.5 * (halves[0] + halves[1]), 1e-30)))
        log(f"reference {case} spp={spp}: split-seed RGB uncertainty {relative:.5f}")
        if relative <= REF_UNCERTAINTY:
            final, elapsed = render_hyperion(f"ref_{case}_final_{spp}", scene, spp, SEEDS[0])
            reference_energy(final, roi)
            record({"stage": "reference", "case": case, "status": "accepted", "spp": spp,
                    "uncertainty": relative, "file": final.name, "reference_sha256": sha256(final),
                    "completed_frame_seconds": elapsed, "provenance": provenance(case)})
            log(f"reference {case}: accepted at {spp} spp; completed-frame time {elapsed:.3f}s")
            return final
    raise SystemExit(f"reference {case}: precision ladder exhausted; report reference-budget blocker")


def load_reference(case: str) -> Path:
    manifest = OUT / "run_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"run --stage reference --case {case} first")
    current = provenance(case)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in reversed(entries):
        if entry.get("stage") != "reference" or entry.get("case") != case or entry.get("status") != "accepted":
            continue
        if entry.get("provenance") != current:
            raise SystemExit(f"{case}: reference provenance is stale; rerun the reference")
        path = OUT / entry["file"]
        if not path.is_file() or sha256(path) != entry["reference_sha256"]:
            raise SystemExit(f"{case}: reference bytes changed or are missing")
        read_checked(path)
        return path
    raise SystemExit(f"run --stage reference --case {case} first")


def stage_correctness(case: str, reference: Path) -> None:
    scene = prepare_run_dir(case)
    roi = CASES[case]["roi"]
    expected = reference_energy(reference, roi)
    scores: dict[str, dict[int, float]] = {}
    details = {}
    for mode in ("local", "reuse"):
        scores[mode] = {}
        for frames in CANDIDATE_FRAMES:
            energies, mses = [], []
            per_seed = {}
            for seed in SEEDS:
                path, elapsed = render_theia(f"cand_{case}_{mode}_{frames}_{seed}", scene, mode, frames, seed)
                energies.append(roi_energy(path, roi))
                mses.append(roi_mse(reference, path, roi))
                per_seed[str(seed)] = {
                    "energy_ratio": (energies[-1] / expected).tolist(),
                    "mse": mses[-1],
                    "completed_frame_seconds": elapsed,
                }
            ratios = (np.mean(energies, axis=0) / expected).tolist()
            pooled_mse = float(np.mean(mses))
            per_channel = np.array(energies, dtype=np.float64) / expected
            spread = float(np.max(np.std(per_channel, axis=0, ddof=1)))
            stderr = spread / np.sqrt(len(SEEDS))
            if not np.isfinite(ratios).all() or not np.isfinite(pooled_mse):
                raise SystemExit(f"correctness {case}: nonfinite metric")
            log(f"correctness {case} {mode} @{frames}f: pooled ratios {ratios}; "
                f"max per-seed spread {spread:.4f} (pooled SE {stderr:.4f}); mean MSE {pooled_mse:.7g}")
            if frames == BIAS_GATE_FRAMES and max(abs(ratio - 1.0) for ratio in ratios) > ENERGY_TOLERANCE:
                raise SystemExit(
                    f"correctness {case} {mode} @{frames}f: pooled RGB energy outside 2% "
                    f"(pooled SE {stderr:.4f}; a genuine bias, not tail variance)")
            scores[mode][frames] = pooled_mse
            details[f"{mode}_{frames}"] = {"ratios": ratios, "mse": pooled_mse, "seed_spread": spread,
                                           "pooled_standard_error": stderr, "gated": frames == BIAS_GATE_FRAMES,
                                           "seeds": per_seed}
    for mode in ("local", "reuse"):
        for lower, higher in itertools.pairwise(CANDIDATE_FRAMES):
            if scores[mode][higher] >= scores[mode][lower]:
                raise SystemExit(f"correctness {case} {mode}: {higher}-frame error did not improve over {lower}")
    record({"stage": "correctness", "case": case, "status": "accepted", "results": details,
            "provenance": provenance(case)})


def stage_performance(case: str, reference: Path) -> None:
    scene = prepare_run_dir(case)
    roi = CASES[case]["roi"]
    expected = reference_energy(reference, roi)
    timings = {}
    for mode in ("off", "local", "reuse"):
        samples = []
        for pilot in range(TIMING_PILOTS):
            _, elapsed = render_theia(f"pilot_{case}_{mode}_{pilot}", scene, mode, TIMING_PILOT_FRAMES, SEEDS[pilot])
            samples.append(elapsed)
        timings[mode] = float(np.median(samples))
        log(f"pilot {case} {mode}: median completed-frame time {timings[mode]:.3f}s / {TIMING_PILOT_FRAMES}f")
    results = {}
    timing_valid = {}
    for mode in ("off", "local", "reuse"):
        results[mode] = {}
        timing_valid[mode] = {}
        per_frame = timings[mode] / TIMING_PILOT_FRAMES
        for target in TARGET_SECONDS:
            frames = max(1, round(target / per_frame))
            # Timing-only calibration, never use its pixels as quality evidence.
            _, elapsed = render_theia(f"timecheck_{case}_{mode}_{target}_{frames}", scene, mode, frames, SEEDS[0])
            refined = False
            if abs(elapsed - target) > target * TIME_TOLERANCE:
                refined = True
                frames = max(1, round(frames * target / elapsed))
                _, elapsed = render_theia(f"timecheck2_{case}_{mode}_{target}_{frames}", scene, mode, frames, SEEDS[0])
            timing_valid[mode][str(target)] = abs(elapsed - target) <= target * TIME_TOLERANCE
            if not timing_valid[mode][str(target)]:
                log(f"equal-time {case} {mode} @{target}s: timing inconclusive after one refinement")
            energies, mses, times = [], [], []
            per_seed = {}
            for seed in SEEDS:
                path, elapsed = render_theia(f"eqt_{case}_{mode}_{target}_{seed}", scene, mode, frames, seed)
                if abs(elapsed - target) > target * TIME_TOLERANCE:
                    timing_valid[mode][str(target)] = False
                energies.append(roi_energy(path, roi))
                mses.append(roi_mse(reference, path, roi))
                times.append(elapsed)
                per_seed[str(seed)] = {"mse": mses[-1], "completed_frame_seconds": elapsed}
            ratios = (np.mean(energies, axis=0) / expected).tolist()
            mse = float(np.mean(mses))
            stderr = float(np.std(mses, ddof=1) / np.sqrt(len(mses)))
            results[mode][str(target)] = {"frames": frames, "refined": refined, "ratios": ratios,
                "mse": mse, "mse_standard_error": stderr, "mean_seconds": float(np.mean(times)),
                "min_seconds": min(times), "max_seconds": max(times), "timing_within_5_percent": timing_valid[mode][str(target)], "seeds": per_seed}
            log(f"equal-time {case} {mode} @{target}s: frames={frames} MSE={mse:.7g}±{stderr:.3g}; "
                f"times={times}")
    for target in TARGET_SECONDS:
        if results["reuse"][str(target)]["mse"] >= results["local"][str(target)]["mse"]:
            record({"stage": "performance", "case": case, "status": "no-benefit", "results": results,
                    "provenance": provenance(case)})
            raise SystemExit(f"equal-time {case}: reuse did not reduce MSE at {target}s")
    if not all(timing_valid[mode][str(target)] for mode in ("off", "local", "reuse") for target in TARGET_SECONDS):
        record({"stage": "performance", "case": case, "status": "timing-inconclusive", "results": results,
                "provenance": provenance(case)})
        raise SystemExit(f"equal-time {case}: timing inconclusive; quality is not an accepted equal-time result")
    record({"stage": "performance", "case": case, "status": "accepted", "results": results,
            "provenance": provenance(case)})


def stage_stability(case: str) -> None:
    scene = prepare_run_dir(case)
    first, _ = render_theia(f"stab_{case}_a", scene, "reuse", 64, SEEDS[0])
    second, _ = render_theia(f"stab_{case}_b", scene, "reuse", 64, SEEDS[0])
    if sha256(first) != sha256(second):
        raise SystemExit("stability: same-seed captures are not byte-identical")
    energies = []
    for seed in SEEDS:
        path, _ = render_theia(f"stab_{case}_{seed}", scene, "reuse", 64, seed)
        energies.append(roi_energy(path, CASES[case]["roi"]))
    mean = np.mean(energies, axis=0)
    spread = float(np.max(np.abs(np.array(energies) - mean) / np.maximum(mean, 1e-30)))
    record({"stage": "stability", "case": case, "status": "accepted", "seed_spread": spread,
            "provenance": provenance(case)})
    log(f"stability {case}: deterministic; per-seed ROI energy spread {spread:.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["smoke", "reference", "correctness", "performance", "stability"])
    parser.add_argument("--case", default="C1", choices=sorted(CASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        stage_smoke(args.case)
    elif args.stage == "reference":
        stage_reference(args.case)
    else:
        reference = load_reference(args.case)
        if args.stage == "correctness":
            stage_correctness(args.case, reference)
        elif args.stage == "performance":
            stage_performance(args.case, reference)
        else:
            stage_stability(args.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
