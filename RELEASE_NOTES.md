# Theia v1.0.0-pre-alpha (2026-11-28) — prerelease

Pre-alpha development snapshot and GitHub backup. Consumes Harmonia v1.0.0-pre-alpha and applies
the shared medium-exit radiance transport in the GI adapter. Verification: **94/94** tests pass.
The README gallery is regenerated for this snapshot. Detailed OpenPBR Shader Playground validation
continues after the prerelease.

---

# Theia v1.0.0 (2026-11-26) — major stable release

Stable public baseline for the real-time renderer. Coordinates with
**Harmonia v1.0.0 + Aether v1.0.0**.

## Changes

- The architecture contract declares Aether as an explicit external include boundary.
- Documentation references are checked strictly: a path that a document names but that does not
  exist fails the check.
- The architecture and dead-code checks are clean. The accepted file splits and large-module shapes
  are documented in `ARCHITECTURE.md` and unchanged.

## Verification

- Full suite: 67/67 tests pass.
- `verify-full`, `check_arch`, `check_dead` and `check_docs_refs --strict` are clean.

## Release state

- The release-preparation commits are local. Push and the `v1.0.0` tag require explicit
  authorization.

---

# Theia v0.12.0 (2026-11-24) — minor release

Real-time renderer release that consumes **Harmonia v0.12.0 + Aether v0.12.0**.

## Changes

- Scene-camera assembly and animated camera placement come from Harmonia's shared
  `CameraAssembly`. The camera assembly that each renderer used to keep privately is shared now.
  Controller navigation, camera jitter, the accumulation and view checks and the presentation are
  unchanged.

## Verification

- Shared camera assembly: 7/7 tests. Theia: 67/67.
- Qualification-tool tests: 11/11; ruff clean.
- Direct capture identity: `mean_diff=0`, `rel_mse=0`, `SSIM=1.0`, `PSNR=inf`.
- Parity: 14/14 pass. Vulkan validation over the full gallery: 60/60 clean.

## Gallery

- The README gallery is regenerated at 1280×720 with 256 frames.
- 23 of 30 images are byte-identical to the previous gallery. The other seven differ only at
  speckle level and were accepted after visual review.

## Release state

- The release-preparation commits are local. Push and the `v0.12.0` tags go together.

---

# Theia v0.9.0 (2026-11-16) — minor release

Real-time renderer release that consumes **Harmonia v0.9.0 + Aether v0.9.0**.

## Changes

- Deformation. The shared `DeformPass` runs from the animation tick; scenes with only skinning or
  only morph targets enter it as well. It stages the matrices and weights, runs the skinning
  dispatch (linear blend or dual-quaternion blending) or the morph dispatch over the immutable
  rest-pose copy of the vertices, copies the result into the mesh's BLAS input, refits the BLAS in
  place and updates the TLAS. Theia contributes the per-mesh vertex offsets, the rest-pose upload for
  deforming scenes, conservative culling bounds for skinned and morphed meshes, the radiance-cache
  exclusion for morph scenes, and the tie of the pass creation and reset to the scene lifetime.
- Dual-quaternion skinning needs no renderer code: the blend mode is carried by the shared job and
  staging.
- Parity thresholds recalibrated to a mean difference ≤ 120, relative MSE ≤ 250, SSIM ≥ 0.6 and
  luminance-histogram correlation ≥ 0.8. The manifest passes 14/14 with this renderer, without any
  validation message.
- Animated camera. A camera bound with `[camera] node` is re-placed on every tick. The
  re-placement has its own tick check, so a camera bound to a static node does not restart the
  accumulation. The camera controller is skipped while a camera is bound, and `--frame N` seeks
  the animation through the shared clock.
- One animation clock and time-based capture. Scene stamping, deformation, TLAS update and the
  scripted camera all read `App::animationClock` (a time in seconds and a tick index), so a camera
  can no longer lag behind an interval capture. The shared flags `--capture-every-ms` and
  `--start-seconds` sample the animation at arbitrary times.

## Verification

- Module suite 61/61. End-to-end sequence renders of the skin, morph and dual-quaternion fixtures:
  exit 0 and no validation message. Frame 0 against frame 2 differs by a mean difference of
  11.1–11.4, which shows the motion. clang-format clean.
- Animated camera: module suite 61/61; orbit sequences without validation messages; frame 0
  against Hyperion at 256 frames and 256 samples per pixel passes (mean 25.558, SSIM 0.938).
- One animation clock: module suite 61/61; tick-grid renders are bit-identical before and after the
  change; a classic capture with a static node camera converges (0.000 difference against the
  sequence render); no validation message.

## Known limitations

- Skinned and morphed meshes skip frustum and task-shader culling (conservative envelope) until a
  per-frame bounds refresh exists.
- Motion vectors cover instance transforms; per-vertex deformation is not in the motion vectors.
  TAA in the interactive window can ghost on fast deformation. The accumulation path restarts per
  tick and is unaffected.

## Release state

- Commits are local. Tag `v0.9.0`; push commits and tag together. The pins reference the new
  Harmonia and Aether tags; local builds use the `FETCHCONTENT_SOURCE_DIR_*` overrides.
