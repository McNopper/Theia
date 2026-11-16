# Theia v0.9.0 (2026-11-16) - minor release

Real-time renderer release consuming **Harmonia v0.8.0 + Aether v0.9.0**.

## Shipped
- **Deformation (ANI4/ANI5 renderer half)**: the shared `DeformPass` runs from the animation
  tick (skin-only/morph-only scenes enter it too): stage matrices+weights -> LBS/morph/DQS
  dispatch over the immutable rest copy -> copy into the mesh-local BLAS input -> in-place BLAS
  refit -> TLAS, before `updateTlas`. Renderer substrate: `meshSceneVertexOffsets`, the
  `scene.vertexRest` upload for deforming scenes, conservative culling bounds for
  skinned/morphed meshes (deformed vertices leave rest bounds), radiance-cache gate extended to
  `hasMorphs()`, pass create/reset bound to scene lifetime.
- **DQS**: consumed with zero renderer code (the blend mode rides the shared job/staging).
- **Parity gate**: recalibrated thresholds (mean<=120 / rel<=250 / SSIM>=0.6 / hist>=0.8) -
  fresh manifest run **14/14 PASS** with this renderer (0 VUID).
- Triage sync: C11 consumed-claim removed (never implemented), C12 confirmed shipped.

## Receipts
- Module suite **61/61**; end-to-end sequence renders (skin, morph, DQS fixtures): exit 0,
  **0 VUID / 0 validation messages**, motion proof (frame0 vs frame2 mean_diff 11.1-11.4);
  clang-format clean.

## Known limitations (recorded)
- Skinned/morphed meshes skip frustum/task culling (conservative envelope) until a per-frame
  bounds refresh exists.
- ANI3 motion vectors cover INSTANCE transforms; per-VERTEX deformation is not in the motion
  vectors (interactive-window TAA can ghost on fast deformation; the accumulation/reference
  path resets per tick and is unaffected).

## Release mechanics
- Local commits only: **tag `v0.9.0`, push commits + tag together** (pins reference the new
  Harmonia/Aether tags; local builds use `FETCHCONTENT_SOURCE_DIR_*` overrides).

## Also in this release (added after the notes assembly)

- **ANI8 half** (commit 8676a85): per-tick camera-from-node placement in
  `record()` with its own tick guard (static bound nodes never trip the
  animation block's accumulation reset), scripted-camera handling in `onUpdate`
  (controller skipped while bound), `--frame N` via the shared clock.
  Receipts: 61/61, orbit sequences 0 VUID, frame0 parity vs Hyperion PASS
  (mean 25.558 / SSIM 0.938).
