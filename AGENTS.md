# AGENTS.md — Theia

Quick-start context for AI agents so basic facts don't have to be rediscovered each session.

## What this repo is

**Theia** is a **lean accumulation path-traced renderer**: a GPU-driven Vulkan renderer that
converges to the **Hyperion** path-traced ground truth through frame accumulation. Multi-bounce
global illumination — diffuse, specular, environment NEE, transmission and refraction — is
**hardware ray-traced** through Harmonia's shared OpenPBR `path_integrator`, the same estimator
and BSDF Hyperion uses. **ReSTIR DI** provides spatiotemporal reservoir resampling for direct
illumination from emissive triangles. An **A-SVGF denoiser** and **TAA** stabilize the image,
and progressive accumulation resolves it to the reference.

Pipeline (dependency direction):

```mermaid
flowchart LR
    A["Aether<br/>file format"] --> H["Harmonia<br/>shared Vulkan lib"]
    H --> Hy["Hyperion<br/>path tracer · ground truth"]
    H --> T["<b>Theia</b><br/>accumulation renderer (this repo)"]
```

Consumes Aether + Harmonia via CMake FetchContent. The demo is a thin `harmonia::App`
subclass injecting `harmonia::IRenderer`. GPU-optimized scene upload / meshlet layouts are
Theia-specific (not in Harmonia).

## Algorithm strategy (drives every technique decision)

- **One best-quality approach that matches the ground truth.** Theia picks the single
  path-traced estimator that converges to Hyperion — the shared OpenPBR BSDF with environment
  NEE and multi-bounce transport — and scales it with frame accumulation.
- **Scales by resolution and frame count.** Accumulation is the quality/performance knob:
  lower the render resolution or frame count for fast iteration (e.g. the 320x240 parity size),
  raise them for final deliverables. The approach itself stays single.
- **Target hardware (capability, not a brand):** a high-end GPU class with **hardware ray
  tracing**, a **large (~100GB-class) unified memory** pool, and AI-based denoising/upscaling.
  HW ray tracing is first-class and memory is abundant → ray-traced techniques are favored.
  It also runs on **other hardware**, kept portable by scaling resolution and frames.
- **Transparency/refraction (shipped v0.4.0+):** transmission/refraction is **HW ray-traced**
  via the scene TLAS and runs through Harmonia's shared `path_integrator` — the same estimator
  and OpenPBR BSDF Hyperion uses. Order-independent, refraction-native, best parity.

- **GPU-driven, latest standard Vulkan, cross-vendor.** Prefer GPU-driven rendering
  (indirect/mesh-shader draws, GPU-side culling, bindless) using the latest Vulkan features
  available, but **cross-vendor only** — core + `KHR`/`EXT`. **Never** vendor-specific
  extensions (`VK_NV_*`, `VK_AMD_*`, `VK_INTEL_*`). E.g. ray tracing uses
  `VK_KHR_ray_query` / `VK_KHR_acceleration_structure` (already in use), not a vendor RT path.
  This keeps Theia portable to the "other hardware" requirement above.

## Running

```powershell
build/theia.exe --scene cornell_classic --output out.exr               # headless EXR+PNG
build/theia.exe --scene fixture_ibl                                    # interactive window
```

CLI flags: `--scene/-s`, `--output/-o` (headless EXR+PNG), `--width`, `--height`,
`--offscreen-frames <n>` (accumulation count), `--validation`/`--no-validation`,
`--rt-gi`/`--no-rt-gi` (GI on/off, default on), `--no-restir-di` (ReSTIR DI off),
`--taa`/`--no-taa`, `--no-camera-jitter`, `--indirect-ambient <f>`.
Theia accumulates frames — use `--offscreen-frames` for convergence quality.

⚠️ No `--offscreen` flag — headless is triggered by `--output`.

## Parity & screenshots: unified RT path

Parity vs Hyperion and showcase screenshots use the unified accumulation RT path.

Compare with `Harmonia/tools/compare_renders.py ref.exr cand.exr` (pass = mean_diff <= 4.0,
pre-tonemap EXR, same color space).

## Gotchas (each has cost a debug cycle)

- **Assets come from `build/_deps/aether-src/assets/`** (FetchContent clone), NOT the working
  Aether tree. Editing `C:\Development\GitHub\Aether\assets` does nothing unless you update the
  `_deps` copy or build with `-DFETCHCONTENT_SOURCE_DIR_AETHER=...`. Symptom: two "different"
  renders give byte-identical metrics.
- **IBL parity reference must be high-spp:** a low-spp Hyperion reference is noisy — render it
  with `hyperion --spp 256` first, or the diff measures noise, not a real discrepancy.
- **Real-time multi-bounce GI now exists (RT-GI):** Theia runs a HW ray-traced GI pass
  (`gi.comp.slang` → shared `path_integrator`) providing path-traced multi-bounce indirect
  (diffuse + specular + env-NEE) and transmission/refraction, default-on (`--no-rt-gi` to
  disable). The old "single-bounce IBL + flat ambient, darker than Hyperion" gap is closed for
  the unified pipeline. The isolated-diffuse `fixture_ibl` passes at 1.76.
- **Bulk subsurface + transmission-scatter run the shared volumetric walk in `gi.comp`:** the
  compute loop executes the same chromatic hero-wavelength free-flight/scatter/boundary
  estimator as Hyperion (`runMediumWalk`), on both primary and secondary vertices. `sampleBSDF`
  sets `entersMedium` with per-channel σ_t exactly like Hyperion; it is NOT a diffuse-tint
  approximation anymore. Thin-walled subsurface keeps the diffuse sheet.
- **IBL specular split-sum is the GI-OFF fallback only:** the prefiltered split-sum map
  (1024x512 / 8-mip, band-limited, can't resolve a sharp HDR sun-disc) is used only on the
  `--no-rt-gi` debug path. The default unified path gets specular reflections from RT-GI.
- **New compute shader entry points** MUST be added to `THEIA_ENTRY_SHADERS` in `CMakeLists.txt`
  or the `.spv` is never compiled and the shader fails to load at runtime ("file not found").
- **GI primary surface = the re-traced closest hit, NOT the rasterized giBuffer materialIdx.**
  The giBuffer is draw-order dependent for overlapping transparent surfaces (depth-write off);
  `gi.comp.slang` trusts the order-independent inline ray query for primary visibility. Do not
  re-add a `materialIdx ==` guard (it caused black holes on overlapping glass; commit 4a04e6e).

## Test scenes

- Quick parity/iteration (cheap): `cornell_classic`, `cornell_spheres`, `cornell_suzanne`,
  `dragon_teapot`, `fixture_ibl`.
- **Never** use `ABeautifulGame` for quick test renders — expensive. (Required only in final
  screenshot/render *deliverable* batches.)

## Build & test

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
cmake --build build
cd build; ctest --output-on-failure
```

SDL3, slangc and volk come from the Vulkan SDK (not vcpkg). vcpkg provides openexr, stb.

## Conventions

- Commit, but do **not** push unless asked.
- Working color space is scene-referred (e.g. `lin_rec2020_scene`).
- **Material model = OpenPBR Surface** (Academy Software Foundation), tagged `model = "openpbr"`.
  OpenPBR's canonical/reference implementation is **MaterialX** (`mx_*` genGLSL nodes); follow
  OpenPBR parameter naming and use MaterialX as the cross-check. Parameters are identical to
  Hyperion — the goal is the best real-time approximation of the same OpenPBR inputs, not
  different parameters. The shared OpenPBR BSDF lives in Harmonia (`bsdf_shared.slang`).

## GPU-driven design (Theia)

**Principle:** GPU-driven by design — all draw submission parameters (dispatch counts, per-draw
instance indices) are GPU-resident and GPU-written. The CPU records commands only; it never reads
back GPU-side state to determine draw counts or parameters.

**Implemented architecture (GD2/GD3/GD6):**

```mermaid
flowchart TD
    A["GpuCullPass.dispatch()"] --> B["compactInstanceList[]<br/>visible instance indices"]
    A --> C["indirectDrawBuf<br/>{visibleCount, 1, 1}"]
    B --> E["ForwardRenderer binding 10"]
    C --> G["GD6: vkCmdExecuteGeneratedCommandsEXT<br/>1 sequence, cmd = indirectDrawBuf"]
    C --> G2["GD3: vkCmdDrawMeshTasksIndirectEXT<br/>drawCount=1, stride=12"]
    E --> H["task shader: instIdx = compactInstanceList[gid.x]"]
    G --> H
    G2 --> H
    H --> I["DispatchMesh(meshletCount, 1, 1)"]
```

**GD6 single GPU-generated draw:** `vkCmdExecuteGeneratedCommandsEXT` issues **one** sequence
whose indirect command is `indirectDrawBuf = {visibleCount, 1, 1}` (GPU-written by the cull
compute). That single draw dispatches `visibleCount` task workgroups, and the task shader indexes
`compactInstanceList[gid.x]` — **identical semantics to the GD3 fallback**, but the command and
its dispatch count stay GPU-resident and are consumed through the modern device-generated-commands
path. Both paths share one task-shader entry point (`gid.x = 0..visibleCount-1`).

> ⚠️ **Do not** use N per-instance DGC sequences keyed on `SV_DrawIndex`: with `maxDrawCount=1`,
> `SV_DrawIndex` is the draw index *within* a sequence (always 0), **not** the sequence index — so
> every sequence would read `compactInstanceList[0]` and only instance 0 (e.g. the floor) would
> rasterize. This was the GD6 "missing geometry" regression; the single-sequence design avoids it.

- `forward_cull.comp.slang`: 64-thread compute; Gribb-Hartmann 5-plane frustum cull; atomic
  `InterlockedAdd` on `indirectDrawBuf` byte-offset 0 accumulates `groupCountX` = visible count;
  thread 0 restores `groupCountY=1` / `groupCountZ=1` after per-frame `vkCmdFillBuffer` reset.
- `VkIndirectCommandsLayoutEXT`: one `DRAW_MESH_TASKS_EXT` token, stride=12.
- DGC preprocess buffer: sized for `maxSequenceCount=1`, allocated with
  `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT` (64-bit flag via `VkBufferUsageFlags2CreateInfo`;
  requires `maintenance5` + `VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT`).
- Cull→draw barrier on `indirectDrawBuf` covers both `DRAW_INDIRECT` (GD3) and `COMMAND_PREPROCESS`
  (GD6) destination stages so the GPU-written count is fully visible before it is consumed.
- Three-way dispatch per pass: DGC (preferred) → GD3 indirect → CPU-count fallback.
- Debug A/B toggles: `THEIA_FORCE_GD3` (skip DGC, use indirect draw), `THEIA_SINGLE_PASS`
  (bypass two-pass Hi-Z), `THEIA_DISABLE_HIZ` (draw all meshlets).
- GD4: Both Hi-Z passes (`cullPhase=1` and `cullPhase=2`) use the same GPU-indirect path;
  per-meshlet Hi-Z occlusion is handled by the mesh shader using `cullPhase` push constant.

**Active extensions for GPU-driven draws:**
- `VK_EXT_mesh_shader`: `vkCmdDrawMeshTasksIndirectEXT` for GPU-count indirect draws ✅
- `VK_EXT_device_generated_commands` (`dgcSupported`): single-sequence GPU-generated mesh draw ✅

**Acceleration structure builds — device-side only (Khronos deprecation compliant):**
- BLAS builds: `vkCmdBuildAccelerationStructuresKHR` (`Geometry::buildBlas`).
- TLAS builds: `vkCmdBuildAccelerationStructuresKHR` (`Scene::buildTlas`).
- `vkBuildAccelerationStructuresKHR` (host-side) is **never used** — deprecated per the
  [Khronos RT AS deprecation blog](https://www.khronos.org/blog/vulkan-ray-tracing-deprecating-host-side-acceleration-structure-builds).
