# AGENTS.md — Theia

Quick-start context for AI agents so basic facts don't have to be rediscovered each session.

## What this repo is

**Theia** is the **real-time** renderer: a Vulkan mesh-shader rasterizer (meshlets) being
aligned to match **Hyperion** (the path-traced ground truth). It uses split-sum IBL,
clustered/forward shading, and screen-space post effects (SSR/SSAO/bloom/SSGI).

Pipeline (dependency direction):

```
Aether (file format)  ->  Harmonia (shared Vulkan lib)  ->  Hyperion (path tracer, ground truth)
                                                         \-> Theia    (this repo, real-time)
```

Consumes Aether + Harmonia via CMake FetchContent. The demo is a thin `harmonia::App`
subclass injecting `harmonia::IRenderer`. GPU-optimized scene upload / meshlet layouts are
Theia-specific (not in Harmonia).

## Algorithm strategy (drives every technique decision)

- **Best quality, future hardware, ONE approach.** Theia targets future hardware, so pick the
  single best-quality real-time algorithm that matches the Hyperion path-traced ground truth —
  **no cheap fallbacks, no dual-track.** Heavy-but-correct is fine; it scales with hardware.
- **Still almost real-time on current hardware.** The chosen technique must stay near-interactive
  on today's dev machine (bounded work + denoise), not offline brute force. For local testing the
  technique still runs here — just **lower the render resolution** (e.g. the 320x240 parity size)
  rather than swapping in a cheaper algorithm. Resolution is the performance knob; the approach
  never forks.
- **Target hardware (capability, not a brand):** a high-end GPU class with **hardware ray
  tracing**, a **large (~100GB-class) unified memory** pool, and AI-based denoising/upscaling.
  HW ray tracing is first-class and memory is abundant → favor ray-traced techniques;
  low-memory approximations (e.g. WBOIT) have no advantage here. It must also run on **other
  hardware**, so keep the technique single and scale it by resolution (below), not by branching.
- **Transparency/refraction (Group 4, v0.6.0):** committed to **HW ray-traced transmission &
  refraction** via the existing scene TLAS (`t_tlas`, inline `TraceRayInline` already used for
  shadows). Same mechanism as Hyperion's BSDF → best parity, order-independent, refraction
  native, and leaves the HDR alpha (indirect-weight) channel untouched. WBOIT is superseded.

- **GPU-driven, latest standard Vulkan, no vendor extensions.** Prefer GPU-driven rendering
  (indirect/mesh-shader draws, GPU-side culling, bindless) using the latest Vulkan features
  available, but **cross-vendor only** — core + `KHR`/`EXT`. **Never** vendor-specific
  extensions (`VK_NV_*`, `VK_AMD_*`, `VK_INTEL_*`). E.g. ray tracing uses
  `VK_KHR_ray_query` / `VK_KHR_acceleration_structure` (already in use), not a vendor RT path.
  This keeps Theia portable to the "other hardware" requirement above.

## Running

```powershell
build/theia.exe --scene cornell_classic --output out.exr --no-postfx   # headless, parity baseline
build/theia.exe --scene fixture_ibl                                    # interactive window
```

CLI flags = the common Harmonia set: `--scene/-s`, `--output/-o` (headless EXR+PNG),
`--width`, `--height`, `--validation`/`--no-validation`, **`--no-postfx`**,
`--indirect-ambient <f>`, `--ssgi-strength <f>`. Theia has no `--spp` (it is not stochastic).

⚠️ No `--offscreen` flag — headless is triggered by `--output`.

## Parity: ALWAYS use `--no-postfx`

Parity vs Hyperion is measured with **`--no-postfx`**. Screen-space effects (SSR/SSAO/bloom/
SSGI) introduce approximations that diverge from a path tracer. Measured on `alignment_suzanne`
(full IBL): no-postfx mean_diff **9.82** vs postfx-ON **24.18** — postfx nearly triples the
error and darkens the image (signed +4.8 -> +22.1).

Why postfx darkens IBL-only scenes: with no analytic lights the item-30 "indirect weight"
mask is ~1.0 everywhere, so SSAO attenuates almost the whole image, while the path tracer
computes true ray-cast occlusion and does not over-darken.

Compare with `Harmonia/tools/compare_renders.py ref.exr cand.exr` (pass = mean_diff <= 4.0,
pre-tonemap EXR, same color space).

## Gotchas (each has cost a debug cycle)

- **Assets come from `build/_deps/aether-src/assets/`** (FetchContent clone), NOT the working
  Aether tree. Editing `C:\Development\GitHub\Aether\assets` does nothing unless you update the
  `_deps` copy or build with `-DFETCHCONTENT_SOURCE_DIR_AETHER=...`. Symptom: two "different"
  renders give byte-identical metrics.
- **IBL parity reference must be high-spp:** scenes using `alignment_16spp_8bounce.render.toml`
  give a 16-spp (noisy) Hyperion reference — render it with `hyperion --spp 512` first, or the
  diff measures noise. (On `alignment_suzanne` this alone inflated mean_diff 9.82 -> 13.56.)
- **Real-time GI gap is expected, not a bug:** Theia does single-bounce IBL + flat ambient,
  no path-traced multi-bounce GI. On scenes with inter-reflection (floor + objects) Theia is
  darker than Hyperion; the error concentrates on lit/occluded surfaces while directly-viewed
  environment matches. The isolated-diffuse `fixture_ibl` passes at 1.76 — the irradiance map
  itself is correct. Tracked with the RT-GI / SSR roadmap.
- **IBL specular is split-sum (band-limited):** sharp HDR sun-disc reflections cannot be
  represented by the 1024x512 / 8-mip prefiltered map. Accepted approximation (see plan item 22).
- **New compute shader entry points** MUST be added to `THEIA_ENTRY_SHADERS` in `CMakeLists.txt`
  or the `.spv` is never compiled and the shader fails to load at runtime ("file not found").
- **HDR alpha channel carries an indirect-weight mask** (forward pass) consumed by SSAO blur so
  AO modulates only indirect light. Don't repurpose the alpha channel.

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

SDL3, slangc and volk come from the Vulkan SDK (not vcpkg). vcpkg provides openexr, stb, glm.

## Conventions

- Commit, but do **not** push unless asked.
- Working color space is scene-referred (e.g. `lin_rec2020_scene`).
- Material model tagged `model = "openpbr"`; parameters are identical to Hyperion — the goal is
  the best real-time approximation of the same OpenPBR inputs, not different parameters.
