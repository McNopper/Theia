# Theia

[![Build](https://github.com/McNopper/Theia/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/McNopper/Theia/actions/workflows/build.yml)

GPU-driven accumulation path-traced renderer for OpenPBR materials.

> *[Theia](https://en.wikipedia.org/wiki/Theia_(mythology)) — Titaness of heavenly light, mother of Helios, Selene and Eos.*

Theia is a Vulkan 1.4 renderer that draws each frame with a mesh-shader rasterizer and hardware ray
tracing, and converges to the image of [Hyperion](https://github.com/McNopper/Hyperion), the family's
offline path tracer, by accumulating jittered frames. Both renderers evaluate the OpenPBR Surface
v1.1.2 material model with the same BSDF and the same path integrator, so they compute the same light
transport and differ only in how they sample it.

- [Screenshots](#screenshots)
- [The repository family](#the-repository-family)
- [How it works](#how-it-works)
- [Running](#running)
- [Scenes](#scenes)
- [Building](#building)
- [Testing and verification](#testing-and-verification)
- [Conventions](#conventions)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Where to read more](#where-to-read-more)
- [Dependencies](#dependencies)
- [References](#references)

## Screenshots

Thirty of the scenes Theia shares with [Hyperion](https://github.com/McNopper/Hyperion), each rendered
with 256 accumulated frames at 1280×720.

| cornell_classic | cornell_classic_rec709 | cornell_empty |
|:---:|:---:|:---:|
| ![cornell_classic](screenshots/cornell_classic.png) | ![cornell_classic_rec709](screenshots/cornell_classic_rec709.png) | ![cornell_empty](screenshots/cornell_empty.png) |

| cornell_spheres | cornell_suzanne | cornell_textured_cube |
|:---:|:---:|:---:|
| ![cornell_spheres](screenshots/cornell_spheres.png) | ![cornell_suzanne](screenshots/cornell_suzanne.png) | ![cornell_textured_cube](screenshots/cornell_textured_cube.png) |

| openpbr_metals | openpbr_dielectrics | openpbr_coat |
|:---:|:---:|:---:|
| ![openpbr_metals](screenshots/openpbr_metals.png) | ![openpbr_dielectrics](screenshots/openpbr_dielectrics.png) | ![openpbr_coat](screenshots/openpbr_coat.png) |

| openpbr_fuzz | openpbr_specular | openpbr_organics |
|:---:|:---:|:---:|
| ![openpbr_fuzz](screenshots/openpbr_fuzz.png) | ![openpbr_specular](screenshots/openpbr_specular.png) | ![openpbr_organics](screenshots/openpbr_organics.png) |

| openpbr_thinfilm | openpbr_advanced | shaderball_base |
|:---:|:---:|:---:|
| ![openpbr_thinfilm](screenshots/openpbr_thinfilm.png) | ![openpbr_advanced](screenshots/openpbr_advanced.png) | ![shaderball_base](screenshots/shaderball_base.png) |

| shaderball_coat | shaderball_fuzz | shaderball_metal |
|:---:|:---:|:---:|
| ![shaderball_coat](screenshots/shaderball_coat.png) | ![shaderball_fuzz](screenshots/shaderball_fuzz.png) | ![shaderball_metal](screenshots/shaderball_metal.png) |

| shaderball_opacity | shaderball_specular | shaderball_subsurface |
|:---:|:---:|:---:|
| ![shaderball_opacity](screenshots/shaderball_opacity.png) | ![shaderball_specular](screenshots/shaderball_specular.png) | ![shaderball_subsurface](screenshots/shaderball_subsurface.png) |

| shaderball_thinfilm | shaderball_transmission | shaderball_emission |
|:---:|:---:|:---:|
| ![shaderball_thinfilm](screenshots/shaderball_thinfilm.png) | ![shaderball_transmission](screenshots/shaderball_transmission.png) | ![shaderball_emission](screenshots/shaderball_emission.png) |

| shaderball_checker | shader_ball | bunny_shaderball |
|:---:|:---:|:---:|
| ![shaderball_checker](screenshots/shaderball_checker.png) | ![shader_ball](screenshots/shader_ball.png) | ![bunny_shaderball](screenshots/bunny_shaderball.png) |

| camera_suzanne | ABeautifulGame | dragon_teapot |
|:---:|:---:|:---:|
| ![camera_suzanne](screenshots/camera_suzanne.png) | ![ABeautifulGame](screenshots/ABeautifulGame.png) | ![dragon_teapot](screenshots/dragon_teapot.png) |

---

## The repository family

Theia is the real-time renderer in a family of five repositories. Arrows point in the direction of
dependency:

```mermaid
flowchart LR
    SM["slang-math<br/>math"] --> A["Aether<br/>file format"]
    SM --> H
    A --> H["Harmonia<br/>shared Vulkan library"]
    H --> Hy["Hyperion<br/>offline path tracer · reference"]
    H --> T["<b>Theia</b><br/>real-time renderer"]
```

| Repository | Role |
|------------|------|
| [slang-math](https://github.com/McNopper/slang-math) | Math library, the leaf dependency |
| [Aether](https://github.com/McNopper/Aether) | File formats (scenes, materials, cameras, presets in TOML) and asset loaders; no Vulkan |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan library: application host, scene loading, shaders, the BSDF and path integrator both renderers call |
| [Hyperion](https://github.com/McNopper/Hyperion) | Offline path tracer; the reference Theia converges to |
| Theia | This repository: the real-time renderer |

Theia fetches Aether and Harmonia with CMake `FetchContent`, pinned to release tags in
`CMakeLists.txt`; slang-math arrives through Harmonia. The application is a thin subclass of the
shared `harmonia::app::App` host. Harmonia owns the window, swapchain, HDR image, scene loading,
accumulation, denoiser, tone mapping and presentation; Theia injects its renderer through the
`harmonia::app::IRenderer` interface and owns everything specific to a rasterizer: the meshlet
scene layout, the culling, the forward pass and the global-illumination pass. Code that both
renderers need identically belongs in Harmonia; Theia never includes Hyperion.

## How it works

### Converging to the reference

Theia and Hyperion evaluate the same integral with the same code: both call `integrateSurface` from
`Harmonia/shaders/path_integrator.slang` and import the same BSDF from
`Harmonia/shaders/bsdf_shared.slang`. Theia renders
each frame as one stochastic sample. The projection is jittered with a Halton (2,3) sequence, and
cutout coverage, light candidates and path directions are random. The shared accumulation stage
averages the frames for as long as the view is unchanged. In the interactive window and in a single
capture, a change of camera position, target, up vector or exposure, a new animation tick or a
resize restarts the accumulation (`ViewAccumulationGate` in `src/demo/ViewAccumulation.hpp`); a
sequence capture restarts it once per output frame. Convergence comes from the frame count and from
nothing else.

Resolution and frame count are the two knobs between speed and quality: lower them for iteration (the
parity budget is 320×240 with 256 frames), raise them for final images. The design assumes a GPU with
hardware ray tracing and ample memory, so ray-traced techniques are preferred over raster
approximations; slower hardware is served by scaling resolution and frame count down.

Both renderers take the following from shared code, so the two cannot drift apart: the path
integrator and the BSDF, scene parsing, camera assembly (`CameraAssembly`), exposure
(`PhysicalCamera`), the emissive-triangle list with its power-proportional CDF (`harmonia::scene_rt`),
and the host stages for accumulation and tone mapping. Only Theia rasterizes the primary surface
and jitters the projection once per frame with a Halton (2,3) sequence, where Hyperion draws a
random jitter for every sample in the shader; shades analytic lights in a forward pass; and
resamples light with reservoirs.

The renderer therefore has two output modes:

| | Interactive window | Capture (`--output`) |
|---|---|---|
| Purpose | a stable image at interactive frame rates | the raw estimator result, compared with Hyperion |
| A-SVGF denoiser | on | off |
| TAA | on while the camera moves | off (never created) |
| Firefly clamps | on | off |
| Secondary-bounce roughness regularization | on | off |
| Material LOD tier and radiance cache | on | off |
| Camera jitter | can be switched off with `--no-camera-jitter` | always on (`--no-camera-jitter` is ignored) |

Everything in the middle column is a presentation aid whose effect does not shrink with more
samples, so none of it is allowed on the converging path. Bright speckle in a capture is the true
variance of the estimator, not a defect. `--no-postfx` gives the interactive window the capture
identity (denoiser and TAA off). The shared part of this contract, the denoiser, is described in
[Harmonia's README](../Harmonia/README.md#denoiser-stage-and-the-two-tier-output-contract).

### Frame pipeline

```mermaid
flowchart TD
    T["Animation tick<br/>camera node, deformation, TLAS update"] --> L["Light culling<br/>LightCuller"]
    L --> C["Instance frustum cull<br/>GpuCullPass"]
    C --> P1["Opaque pass 1<br/>meshlets visible last frame, sky"]
    P1 --> Z["Hi-Z pyramid<br/>HiZPass"]
    Z --> P2["Opaque pass 2<br/>remaining meshlets, Hi-Z test"]
    P2 --> TP["Transparent pass"]
    TP --> GI["GI compute pass<br/>GiPass"]
    GI --> MV["Motion vectors<br/>MotionVectorPass"]
    MV --> TAA["TAA<br/>only while the camera moves"]
    TAA --> ACC["Accumulation<br/>shared stage"]
    ACC --> DEN["A-SVGF denoiser<br/>shared stage"]
    DEN --> TM["Tone map and present<br/>shared stage"]
```

The renderer-owned stages are recorded in `src/demo/Application.cpp`. Accumulation, denoiser and tone
mapping belong to Harmonia's `App`. The order is fixed: motion vectors and TAA are recorded in
`Application::onSceneFrameReady`, after the GI pass and all of its continuation batches have
completed, because both read the HDR image the GI pass writes. When the device has a dedicated
compute queue family, the GI pass is submitted to that queue and overlaps the next frame's raster
work (`src/demo/ApplicationCompute.cpp`). Frames that run manifold sampling use the single-queue
path, because their work spans several batches.

### Forward pass

Geometry is stored as meshlets of at most 64 vertices and 124 triangles, built with meshoptimizer.
Each frame, one task-shader workgroup per visible instance emits one mesh workgroup per meshlet
(`shaders/forward_render.task.slang`), and the mesh shader transforms the vertices
(`shaders/forward_render.mesh.slang`). The forward pass uses dynamic rendering
(`vkCmdBeginRendering`, no render pass objects) and writes four attachments:

| Attachment | Format | Content |
|---|---|---|
| HDR colour | RGBA32F | direct light and emission from the forward pass; the GI pass adds its terms. The alpha channel ends up holding the reservoir sample count that the denoiser reads |
| G-buffer | RGBA16F | view-space normal (xyz) and roughness (w) |
| GI buffer | RGBA32F | world-space position (xyz) and the instance index (w); 0 means background |
| Depth | D32 | depth |

The opaque pipeline writes depth. The transparent pipeline, used for instances with transmission or
with an opacity below 1, tests depth but does not write it and leaves the G-buffer untouched. A
fullscreen pass draws the environment panorama as background when the scene has an environment map.

`shaders/forward_render.frag.slang` shades a fragment with:

- the bindless material texture maps, sampled explicitly at mip level 0;
- a stochastic coverage test for `geometry_opacity`: the fragment is discarded with probability
  1 − α, so a cutout converges to the same image as in Hyperion;
- one-sided emission, visible only from the emitting side;
- direct lighting from the scene's analytic lights: point, spot and directional lights are
  evaluated analytically, rect lights are integrated numerically over an 11×11 grid. Each light is
  shadowed by a ray-traced shadow ray (an inline ray query that accumulates the transmittance of
  cutouts it crosses) and modulated by the shadow-terminator factor. The lights are binned per frame
  into 16×16-pixel screen tiles with at most 128 lights per tile (`LightCuller`,
  `shaders/light_cull.comp.slang`); directional lights go into every tile. When no tile data exists,
  the fragment loops over all lights.

Camera exposure (EV100) is applied here. Indirect lighting is left entirely to the GI pass. When
the scene declares no lights, emissive meshes become analytic rect or point lights at load
(`Scene::synthesizeEmissiveLights`); the forward pass skips these rect lights whenever the GI
pass's direct-light reservoir is active, which is the default, so that emissive direct lighting is
counted once.

### GPU culling and Hi-Z occlusion culling

The CPU never reads back how much is visible. Every count that the draws consume is written by the
GPU; the host only waits for completion.

- Instance frustum culling (`GpuCullPass`, `shaders/forward_cull.comp.slang`): a 64-thread compute
  group per 64 instances; one thread per
  instance tests the instance's world-space bounding box against the four side planes and the near
  plane of the view frustum. The bounding box is computed when the scene is built and recomputed
  and uploaded whenever an animation changes the instance's transform
  (`Scene::syncInstanceTransforms`), so a moved instance is culled at its current position.
  Before each dispatch the 12-byte draw command is zeroed with
  `vkCmdFillBuffer`; thread 0 of the shader restores the constant fields (`groupCountY = 1`,
  `groupCountZ = 1`). Every visible instance claims a slot in the compact instance list with one
  atomic add on the command, so its `groupCountX` equals the number of visible instances.
- One draw (`GpuDrivenState::issueDraw`): a single mesh-task draw whose command is
  `{visibleCount, 1, 1}`. With `VK_EXT_device_generated_commands` available this is
  `vkCmdExecuteGeneratedCommandsEXT` with one sequence over an indirect-commands layout of one
  `DRAW_MESH_TASKS_EXT` token of 12 bytes; otherwise it is `vkCmdDrawMeshTasksIndirectEXT`. A
  barrier between the cull and the draw makes the GPU-written count visible to both consumers. Both
  paths share the task-shader entry point, where workgroup `gid.x` indexes the compact instance list.

The draw is one sequence on purpose. With one draw per sequence, `SV_DrawIndex` is always 0, because
it counts within a sequence and not across sequences, so a sequence per instance would make every
sequence read entry 0 of the list and only instance 0 would be drawn. The workgroup id in the task
shader is what selects the instance.
- Two-pass Hi-Z occlusion culling (`HiZPass`, `shaders/hiz_build.comp.slang`,
  `shaders/forward_render.mesh.slang`): pass 1 draws only the meshlets that were visible in the
  previous frame. A max-depth pyramid (R32F, halving down to 1×1) is built from that depth. Pass 2
  draws the remaining meshlets whose world-space bounding sphere is not completely behind the
  pyramid. Two per-meshlet visibility buffers alternate between frames.
- Camera cuts: after a large view change (more than about 10° of rotation, or a camera move of
  more than half the distance from camera to target) the occlusion test is skipped for one frame,
  so that newly revealed meshlets are drawn.
- Meshlets of transparent instances are drawn without occlusion culling.

The compact instance list starts with 4096 entries and has no fixed upper limit.
`GpuCullPass::reserve(instanceCount)` grows it to the instance count of the scene when a scene is
set (`ForwardRenderer::setScene`, and once more after initialization for a scene that was set
earlier). The capacity never shrinks. A dispatch with more instances than the list holds culls only
what fits and logs an error once; `reserve` keeps that off the normal path.

Acceleration structures are built on the device only (`vkCmdBuildAccelerationStructuresKHR`); the
host-side build function is never used.

### Global-illumination pass

The GI pass (`GiPass`, `shaders/gi.comp.slang`) runs one 8×8 thread group per 64 pixels after the
forward pass and adds the remaining light terms to the HDR image. It first rebuilds the primary
surface of each pixel by re-tracing the camera ray to the position the rasterizer recorded. This is
order-independent for overlapping transparent surfaces, so the re-traced hit is authoritative.
Background pixels and pixels on an emitter are skipped. The acceleration structure is bound for
each frame: the periodic TLAS rebuild replaces it (see [Animation](#animation)), so the pass writes
the current one into the descriptor buffer of the frame slot it records, like the reservoirs; the
other scene bindings are written once per slot. Per pixel the pass computes:

- Direct light from emissive triangles, or from the environment when the scene has no emitters,
  with a ReSTIR direct-light reservoir. Reservoir spatiotemporal importance resampling (ReSTIR)
  keeps one candidate per pixel in a weighted reservoir and reuses candidates instead of drawing all
  of them anew: 8 initial candidates from the emissive power CDF or the environment CDF, temporal
  reuse from the previous frame's reservoir at the reprojected pixel (its sample count is capped at
  20), and, with ReSTIR PT on, spatial reuse from 5 random neighbours within 32 pixels that pass a
  position test (1 % of the view distance) and a normal test (dot ≥ 0.98) and are not duplicates.
  The temporal merge applies no surface test: it merges the previous reservoir whenever its sample
  lies above the current surface's horizon, so a pixel whose surface changed since the previous
  frame inherits candidates that were chosen for the old surface (see
  [Known limitations](#known-limitations)). One shadow ray is traced for the surviving candidate.
- Direct light from the environment at the primary surface through next-event estimation of the
  shared estimator, combined with BSDF sampling by multiple importance sampling. The environment is
  importance-sampled through its marginal and conditional CDF; the sky background is sampled raw.
- Indirect light from a multi-bounce path reservoir (ReSTIR PT). The reservoir sample of a pixel is
  the random-number seed of one full light path. Per pixel, 3 neighbours are resampled with it.
  Each neighbour path is either replayed at this pixel (the same random numbers, walked again from
  this surface) or reconnected: the recorded prefix is replayed and the last segment is attached to
  the new surface with a Jacobian that keeps the estimate unbiased. Neighbours are chosen by a
  compatibility score (distance and normal similarity, compatibility-guided neighbour selection)
  from 8 candidates on a disk of radius 32 pixels, and filtered by the same position, normal and
  duplicate tests. The reconnection vertex is the first pair of consecutive path vertices whose
  sampled lobes are both rough (roughness ≥ 0.2) and that lie at least 0.1 scene units apart, up to
  the fourth vertex; when no pair qualifies, the path is replayed. The path reservoir has spatial
  reuse only and no temporal merge (see [Conventions](#conventions)).

The pass runs as three kernels sharing one pipeline layout. `main` draws the candidates, tests the
neighbours and attempts the shifts. Replay walks are appended to a GPU-compacted stream and executed
uniformly by `replayMain`, which is dispatched indirectly with the GPU-written count. `combineMain`
merges the candidates in their original order and adds the indirect term.

Two further tiers run in the interactive window only, because both are biased by design; they are
listed in the two-mode table above:

- Material LOD: rays spawned from rough, non-specular lobes evaluate a reduced material (no coat or
  fuzz) at diffuse secondary hits and sample mip levels chosen from the ray-cone footprint, capped
  at level 6. Delta, glass, near-specular and manifold-sampled rays and the first secondary always
  get the full material.
- Radiance cache: a hashed table of 2¹⁷ world cells (2 MiB) holds an exponential moving average
  (α = 0.25) of the outgoing radiance seen by finished walks. Diffuse secondary hits from the first
  bounce on may terminate with the cached value.

Transmission, refraction, subsurface scattering and scattering media are evaluated by the shared
path integrator and its volumetric random walk (`runMediumWalk`) on the primary and on every
secondary vertex; pure absorbers take the exact Beer–Lambert branch.

Where the denoiser's variance buffer is available, up to 4 samples per pixel are spent in
high-variance pixels, and the firefly clamp is relaxed there.

### Specular manifold sampling

Specular manifold sampling (SMS) adds samples that travel along chains of specular events (mirrors,
glass interfaces) toward an emitter, which lowers the variance of caustics that plain path tracing
rarely finds. It follows PSMS-ReSTIR (Hong et al., SIGGRAPH Asia 2025). The sampler, the shift and
the resampling are shared with Hyperion through Harmonia; Theia contributes the scene adapter
(`shaders/sms_scene.slang`, which reads Theia's own descriptor layout) and the frame wiring
(`src/theia/gi/GiPassSms.cpp`).

Manifold sampling runs only on scenes that have emissive triangles and a `max_depth` of at least 3.
`--sms-mode local` (the default) evaluates one independent receiver chain per pixel; `reuse` adds a
temporal merge and spatial reservoir reuse and is experimental. The work runs in resumable batches
on the graphics queue; the host drains the batches and waits for completion before it accumulates,
and never accumulates an unfinished estimate.

### Denoiser, TAA and accumulation

- Motion vectors (`MotionVectorPass`, `shaders/motion_vector.comp.slang`) reproject each pixel into
  the previous frame's view-projection. Background pixels reproject as infinitely distant points,
  and instance motion is undone with the previous instance transforms.
- TAA (`TaaPass`, `shaders/taa.comp.slang`) runs only in the interactive window and only in frames
  in which the view changed. It reprojects the history, clamps it to the 3×3 neighbourhood in
  YCoCg space and blends 90 % history with 10 % of the current frame. A stationary camera is
  stabilised by accumulation instead.
- The shared A-SVGF denoiser is an à-trous wavelet filter guided by the normal and depth buffers,
  the motion vectors and a gradient/variance buffer. Theia raises its default strength from 0.45 to
  0.55 while ReSTIR PT is on, because the spatial reuse correlates neighbouring pixels; an explicit
  `--denoiser-strength` is kept. The GI pass writes the reservoirs' effective sample count into the
  HDR alpha channel, and the denoiser weights its temporal blend with it.
- Accumulation is Harmonia's shared stage.

### Animation

Animated scenes (transform tracks, skinning, morph targets, a camera bound to a node) advance on one
tick clock, `App::animationClock`, which provides a time in seconds and a tick index. On a new tick
Theia stamps the scene, runs the shared deformation pass (linear-blend or dual-quaternion skinning
and morph targets) and updates the TLAS: a refit in place on each update, and a full rebuild on
every 128th update of an animated scene and on every update of a scene that was not built for
updates. A rebuild creates a new acceleration structure, so the handle and the device address
change; the GI pass therefore binds the current structure for each frame, and the forward pass
rewrites its own binding each frame. In the interactive window a new tick also restarts the
accumulation; a sequence capture accumulates each output frame on its own. The camera controller is
disabled while the camera is bound to a node. Skinned and morphed meshes are not culled until a
per-frame bounds update exists (see [Known limitations](#known-limitations)).

### Material model

Theia uses OpenPBR Surface v1.1.2 (Academy Software Foundation) through the shared BSDF in
`Harmonia/shaders/bsdf_shared.slang`. The layer stack (base, specular, coat, fuzz, emission, thin film,
transmission, subsurface) and the volumetric subsurface and transmission walk are described in
[Harmonia's README](../Harmonia/README.md#material-model--openpbr-surface-v112). Theia adds only how
the material is evaluated on the rasterized primary:

- `geometry_opacity` is a presence weight α. The rasterizer discards a fragment with probability
  1 − α, shadow rays multiply the transmittance by (1 − α) for every cutout they cross, and the GI
  pass applies the same coverage test at the primary vertex. A mesh may carry an authored opacity
  micromap. It only accelerates ray traversal and does not change the result; a scene that uses one
  needs `VK_KHR_opacity_micromap`.
- The rasterized primary reads the base-colour, normal, ORM, emission and opacity maps at mip level
  0, as Hyperion does. The other `map_*` keys are resolved by Harmonia's shared surface module on
  the ray-traced paths.
- Textures are bindless: one combined image–sampler array with up to 256 entries
  (`kMaxBindlessTextures` in `src/theia/renderer/RendererConstants.hpp`), bound to the forward pass
  as descriptor set 3 and to the GI pass at binding 13, and indexed with
  `NonUniformResourceIndex`.

### Colour pipeline

Scenes are rendered scene-referred in the working colour space the scene selects
(`working_color_space`, linear Rec.2020 or linear Rec.709). Exposure follows EV100 (`ev100`), the
environment map is scaled by `environment_unit_nits`, and tone mapping (AgX, ACES, Reinhard, Hable)
and display output (SDR, HDR10, scRGB) are shared Harmonia stages; see
[Harmonia's README](../Harmonia/README.md#scope). A capture writes the scene-referred, untonemapped
EXR and a PNG produced by the same tone-mapping stage the window uses.

---

## Running

```powershell
build/theia.exe --scene cornell_classic                               # interactive window
build/theia.exe --scene cornell_classic --output out.exr               # headless capture
build/theia.exe --scene shaderball_base --width 640 --height 480      # smaller window
```

There is no `--offscreen` flag: a capture starts when `--output` is given. The window stays black
while the scene uploads and the pipelines are built (several seconds on heavy scenes such as
`ABeautifulGame`), and every resize rebuilds the passes at the new resolution and restarts the
accumulation.

### Flags parsed by Theia

These flags are read in `src/demo/main.cpp`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--sms-mode off\|local\|reuse` | `local` | Manifold sampling: `off` is plain path tracing, `local` evaluates one independent receiver chain per pixel, `reuse` adds a temporal merge and spatial reservoir reuse (experimental) |
| `--no-restir-pt` | ReSTIR PT on | Falls back to the direct-light reservoir alone (temporal reuse only) and the classic per-sample multi-bounce walk |
| `--no-restir-pt-path` | on | Keeps ReSTIR PT but replaces the path reservoir with the classic per-sample multi-bounce walk |
| `--no-restir-di` | on | Disables the direct-light reservoir. It has no effect while ReSTIR PT is on, because that reservoir already holds the direct-light candidates. Together with `--no-restir-pt`, the forward pass's rect lights remain the only emissive direct lighting |
| `--taa`, `--no-taa` | on | TAA in the interactive window. `--taa` combined with `--output` is rejected (exit code 2), because TAA would double-filter the accumulation |
| `--no-camera-jitter` | jitter on | Disables the sub-pixel camera jitter in the interactive window; ignored in a capture |

An unknown argument is an error (exit code 2).

### Shared flags Theia relies on

The remaining flags are parsed by Harmonia's `CliParser`. The complete list, including the
`--denoiser-*` options, is in [Harmonia's README](../Harmonia/README.md#command-line-flags).
The ones that matter for Theia:

| Flag | Default in Theia | Meaning in Theia |
|------|------------------|-------------------|
| `--scene <name\|path>` / `-s` | `cornell_classic` | A bare name resolves against the assets directory (below); a bare argument without a flag also names the scene |
| `--output <file>` / `-o` | — | Capture mode: accumulate `--offscreen-frames` frames, write, exit. With a `.png` path only the PNG is written; with any other path an EXR at that path and a PNG with the same stem |
| `--offscreen-frames <n>` | `4` | Frames accumulated in a capture (at least 1). This is the convergence quality |
| `--width <n>`, `--height <n>` | `1024`, `768` | Window size, and the image size of a capture |
| `--frames-per-flip <n>` | `1` | Accumulation frames per swapchain flip in the window |
| `--sequence-frames <N>`, `--frame <n>`, `--capture-every-ms <f>`, `--start-seconds <f>` | off, `0`, –, – | Animation sequences and time-based captures |
| `--rng-seed <n>` | `305419896` (`0x12345678`) | Base seed of the stochastic stages. With the same binary and scene, the same seed, frame count and resolution reproduce a capture pixel for pixel |
| `--no-postfx` | off | Denoiser and TAA off in the window |
| `--validation` / `--no-validation` | on | Vulkan validation layers |
| `--indirect-ambient <f>` | `0.0` | Accepted by the shared parser and forwarded to the forward pass's push constants; no Theia shader reads it, so it does not change the image |
| `--deterministic-replay`, `--rng-debug` | off | Accepted by the shared parser; no Theia shader reads the flags they set |

Scenes and assets are resolved relative to the Aether assets directory that CMake baked into the
executable (`THEIA_ASSETS_DIR`). Without an override that is the clone that FetchContent makes of
Aether inside the build directory; with `-DFETCHCONTENT_SOURCE_DIR_AETHER=<path>` it is that working
tree. Editing any other Aether checkout has no effect on the render, and two "different" edits then
give identical metrics.

### Interactive window

```text
WASD          move the camera
Right mouse   hold to look around
Q / E         move down / up
Mouse wheel   slower / faster movement
[ / ]         decrease / increase exposure (EV100) by 0.5 stops
ESC           release the mouse; press again to quit
```

A stationary camera accumulates; moving it shows TAA-stabilised single samples. The window title
shows the frame rate, frame time and EV100.

### Debug environment variables

All of them are read once at start-up. "Set" means any value except an empty one or one starting
with `0`. `THEIA_SMS_DIAGNOSTICS` and `THEIA_PASS_TIMING` take exactly the value `1`; every other
value keeps the feature off.

| Variable | Default | Meaning |
|----------|---------|---------|
| `THEIA_FORCE_GD3` | unset | Set: skip device-generated commands and use the indirect mesh-task draw |
| `THEIA_SINGLE_PASS` | unset | Set: bypass the two-pass occlusion culling; one opaque pass draws every meshlet |
| `THEIA_DISABLE_HIZ` | unset | Set: keep the two passes but skip the occlusion test, so every meshlet is drawn |
| `THEIA_NO_RECONNECTION` | unset | Set: disable the reconnection shift; every reused path is replayed in full |
| `THEIA_SHIFT_MASK=<n>` | `3` | Bit mask of the reconnection shift. `0`: replay only. Bit 1 (`2`): hybrid rule for the reconnection vertex (see the GI pass); without it the first secondary vertex is always used. Bit 2 (`4`): allow the multi-hop shift for reconnections at the fourth vertex; without it those records fall back to replay. Bit 0 (`1`) is not read on its own, so `2` and `3` behave alike |
| `THEIA_BIAS_CORRECTION=<n>` | `1` | Bias correction of the direct-light reservoir's temporal and spatial merges. `1` (default) and `0` compute the same formula, `w_sum / (p_hat · M)`, which is the `OFF` mode of NVIDIA's RTXDI. `2` additionally rejects a reused sample that its source pixel could not have generated (one conservative visibility test per source). Larger values are clamped to `2` |
| `THEIA_MATERIAL_LOD=<0\|1>` | on in the window, off in captures | Force the material LOD tier |
| `THEIA_RADIANCE_CACHE=<0\|1>` | on in the window, off in captures and for animated or deforming scenes | Force the radiance cache |
| `THEIA_SMS_PARTITION=<1..32>` | `16` | Resolution of the trust-region partition that builds the manifold-sampling prior; `1` removes the partition. Out-of-range values keep the default |
| `THEIA_SMS_DIAGNOSTICS=1` | unset | Allocate a 64-slot diagnostics buffer and log manifold-sampling counters after each frame |
| `THEIA_DEBUG_RAY_HIT_MODE=<f>` | `0` | Debug view of the primary ray hit, drawn by the forward pass instead of its shading. `1`: base colour of the re-traced hit. `2`: its shaded radiance without normal maps, which is direct light and emission only, so a scene lit only by the environment renders black. `3`: on the transparent pass the material index as colour (0 red, 1 green, 2 blue, anything else yellow); on the opaque pass it behaves as `2`. Clamped to 0..6; values above 3 have no effect |
| `THEIA_PASS_TIMING=1` | unset | Log the GPU time of the GI, motion-vector and TAA passes once per frame, for example `timing: gi 4.32 ms \| motion 0.10 ms \| taa 0.62 ms` |

The toggles of the shared library that Theia runs with — the pipeline binary cache
(`HARMONIA_PIPELINE_BINARY_CACHE`), frame timing and present-timing telemetry — are documented in
[Harmonia's README](../Harmonia/README.md#environment-variables).

### Captures

- A capture is the raw estimator result (see the two-mode table above). Raise `--offscreen-frames`
  to converge; 256 frames at 320×240 is the budget the parity thresholds are calibrated for.
- An animated scene captured with `--output` but without `--sequence-frames` restarts the
  accumulation on every tick; the result is a single-sample image of the last tick. Theia logs a
  warning that says so once.
- Motion blur: with Harmonia's `--shutter <0..1>` and `--time-samples <n>` the accumulation frames
  of an output frame are spread over the frame's exposure and the animation is advanced to each
  time sample, so one capture integrates the motion blur. All time samples of an output frame
  accumulate together. The scene needs an animation, the capture `--sequence-frames`, and at
  least `n` accumulation frames per output frame. A shuttered frame is compared with a Hyperion
  frame rendered with the same two flags and the same `--rng-seed`: the position of each time
  sample inside its interval depends on the seed, and two seeds give two different estimates of
  the same blur. The Aether scene `motion_blur` is made for this comparison.
- The capture lowers its own CPU priority for its duration so that the desktop stays usable.

---

## Scenes

Theia loads every scene of the Aether asset collection, the same scenes Hyperion renders. The file
format (`.scene.toml`, `.materials.toml`, OBJ geometry, TOML presets) is described in
[Aether's README](../Aether/README.md#file-format-reference); materials carry `model = "openpbr"`.
Theia has one requirement of its own: a scene must contain a camera placement, that is a `[camera]`
table with `translate` and a rotation, inline or through a preset that provides both. Harmonia's
scene loader rejects a scene without one; a preset that only sets `vertical_field_of_view` or
`ev100` does not satisfy the requirement.

| Use | Scenes |
|-----|--------|
| Quick iteration (small Cornell boxes) | `cornell_classic`, `cornell_spheres`, `cornell_suzanne`, `dragon_teapot` |
| Parity comparison (14 scenes) | the Cornell set, `dragon_teapot` and the `shaderball_*` and `openpbr_*` fixtures listed in `Harmonia/tools/validation_manifest.toml` |
| Screenshot gallery (30 scenes) | the list in `tools/generate_gallery.ps1` |
| Expensive | `ABeautifulGame`: avoid it for quick test renders; it is used only in gallery renders |

---

## Building

Requirements: the Vulkan SDK 1.4 (`VULKAN_SDK` must be set; it provides SDL3, slangc, volk and the
VMA header), CMake 3.28+, Ninja, vcpkg (tomlplusplus, OpenImageIO, meshoptimizer) and either clang-cl
in a Visual Studio command prompt or g++ on Linux. The GPU needs Vulkan 1.4, hardware ray tracing
and mesh shaders: `VK_EXT_mesh_shader` is required, `VK_EXT_device_generated_commands` is optional.
The features and extensions the device needs are listed in
[Harmonia's README](../Harmonia/README.md#device-layer).

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

The same configure, build and test steps exist as presets: `cmake --preset win`,
`cmake --build --preset win`, `ctest --preset win`.

On Linux, configure with `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` and run the tests with
`ctest --test-dir build`. The Linux SDK does not contain SDL3; the distribution package is linked
into the SDK directory. The recipe is in
[Harmonia's README](../Harmonia/README.md#linux). macOS is not supported.

To develop against working copies of the sibling repositories instead of the pinned release tags,
add `-DFETCHCONTENT_SOURCE_DIR_AETHER=<path>` and `-DFETCHCONTENT_SOURCE_DIR_HARMONIA=<path>` to the
configure (CI also sets `FETCHCONTENT_SOURCE_DIR_SLANG_MATH`).

Slang shaders in `shaders/` are compiled at build time into `build/shaders/` by Harmonia's
`compile_slang_shaders` rule and loaded from that directory through the `THEIA_SHADER_DIR` compile
definition, never from the working directory.

---

## Testing and verification

### Module tests

```powershell
cd build
ctest --output-on-failure      # module suite and analysis tests
ctest -LE analysis              # module suite only
```

The module suite is one GoogleTest binary, `theia_module_tests` (`tests/module/`). It creates a
hidden window and one Vulkan device with validation enabled, and covers:

- scene build (meshlets, spheres, emissive-light synthesis) and the GPU cull pass (the compact list
  grows for a scene past its initial capacity and is never written past its end);
- animation (TLAS refit, instance transform synchronisation);
- CPU contracts: RNG streams, Halton camera jitter, environment sampling, BSDF sampling, the
  accumulation-reset decision, the `--frames-per-flip` flag and the pass-timing toggle;
- manifold-sampling fixtures: the scene adapter, the shift, the resampling weights and whole GI
  frames compared with independent reference integrals.

Tests that need the device skip when no Vulkan context can be created. Every test is registered
with `RUN_SERIAL`.

### Static analysis and checks

| Command | What it does | Exit codes |
|---------|--------------|------------|
| `python tools/check_arch.py` | Measures the include graph of `src/` against the contract in `ARCHITECTURE.md`: declared edges only, layer order, no cycles, no file over 1000 lines | 0 clean, 1 violations, 2 usage error, 77 skipped |
| `python tools/check_dead.py --family ..` | Finds orphan headers, sources that are never compiled and FetchContent declarations that are never consumed; with `--family`, a header used by a sibling repository is not an orphan | 0 clean, 1 findings, 77 nothing scanned |
| `python tools/check_tidy.py` | Runs clang-tidy over `build/compile_commands.json`; `clang-diagnostic-*`, `clang-analyzer-*` and `bugprone-*` findings fail, the rest are reported; the log is `build/clang_tidy.log` | 0 clean, 1 findings, 77 toolchain or database missing |
| `cmake --build build --target verify-full` | Build, `ctest`, `format-check`, `cppcheck` and `tidy` | non-zero on any failure |
| `python tools/coverage.py` | Builds an instrumented tree configured with `-DTHEIA_COVERAGE=ON`, runs the tests and prints the coverage report for `src/` | 0 report, 1 failure, 77 LLVM tools missing, 2 usage |

`ctest` registers `check_tidy.py` as `test_tidy`, and the architecture and dead-code checks as
`test_arch` and `test_dead` (all with label `analysis`). The latter two are registered only where the
per-user check-arch and check-dead engines are installed, so CI does not run them. Sanitizers are
available for Clang and GCC configures with `-DTHEIA_SANITIZER=address`, `undefined` or `thread`.
The tools are described in [tools/README.md](tools/README.md).

### Lint and continuous integration

`.github/workflows/build.yml` runs on pushes to `main`, on pull requests targeting `main` and on
manual dispatch:

- A blocking lint job: clang-format pinned to 22.1.3, cppcheck over `src/` (no `error` finding
  allowed), ruff, bandit, actionlint and yamllint. A different clang-format version reformats
  differently and fails, so run the same commands locally before pushing.
- Build jobs for Windows (clang-cl) and `ubuntu-26.04` (g++). They build only, because the tests
  need a GPU. They check out the sibling repositories at their `main` heads through
  `FETCHCONTENT_SOURCE_DIR_*` (for the development branches dev/gi-sms-restir and
  wip/gi-sms-spatial-reuse, Harmonia at the branch of the same name), and fetch Aether without its
  LFS assets.

A local Windows build does not prove a Linux build or a passing CI run; build and lint locally
before pushing.

### Parity against Hyperion

Parity compares a Hyperion EXR (the reference) with a Theia EXR (the candidate). Both are pre-tonemap
linear EXRs at the same resolution and in the same working colour space, over the 14 scenes of
`Harmonia/tools/validation_manifest.toml` at 320×240, with 256 samples per pixel from Hyperion and
256 accumulated frames from Theia. The comparison passes only when all four thresholds hold: mean
difference ≤ 120 (in 1/255 units), relative MSE ≤ 250, SSIM ≥ 0.6 and luminance-histogram
correlation ≥ 0.8. The thresholds belong to this budget and are recalibrated when it changes. The
method and the scripts (`Harmonia/tools/compare_renders.py`,
`Harmonia/tools/render_and_validate.py`, `Harmonia/tools/check_vulkan_validation.py`) are described
in [Harmonia's README](../Harmonia/README.md#parity-methodology) and
[Harmonia's tools/README.md](../Harmonia/tools/README.md); the measured values behind the
thresholds are in [Harmonia's PLAN.md](../Harmonia/PLAN.md#7-parity-methodology).

A reference must be clean: render Hyperion at high sample counts (`hyperion --spp 256`). A
low-sample reference is noisy and inflates the difference; see
[Aether's AGENTS.md](../Aether/AGENTS.md#render-presets--sample-counts-the-low-spp-reference-trap).

### Screenshots

`tools/generate_gallery.ps1` renders the 30 gallery scenes at 1280×720 with 256 frames into
`screenshots/`. Hyperion's gallery of the same scenes uses 64 samples per pixel, because 256 is too
slow for the whole set and fireflies are acceptable in a showcase image. A parity reference is a
separate artifact: a clean Hyperion EXR at 256 samples per pixel and 320×240. A release regenerates
the gallery and runs the Vulkan validation check over it.

### Running on a shared machine

Tests are serialised and the GPU jobs (renders, the gallery, parity comparisons) saturate the GPU.
Run one job at a time and one repository's test suite at a time. A capture lowers its own CPU
priority, which helps during the capture but is no licence for two captures at once.

---

## Conventions

- Defects are fixed in the code that owns them. A tolerance is never loosened, and no special case
  or clamp is added, to make a test pass. A compiler warning fails the build and a Vulkan
  validation message is a bug.
- Behaviour is parameter-driven. A change in behaviour flows through a flag, a config struct, a
  push constant or a `THEIA_*` toggle, and each knob is documented where it lives.
- The path reservoir has no temporal merge, and none is added back. A streamed temporal merge
  re-selects history and double-counts it on heavy-tailed path candidates; measured, it rendered
  cornell indirect lighting 2.8 % too dark at 1024 frames, and capping the sample count does not
  help. Spatial reuse reads the neighbours' unmerged local reservoirs, and accumulation already
  integrates over time. The direct-light reservoir's temporal merge has the same latent bias, which
  is invisible on light-tailed direct lighting.
- The denoiser, TAA and the interactive tiers are presentation stages, never part of the estimator.
  A technique belongs on the converging path only if its contribution vanishes as the sample count
  grows, or if it lives in the interactive-only tier that capture mode bypasses. Neural and
  AI-based techniques are not used.
- Multi-frame captures are reproducible run to run, and stay that way. Theia's random numbers come
  from the pixel, the frame index, the bounce and the base seed. `GiPass` needs its per-frame-slot
  descriptor buffers and the reservoir barrier in `GiPass::record`; both protect in-flight frames.
- The GPU decides what is drawn. Dispatch counts and per-draw indices are GPU-resident, and the host
  reads back only completion and error status.
- Vulkan features are core or `KHR`/`EXT` only. Vendor extensions (`VK_NV_*`, `VK_AMD_*`,
  `VK_INTEL_*`) are not used. An optional capability is probed and enabled, and its absence must
  leave the image identical.
- Code shared 1:1 with Hyperion goes down into Harmonia; Theia never depends on Hyperion. Includes
  follow the contract in [ARCHITECTURE.md](ARCHITECTURE.md): declared edges only, no cycles, and no
  source file above 1000 lines (split along a seam, as `Application.cpp` is split into
  `ApplicationCompute.cpp`). Theia's classes live in `namespace theia`.
- A new compute entry point is added to `THEIA_ENTRY_SHADERS` in `CMakeLists.txt`; without it the
  SPIR-V is never compiled and the shader fails to load at runtime. A new Slang module that is not
  an entry point is added to `COMPILE_SLANG_EXTRA_DEPENDS` there as well.
- Flags specific to Theia go into `src/demo/main.cpp`; flags shared with Hyperion go into
  Harmonia's `CliParser`.
- The contracts of the shared BSDF apply unchanged: hit records carry the raw outward-winding
  normal (`GiHit.geoNormal` is never pre-flipped; consumers flip a local copy by the outgoing
  direction), the shadow-terminator factor is applied at every next-event-estimation and
  continuation site, transmission tints follow MaterialX, pure absorbers use the exact Beer–Lambert
  law, and acceleration structures are built on the device only.
- Work is committed per repository and pushed only on request. Dependencies move in coordinated
  releases in the order slang-math, Aether, Harmonia, Theia; the release commit bumps the
  `FetchContent` pins.

---

## Repository layout

| Path | Content |
|------|---------|
| `src/demo/` | The application: `Application.cpp` (frame recording, input, resize), `ApplicationCompute.cpp` (GI dispatch, async-compute path, camera placement, `buildGiFrameParams`), `main.cpp` (argument parsing), `ViewAccumulation.hpp`, `EnvImportanceResources.hpp` |
| `src/theia/scene/` | `Scene`: meshlet build, GPU buffers, BLAS and TLAS build and update |
| `src/theia/camera/` | `CameraController.hpp` (first-person camera), `CameraJitter.hpp` (Halton jitter) |
| `src/theia/depth/` | `HiZPass`: the depth pyramid |
| `src/theia/cull/` | `GpuCullPass` (frustum cull), `GpuDrivenState` (draw issue, meshlet visibility), `LightCuller` (Forward+) |
| `src/theia/renderer/`, `src/theia/pipelines/` | `ForwardRenderer` (the rasterizer) and `PassTimer` |
| `src/theia/gi/` | `GiPass`: the GI compute stage and its manifold-sampling wiring |
| `src/theia/postfx/` | `MotionVectorPass`, `TaaPass` |
| `shaders/` | The entry shaders and shader modules; compiled to `build/shaders/` |
| `tests/module/`, `tests/shaders/` | The module suite and its test shaders |
| `tools/` | Analysis and qualification scripts, see [tools/README.md](tools/README.md) |
| `screenshots/` | The gallery |
| `CMakeLists.txt` | Build graph, dependency pins, `THEIA_ENTRY_SHADERS` |

The module boundaries and the allowed dependencies are defined in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Known limitations

- Skinned and morphed meshes skip frustum and occlusion culling until a per-frame bounds update
  exists.
- Motion vectors cover instance transforms only. Per-vertex deformation is not in the motion
  vectors, so TAA can ghost on fast deformation; the accumulation renders each animation state on
  its own and is not affected.
- The manifold-sampling reuse mode is experimental.

The open items and the measured baselines are listed in [PLAN.md](PLAN.md).

## Where to read more

| Document | Content |
|----------|---------|
| [AGENTS.md](AGENTS.md) | Short orientation: where things are and which mistakes are easy to make |
| [PLAN.md](PLAN.md) | Outstanding work, working rules, measured baselines and the release history |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The layers, modules and the machine-checked include contract |
| [tools/README.md](tools/README.md) | Reference of every script in `tools/` |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Notes of the recent releases |
| [Harmonia README](../Harmonia/README.md) | Command-line flags, scene loading, the material model, the shared shaders, the denoiser and the two-tier output contract, the device layer and the pipeline binary cache, the parity methodology |
| [Harmonia tools](../Harmonia/tools/README.md) | The parity, validation and source-checking tools |
| [Harmonia PLAN.md](../Harmonia/PLAN.md) | The estimator roadmap and the measured parity method (section 7) |
| [Aether README](../Aether/README.md) | The scene, material and preset file formats |
| [Aether AGENTS.md](../Aether/AGENTS.md) | The asset tree and render presets |

## Dependencies

| Library | Purpose |
|---------|---------|
| [Aether](https://github.com/McNopper/Aether) | Scene and material file formats (`.scene.toml` / `.materials.toml` / OBJ), GPU-agnostic CPU data |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation (`harmonia::app::App` host, core, presentation, color, tonemapping, shared GPU types) |
| [Vulkan SDK](https://vulkan.lunarg.com/) | Vulkan 1.4 API, SDL3, volk, VMA, Slang compiler |
| [volk](https://github.com/zeux/volk) | Vulkan loader |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory allocation (via Harmonia) |
| [SDL3](https://libsdl.org/) | Window management and surface |
| [slang-math](https://github.com/McNopper/slang-math) | Mathematics, via Harmonia (transitive FetchContent) |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | Meshlet generation and mesh optimization |
| [OpenImageIO](https://openimageio.readthedocs.io/) | Image I/O: PNG/JPEG/EXR load and save (via Harmonia; stb and OpenEXR are transitive dependencies) |
| [Slang](https://shader-slang.com/) | Shader compilation (Slang → SPIR-V) |

## References

The BSDF, volumetric, colour-science and denoising references are listed in
[Harmonia's README](../Harmonia/README.md#references), the shared implementation. Below are the
references specific to a real-time renderer.

### Real-time rendering and GPU-driven

| Resource | Relevance |
|----------|-----------|
| Haar & Aaltonen — ["GPU-Driven Rendering Pipelines" (SIGGRAPH 2015, Advances course)](https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf) | Indirect draw, GPU command generation, per-cluster culling |
| Khronos — [Mesh Shading for Vulkan](https://www.khronos.org/blog/mesh-shading-for-vulkan) | `VK_EXT_mesh_shader` task/mesh pipeline, `EmitMeshTasksEXT` |
| [Meshoptimizer — Arseny Kapoulkine](https://github.com/zeux/meshoptimizer) | `meshopt_buildMeshlets`, `meshopt_computeMeshletBounds`, cone culling |
| Bitterli, Wyman, Pharr, Shirley, Lefohn & Jarosz — ["Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting" (ACM TOG / SIGGRAPH 2020)](https://doi.org/10.1145/3386569.3392481) | ReSTIR for emissive-triangle direct lighting |
| Lin, Kettunen & Wyman — ["ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing" (PACMCGIT 2026)](https://doi.org/10.1145/3804494) | The multi-bounce path reservoir and its reconnection shift |
| Junkins, Kettunen, Lin, Ramamoorthi & Wyman — [Compatibility-Guided Neighbor Selection (HPG 2026)](https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/) | Neighbour choice for spatial reuse |
| Hong et al. — ["Sample Space Partitioning and Spatiotemporal Resampling for Specular Manifold Sampling" (SIGGRAPH Asia 2025)](https://doi.org/10.1145/3757377.3763927) | Manifold sampling with reservoir reuse |
| Gribb & Hartmann — ["Fast Extraction of Viewing Frustum Planes from the World-View-Projection Matrix" (2001)](https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf) | Frustum plane extraction in the cull shader |
| Olsson & Assarsson — ["Tiled Shading" (Journal of Graphics, GPU, and Game Tools 15(4), 2011)](https://doi.org/10.1080/2151237X.2011.621761) | Forward+ light lists |

### Vulkan and API

| Resource | Relevance |
|----------|-----------|
| [Vulkan Specification 1.4](https://registry.khronos.org/vulkan/specs/latest/html/) | `vkCmdBeginRendering`, mesh shaders, ray queries, descriptor indexing, timeline semaphores |
| [Vulkan Roadmap 2024](https://docs.vulkan.org/spec/latest/appendices/roadmap.html) | Khronos feature baseline for Theia (no vendor extensions) |
| [Khronos — VK_EXT_mesh_shader Specification](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html) | Task shader payload, `EmitMeshTasksEXT`, output topology |
| [Khronos — Deprecating host-side acceleration structure builds](https://www.khronos.org/blog/vulkan-ray-tracing-deprecating-host-side-acceleration-structure-builds) | Why acceleration structures are built on the device only |
| [Slang Shading Language](https://shader-slang.com/) | Mesh shader `[outputtopo]`, task payload, `DispatchMesh`, SPIR-V emission |

### Scene and asset formats

| Resource | Relevance |
|----------|-----------|
| [Wavefront OBJ](http://paulbourke.net/dataformats/obj/) | Geometry-only OBJ import (no MTL; materials are assigned in the scene TOML) |
