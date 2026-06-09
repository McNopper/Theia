# Theia

GPU-driven real-time renderer for OpenPBR materials.

> *[Theia](https://en.wikipedia.org/wiki/Theia_(mythology)) — Titaness of heavenly light, mother of Helios, Selene and Eos.*

Theia is a modern Vulkan 1.4 renderer built on GPU-driven forward rendering techniques.  
It implements the [OpenPBR Surface v1.1](https://academysoftwarefoundation.github.io/OpenPBR/) material model in real-time, delivering **60 FPS at 4K** with visual parity to [Hyperion](https://github.com/McNopper/Hyperion) on identical test scenes.

**Interactive real-time rendering** — explore complex materials, dynamic lighting, and HDR output in real-time.  
**Architecture driven by [GPU-Driven Rendering](https://vkguide.dev/docs/gpudriven)** — compute-based culling, indirect dispatch, and clustered lighting.

> ⚠️ **Early stage / work in progress.** Theia is under active development. APIs, rendering
> techniques, and visual output are still evolving, and some features are incomplete or
> approximate. Expect rough edges and breaking changes.

---

## Screenshots

*Same test scenes as [Hyperion](https://github.com/McNopper/Hyperion) — rendered in real-time with visual parity.*

| Cornell Box | Spheres | Suzanne |
|:-----------:|:-------:|:-------:|
| ![cornell_classic](screenshots/cornell_classic.png) | ![cornell_spheres](screenshots/cornell_spheres.png) | ![cornell_suzanne](screenshots/cornell_suzanne.png) |

| Metals | Dielectrics | Coat |
|:------:|:-----------:|:----:|
| ![openpbr_metals](screenshots/openpbr_metals.png) | ![openpbr_dielectrics](screenshots/openpbr_dielectrics.png) | ![openpbr_coat](screenshots/openpbr_coat.png) |

| Fuzz | Specular | Organics |
|:----:|:--------:|:--------:|
| ![openpbr_fuzz](screenshots/openpbr_fuzz.png) | ![openpbr_specular](screenshots/openpbr_specular.png) | ![openpbr_organics](screenshots/openpbr_organics.png) |

| Thin-film | Meadow IBL | Textured Cube |
|:---------:|:----------:|:-------------:|
| ![openpbr_thinfilm](screenshots/openpbr_thinfilm.png) | ![meadow_scene](screenshots/meadow_scene.png) | ![textured_cube](screenshots/textured_cube.png) |

| A Beautiful Game (IBL) | Dragon & Teapot (IBL) | Advanced Transmission |
|:----------------------------------:|:---------------------:|:---------------------:|
| ![ABeautifulGame](screenshots/ABeautifulGame.png) | ![dragon_teapot](screenshots/dragon_teapot.png) | ![openpbr_advanced](screenshots/openpbr_advanced.png) |

---

## Features

### Rendering
- **Vulkan 1.4 dynamic rendering** — `vkCmdBeginRendering` (no render passes; modern efficient rendering)
- **GPU-driven forward rendering** — compute-based culling, indirect command generation, and dispatch
- **Direct lighting** — 1-2 directional lights + Forward+ tile-based point light culling (16×16 px tiles, up to 128 lights/tile)
- **Image-based lighting (IBL)** — equirectangular HDR panorama; diffuse irradiance pre-convolution + per-roughness GGX prefiltered specular map; MaterialX analytic GGX directional albedo (no BRDF LUT)
- **Real-time performance** — **60 FPS @ 4K** target (3840×2160)
- **Interactive camera control** — WASD movement, mouse look, EV100 physical exposure adjustment
- **Screen-Space Reflections (SSR)** — linear view-space ray march (64 steps + 8-step binary refinement) with additive composite blend; roughness cutoff 0.45; IBL as fallback for off-screen misses; adjustable strength via ImGui

### Material model — OpenPBR Surface v1.1
All parameters follow the [OpenPBR spec](https://academysoftwarefoundation.github.io/OpenPBR/) naming. All 8 material layers are fully supported:

| Layer | Parameters | Status |
|-------|-----------|--------|
| Base | `base_weight`, `base_color`, `base_diffuse_roughness`, `base_metalness` | ✅ |
| Specular | `specular_weight`, `specular_color`, `specular_ior`, `specular_roughness`, `specular_roughness_anisotropy` | ✅ |
| Coat | `coat_weight`, `coat_color`, `coat_ior`, `coat_roughness`, `coat_darkening` | ✅ |
| Fuzz | `fuzz_weight`, `fuzz_color`, `fuzz_roughness` | ✅ |
| Emission | `emission_luminance`, `emission_color` | ✅ |
| Thin-film | `thin_film_weight`, `thin_film_thickness`, `thin_film_ior` | ✅ |
| Transmission | `transmission_weight`, `transmission_color`, `transmission_depth` | ✅ |
| Subsurface | `subsurface_weight`, `subsurface_color`, `subsurface_radius`, `subsurface_scale` | ⚠️ (screen-space approximation) |
| Geometry | `geometry_opacity` | ✅ |

Conductor reflectance uses the OpenPBR generalized-Schlick **F82-tint** model (`base_color` = F0, `specular_color` = 82° tint). Specular and coat microfacets use GGX with the spec's anisotropy remapping.

### Color pipeline
- All internal calculations in **linear Rec.2020**
- Physical camera exposure via **EV100** (`ev100` scene keyword)
- Physical environment scale via **`env_unit_nits`** (cd/m² per EXR unit)
- Tone mapping: **AgX** (Troy Sobotka), **ACES** RRT+ODT, **Reinhard** luminance, **Hable** / Uncharted-2 filmic
- Display output: **SDR** (sRGB), **HDR10** (PQ/ST2084), **scRGB** — runtime negotiated with swapchain

**Identical color pipeline to Hyperion** — same algorithms, same visual output (given identical lighting conditions).

### Bindless textures
- Descriptor set 1, binding 4: `COMBINED_IMAGE_SAMPLER` array (up to 1024 entries)
- `NonUniformResourceIndex` for correct divergent access
- Per-material texture maps: `map_base_color`, `map_normal`, `map_orm` (packed occlusion/roughness/metalness), `map_emission_color`

### Scene format
Identical to Hyperion. Line-based text format (`.scene`) with companion `.mtlx` material libraries:

```
mtllib cornell.mtlx         # load material library (.mtlx)

camera
  translate  278  273  -800
  look_at    278  273   279
  vfov       39.1

ev100        7.0            # physical camera exposure
spp          1              # Theia: always 1 SPP (real-time direct frame)

instance cornell.obj        # instantiate geometry (materials assigned here)
  material Floor     WhiteWall
  material LeftWall  RedWall
  material RightWall GreenWall

sphere  60.0
  usemtl Glass
  translate  430  60  200

env_map       meadow_2_4k.exr
env_unit_nits 10000
tonemapper    agx             # aces (default) | agx | reinhard | hable
```

OBJ files contribute **only geometry**; all material assignments are declared in the `.scene` file.

---

## Architecture

Theia is the **real-time** renderer in a family of four repositories:

| Repository | Role |
|------------|------|
| [Aether](https://github.com/McNopper/Aether) | GPU-agnostic file formats & scene data (`.scene` / `.mtlx` / OBJ → plain CPU structs); no Vulkan |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation reused **1:1** by both renderers — core/context, presentation, color management, tonemapping, bindless textures, shared GPU types |
| [Hyperion](https://github.com/McNopper/Hyperion) | Offline path tracer (ground truth) |
| **Theia** | This repo — real-time GPU-driven forward renderer |

Theia consumes Aether and Harmonia via CMake `FetchContent`. The **GPU scene layout is
renderer-specific**: Theia owns its own `Scene`, `GpuInstance` and `GpuMeshlet`
(`src/theia/scene/`) built around meshlets and the mesh-shader pipeline, distinct from
Hyperion's index-buffer / ray-tracing layout. Only code shared 1:1 lives in Harmonia.

---

## Building

**Requirements:** Vulkan SDK 1.4, CMake 3.28+, Ninja, clang-cl, vcpkg.

```bash
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-cl \
      -DCMAKE_CXX_COMPILER=clang-cl \
      -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"

cmake --build build
```

---

## Running

```bash
# Interactive window (default scene: cornell_classic)
build/theia.exe assets/cornell_classic.scene

# Load different scene
build/theia.exe assets/meadow_scene.scene

# Camera controls
# WASD        — move camera
# Mouse       — look around
# E/Q         — adjust exposure (EV100)
# R           — reload scene
# ESC         — quit
```

### Command-line flags

| Flag | Default | Description |
|------|---------|-------------|
| `<scene>` | `assets/cornell_classic.scene` | Path to `.scene` file (first positional argument) |
| `--width <n>` | 3840 | Render width in pixels |
| `--height <n>` | 2160 | Render height in pixels |
| `--no-validation` | — | Disable Vulkan validation layers |
| `--vsync` | enabled | Enable vertical sync |

---

## Tests

Tests cover rendering correctness and material validation:

```bash
cd build && ctest --output-on-failure
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Aether](https://github.com/McNopper/Aether) | Scene & material file formats (`.scene` / `.mtlx` / OBJ) — GPU-agnostic CPU data |
| [Harmonia](https://github.com/McNopper/Harmonia) | Shared Vulkan foundation (core, presentation, color, tonemapping, shared GPU types) |
| [Vulkan SDK](https://vulkan.lunarg.com/) | Modern Vulkan 1.4 API |
| [volk](https://github.com/zeux/volk) | Vulkan loader |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory allocation |
| [SDL3](https://libsdl.org/) | Window management & surface |
| [GLM](https://github.com/g-truc/glm) | Mathematics |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | Meshlet generation and mesh optimization |
| [stb_image](https://github.com/nothings/stb) | PNG/JPEG image loading |
| [tinyexr](https://github.com/syoyo/tinyexr) | EXR image I/O |
| [Slang](https://shader-slang.com/) | Shader compilation (GLSL → SPIR-V) |
| [Google Test](https://github.com/google/googletest) | Unit testing framework |

---

## References

The following specifications, papers, textbooks, and learning resources informed the design of Theia.  
Where a technique is shared with [Hyperion](https://github.com/McNopper/Hyperion), the same reference applies to both.

### Rendering & GPU-Driven Rendering
| Resource | Relevance |
|----------|-----------|
| [Physically Based Rendering: From Theory To Implementation, 4th ed.](https://www.pbrt.org/) (Pharr, Jakob, Humphreys) | BSDF sampling, Monte Carlo integration, MIS balance heuristic, area light PDF conversion |
| [Real-Time Rendering, 4th ed.](https://www.realtimerendering.com/) (Akenine-Möller et al.) | Real-time algorithms, shadows, BRDF models, IBL, anti-aliasing |
| [Wihlidal — "GPU-Driven Rendering Pipelines" (SIGGRAPH 2015)](https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf) | Indirect draw, GPU command generation, per-cluster culling |
| [VK Guide — GPU-Driven Rendering](https://vkguide.dev/docs/gpudriven) | Practical Vulkan indirect dispatch, descriptor binding patterns |
| [Khronos — Mesh Shading for Vulkan](https://www.khronos.org/blog/mesh-shading-for-vulkan) | `VK_EXT_mesh_shader` task/mesh pipeline, `EmitMeshTasksEXT` |
| [Meshoptimizer — Arseny Kapoulkine](https://github.com/zeux/meshoptimizer) | `meshopt_buildMeshlets`, `meshopt_computeMeshletBounds`, cone culling |
| [Heitz, Dupuy, Hill & Neubelt — "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines" (SIGGRAPH 2016)](https://eheitzresearch.wordpress.com/415-2/) | Analytic area-light evaluation reference |
| [McGuire & Mara — "Efficient GPU Screen-Space Ray Tracing" (JCGT 2014)](https://jcgt.org/published/0003/04/04/) | Screen-space reflection depth intersection / ray march |
| [AMD FidelityFX SSSR](https://github.com/GPUOpen-Effects/FidelityFX-SSSR) | SSR reference implementation (MIT) — GGX jitter, denoising, confidence fade |
| [Walter, Marschner, Li & Torrance — "Microfacet Models for Refraction through Rough Surfaces" (EGSR 2007)](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf) | GGX (Trowbridge-Reitz) NDF and Smith G2 — foundation of specular evaluation |
| [Heitz — "Sampling the GGX Distribution of Visible Normals" (JCGT 2018)](https://jcgt.org/published/0007/04/01/) | VNDF importance sampling in the IBL pre-filter compute shader |
| [Ray Tracing Gems I & II](https://www.realtimerendering.com/raytracinggems/) (Haines et al., Marrs et al.) | Hybrid rendering, ray query patterns, shadow ray precision |

### Vulkan & API
| Resource | Relevance |
|----------|-----------|
| [Vulkan Specification 1.4](https://registry.khronos.org/vulkan/specs/latest/html/) | `vkCmdBeginRendering`, mesh shaders, ray queries, descriptor indexing, timeline semaphores |
| [Vulkan Roadmap 2024](https://docs.vulkan.org/spec/latest/appendices/roadmap.html) | Khronos-mandated feature baseline for Theia (no vendor extensions) |
| [Khronos — VK_EXT_mesh_shader Specification](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html) | Task shader payload, `EmitMeshTasksEXT`, output topology |
| [Slang Shading Language](https://shader-slang.com/) | Mesh shader `[outputtopology]`, task payload, `DispatchMesh`, SPIR-V emission |

### Material Model
| Resource | Relevance |
|----------|-----------|
| [OpenPBR Surface Specification v1.1](https://academysoftwarefoundation.github.io/OpenPBR/) | Material layer stack, parameter naming, F82-tint conductor model |
| [MaterialX Standard Surface](https://materialx.org/) | Cross-reference for PBR parameter vocabulary; `mx_ggx_dir_albedo_analytic` for IBL |
| [Blender Principled BSDF](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html) | Cross-reference for PBR parameter vocabulary |

### Color Science
| Resource | Relevance |
|----------|-----------|
| [OpenColorIO](https://opencolorio.org/) | Color space transforms, ACES RRT/ODT, tone mapping nomenclature |
| [AgX by Troy Sobotka](https://github.com/sobotka/AgX) | AgX tone-mapping matrices and S-curve (MIT) |
| [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100/) | PQ/ST2084 and HLG OETF for HDR10 display output |
| [IEC 61966-2-1 (sRGB)](https://www.color.org/srgb.xalter) | sRGB EOTF for SDR display output |

### Scene & Asset Formats
| Resource | Relevance |
|----------|-----------|
| [OpenUSD](https://openusd.org/release/api/index.html) | Naming conventions: Prim, Xform, Mesh, Material, Light, Camera, Instance |
| [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) | PBR material and scene graph conventions |
| [Wavefront OBJ](http://paulbourke.net/dataformats/obj/) | Geometry-only OBJ import (no MTL — materials are assigned in the `.scene` file) |

