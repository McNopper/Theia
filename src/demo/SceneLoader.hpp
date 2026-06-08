#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <optional>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/scene/Scene.hpp"

/// Loads a Hyperion scene definition file (.scene).
///
/// FORMAT OVERVIEW
/// ───────────────
/// Line-based text format inspired by — but distinct from — Wavefront OBJ/MTL.
/// Lines starting with '#' are comments.  All asset paths are resolved
/// relative to the assets directory passed to load().
///
/// ── Global keywords ──────────────────────────────────────────────────────────
///
///   mtllib        <path>        Load a Hyperion .mtlx OpenPBR material library.
///   spp           <n>           Samples per pixel.
///   max_depth     <n>           Maximum ray bounce depth.
///   env_map       <path>        Equirectangular HDR panorama (.exr) for IBL.
///                               The EXR stores unitless scene-linear values in
///                               Rec.709 primaries; they are converted to linear
///                               Rec.2020 on load.  Use env_unit_nits to assign
///                               physical units.
///   env_unit_nits <v>           Physical unit of one EXR sample in cd/m²
///                               (default 1.0).  Set to the luminance in nits
///                               that corresponds to a pixel value of 1.0 in the
///                               EXR (e.g. 10000 for a typical outdoor HDRI).
///                               Combined with the camera EV100, this gives a
///                               correctly-exposed physically-based render.
///   ev100         <v>           Physical camera EV100 override (aperture=f/1,
///                               iso=100, shutter=2^ev100 s⁻¹).  Typical outdoor
///                               values: 13–15.  Lower values → brighter image.
///   tonemapper    <name>        Tone mapper for SDR and Display P3 output
///                               (ignored for HDR10 / HLG / scRGB which use their
///                               own transfer functions).  Valid values:
///                                 aces     — ACES RRT+ODT filmic (default)
///                                 agx      — AgX by Troy Sobotka; wide dynamic
///                                            range, natural highlight rolloff,
///                                            handles direct sun in IBL scenes
///                                 reinhard — luminance-preserving Reinhard
///                                 hable    — Hable / Uncharted-2 filmic
///
/// ── Block keywords  (each starts a new block) ────────────────────────────────
///
///   camera                     Camera block.  Must appear before geometry.
///   instance <path>            Instantiate a Wavefront OBJ as a scene object
///                              (geometry only — OBJ materials are not imported).
///   sphere <r>                 Analytic sphere — radius only; position
///                              via the translate modifier.
///   box    <hx> <hy> <hz>      Procedural box — half-extents only;
///                              position/orientation via modifiers.
///
/// ── Block modifiers  (apply to the block that precedes them) ─────────────────
///
///   Shared by all blocks:
///     translate <x> <y> <z>      World-space position / translation.
///     rotate  <qx> <qy> <qz> <qw>  Orientation as a unit quaternion
///                                   (glTF convention: x y z w).
///     rotate_y <deg>             Convenience: rotation around the Y axis.
///
///   Geometry only:
///     usemtl  <name>             Override material (resolved from mtllibs).
///     scale   <sx> <sy> <sz>     Per-axis scale (one value = uniform).
///
///   Camera only:
///     look_at <x> <y> <z>        Look-at target (alternative to rotate;
///                                last-one-wins if both are given).
///     up      <x> <y> <z>        Up vector used with look_at (default 0 1 0).
///     vfov    <deg>              Vertical field of view in degrees.
///
/// Transform composition follows the glTF convention:  T × R × S.
///
/// NOTES
/// ─────
///   - Block modifiers are collected until the next block keyword or EOF.
///   - 'sphere': translate sets the centre; rotate is ignored (symmetric);
///     scale.x is used as a uniform radius multiplier.
///   - 'box' and 'instance': full TRS is applied.
///   - Camera rotate and look_at are mutually exclusive; last one wins.
///   - Unrecognised keywords are silently ignored, matching OBJ/MTL behaviour.
class SceneLoader {
  public:
    struct SceneConfig {
        std::optional<glm::vec3> cameraPos;
        std::optional<glm::vec3> cameraAt;
        std::optional<glm::vec3> cameraUp;
        std::optional<float> cameraVfov;
        std::optional<float> cameraEv100; ///< physical camera EV100 override
        std::optional<uint32_t> spp;
        std::optional<uint32_t> maxDepth;
        std::optional<float> envUnitNits;                ///< cd/m² per unit EXR value (physical unit multiplier)
        std::optional<std::filesystem::path> envMapFile; ///< equirect EXR IBL path (relative to assetsDir)
        std::optional<uint32_t> tonemapper;              ///< Tonemapper enum value; std::nullopt → eACES default
    };

    /// Populate @p scene from @p sceneFile.
    /// Asset paths are resolved relative to @p assetsDir.
    /// Returns a SceneConfig with camera / render overrides on success,
    /// or std::nullopt if the file cannot be opened.
    [[nodiscard]] std::optional<SceneConfig> load(const std::filesystem::path& sceneFile,
                                                  const std::filesystem::path& assetsDir,
                                                  Scene& scene,
                                                  const DeviceContext& ctx,
                                                  const CommandPool& pool);
};
