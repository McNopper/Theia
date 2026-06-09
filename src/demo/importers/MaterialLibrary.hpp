#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/Texture.hpp"

/// Loads a Hyperion `.mtlx` OpenPBR material library.
///
/// This format is NOT Wavefront MTL and NOT MaterialX XML: it is a line-based
/// text format that reuses the `newmtl` block delimiter but expresses every
/// material property with OpenPBR Surface v1.1 keyword names.  Any classic
/// Wavefront MTL keyword (Kd, Ks, Ni, Tr, Ke, map_Kd, …) is silently ignored.
///
/// All color values are linear Rec.709 by default and are converted to
/// linear Rec.2020 on load.  Declare a different input color space with:
///
///   colorspace lin_rec709    (default — convert to Rec.2020 on load)
///   colorspace lin_rec2020   (already in working color space — no conversion)
///
/// The colorspace keyword is file-level and applies to all materials that
/// follow it.
///
/// OpenPBR Surface v1.1 keywords:
///   base_color              r g b    base_weight                      v
///   base_metalness          v        base_diffuse_roughness           v
///   specular_color          r g b    specular_weight                  v
///   specular_roughness      v        specular_roughness_anisotropy    v
///   specular_ior            v
///   transmission_weight     v        transmission_color               r g b
///   transmission_depth      v        transmission_scatter             r g b
///   transmission_scatter_anisotropy  v   (Henyey-Greenstein g ∈ [-1,1])
///   transmission_dispersion_scale        v
///   transmission_dispersion_abbe_number  v   (Abbe Vd; lower = more dispersion)
///   thin_film_weight        v        thin_film_thickness              v (nm)
///   thin_film_ior           v
///   coat_weight             v        coat_color                       r g b
///   coat_roughness          v        coat_roughness_anisotropy        v
///   coat_ior                v        coat_darkening                   v
///   fuzz_weight             v        fuzz_color                       r g b
///   fuzz_roughness          v
///   emission_color          r g b    emission_luminance               v
///   subsurface_weight       v        subsurface_color                 r g b
///   subsurface_radius       r g b    subsurface_scale                 v
///   subsurface_scatter_anisotropy    v   (Henyey-Greenstein g ∈ [-1,1])
///   geometry_opacity        v        (alias: opacity)
///   geometry_thin_walled    v        (alias: thin_walled; 0/1 double-sided sheet)
///
/// Texture map keywords (path is relative to the .mtlx file directory).
/// Maps tagged [slot N] are uploaded to bindless texture slot N and sampled by
/// the path tracer; map_roughness / map_metalness are parsed but reserved
/// (use the packed map_orm instead — G = roughness, B = metalness):
///   map_base_color     [0] path     map_base_color_colorspace        name
///   map_normal         [1] path     map_normal_colorspace            name
///   map_orm            [2] path     map_orm_colorspace               name
///   map_emission_color [3] path     map_emission_color_colorspace    name
///   map_coat_normal    [4] path     (alias: geometry_coat_normal)
///   map_tangent        [5] path     (alias: geometry_tangent; anisotropy frame)
///   map_coat_tangent   [6] path     (alias: geometry_coat_tangent)
///   map_roughness          path     map_roughness_colorspace         name
///   map_metalness          path     map_metalness_colorspace         name
///
/// Color space names follow the OCIO / OpenEXR IIF registry.
/// Textures are converted to linear Rec.2020 at load time.
/// Defaults: color maps = srgb_texture; data maps (normal/ORM/roughness) = raw.
/// Supported names:
///   srgb_texture    sRGB OETF + Rec.709 primaries (typical PNG/JPEG albedo)
///   lin_srgb        linear Rec.709 primaries
///   lin_rec2020     linear Rec.2020 (render color space; no conversion)
///   acescg          ACEScg / lin_ap1
///   raw             uninterpreted data (normal, ORM, roughness, metalness)
///
/// UsdPreviewSurface aliases (mapped to OpenPBR names):
///   diffuseColor → base_color           metallic → base_metalness
///   roughness → specular_roughness      ior → specular_ior
///   emissiveColor → emission_color      emissiveLuminance → emission_luminance
///   clearcoat → coat_weight             clearcoatRoughness → coat_roughness
///   transmissionAmount → transmission_weight
///   geometry_opacity → opacity          (OpenPBR canonical opacity name)
class MaterialLibrary {
  public:
    /// Reference to one texture map: file path + source color space.
    struct MaterialTextureRef {
        std::string path;
        TextureColorSpace colorSpace = TextureColorSpace::SrgbTexture;
        [[nodiscard]] bool empty() const noexcept { return path.empty(); }
    };

    /// All texture references for one material (one entry per bindless map slot).
    /// Slot order matches GpuMaterial::textureIndices: [0] base_color, [1] normal,
    /// [2] ORM (occlusion/roughness/metalness), [3] emission.
    struct MaterialTextureRefs {
        MaterialTextureRef base_color;
        MaterialTextureRef normal;
        MaterialTextureRef orm;
        MaterialTextureRef emission;
        MaterialTextureRef coat_normal;
        MaterialTextureRef tangent;
        MaterialTextureRef coat_tangent;
    };

    /// Load material definitions from a .mtlx file.
    /// Returns false only if the file cannot be opened.
    bool load(const std::filesystem::path& path);

    /// Look up a material by name.  Returns std::nullopt if not found.
    [[nodiscard]] std::optional<Material> get(const std::string& name) const;

    /// Look up a material by name, or return a default diffuse-gray material.
    [[nodiscard]] Material getOrDefault(const std::string& name) const;

    /// Return the texture references recorded for the named material,
    /// or std::nullopt if the material is not found.
    [[nodiscard]] std::optional<MaterialTextureRefs> textureRefs(const std::string& name) const;

    /// Patch the texture index for slot in the stored material.
    /// Call this after the texture has been uploaded to the GPU so that
    /// subsequent getOrDefault() calls return the correct bindless index.
    void patchTextureIndex(const std::string& name, uint32_t slot, uint32_t idx);

    [[nodiscard]] bool empty() const noexcept { return m_materials.empty(); }
    [[nodiscard]] size_t size() const noexcept { return m_materials.size(); }

  private:
    std::unordered_map<std::string, Material> m_materials;
    std::unordered_map<std::string, MaterialTextureRefs> m_textureRefs;
};
