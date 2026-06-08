#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/importers/MaterialLibrary.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "hyperion/core/Logger.hpp"
#include "hyperion/scene/Texture.hpp"
#include "hyperion/utils/ColorSpace.hpp"

namespace {

/// Texture path + its source color space (OCIO/OpenEXR naming).
struct TextureRef {
    std::string path;
    TextureColorSpace colorSpace;
};

/// All internal state for one material block being parsed.
struct MatParams {
    // base
    glm::vec3 base_color{0.8f, 0.8f, 0.8f};
    float base_weight = 1.0f;
    float base_metalness = 0.0f;
    float base_diffuse_roughness = 0.0f; ///< Oren-Nayar σ for the diffuse lobe
    // specular  (defaults per OpenPBR Surface spec)
    glm::vec3 specular_color{1.0f, 1.0f, 1.0f};
    float specular_weight = 1.0f;
    float specular_roughness = 0.3f;
    float specular_roughness_anisotropy = 0.0f; ///< 0 = isotropic, 1 = fully anisotropic
    float specular_ior = 1.5f;
    // transmission
    float transmission_weight = 0.0f;
    glm::vec3 transmission_color{1.0f, 1.0f, 1.0f};
    float transmission_depth = 0.0f; ///< Beer-law depth (world units); 0 = no absorption
    glm::vec3 transmission_scatter{0.0f, 0.0f, 0.0f};
    float transmission_scatter_anisotropy = 0.0f;      ///< HG mean cosine g ∈ [-1,1]
    float transmission_dispersion_scale = 0.0f;        ///< chromatic dispersion strength; 0 = off
    float transmission_dispersion_abbe_number = 20.0f; ///< Abbe number Vd (lower = more dispersion)
    // coat  (defaults per OpenPBR Surface spec)
    float coat_weight = 0.0f;
    glm::vec3 coat_color{1.0f, 1.0f, 1.0f};
    float coat_roughness = 0.3f;
    float coat_roughness_anisotropy = 0.0f;
    float coat_ior = 1.5f;
    float coat_darkening = 1.0f; ///< energy darkening at base/coat interface
    // fuzz / sheen
    float fuzz_weight = 0.0f;
    glm::vec3 fuzz_color{1.0f, 1.0f, 1.0f};
    float fuzz_roughness = 0.5f;
    // thin film
    float thin_film_weight = 0.0f;
    float thin_film_thickness = 0.0f; ///< nm; 0–2000 typical range
    float thin_film_ior = 1.5f;
    // emission
    glm::vec3 emission_color{1.0f, 1.0f, 1.0f};
    float emission_luminance = 0.0f;
    // subsurface  (defaults per OpenPBR Surface spec; radius scale = Rayleigh-like 1/0.5/0.25)
    float subsurface_weight = 0.0f;
    glm::vec3 subsurface_color{0.8f, 0.8f, 0.8f};
    glm::vec3 subsurface_radius{1.0f, 0.5f, 0.25f};
    float subsurface_scale = 1.0f;
    float subsurface_scatter_anisotropy = 0.0f; ///< HG mean cosine g ∈ [-1,1]
    // opacity
    float opacity = 1.0f;
    bool thin_walled = false; ///< geometry_thin_walled: double-sided thin sheet
    // texture maps — path + source color space (converted to linear Rec.2020 at load)
    TextureRef map_base_color{"", TextureColorSpace::SrgbTexture};
    TextureRef map_normal{"", TextureColorSpace::Raw};
    TextureRef map_orm{"", TextureColorSpace::Raw};
    TextureRef map_roughness{"", TextureColorSpace::Raw};
    TextureRef map_metalness{"", TextureColorSpace::Raw};
    TextureRef map_emission_color{"", TextureColorSpace::SrgbTexture};
    TextureRef map_coat_normal{"", TextureColorSpace::Raw};
    TextureRef map_tangent{"", TextureColorSpace::Raw};
    TextureRef map_coat_tangent{"", TextureColorSpace::Raw};
};

// ── Helpers ───────────────────────────────────────────────────────────────

[[nodiscard]] std::string_view trimSv(std::string_view sv) noexcept {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    const auto last = sv.find_last_not_of(" \t\r\n");
    return sv.substr(first, last - first + 1);
}

[[nodiscard]] std::string_view stripComment(std::string_view line) noexcept {
    const auto pos = line.find('#');
    return trimSv(pos == std::string_view::npos ? line : line.substr(0, pos));
}

/// Map UsdPreviewSurface / MaterialX aliases → OpenPBR keyword names.
/// Classic OBJ/MTL keywords (Kd, Ks, Ni, Tr, Ke, map_Kd, map_Ns, …) are
/// intentionally NOT listed here — they must be silently ignored.
[[nodiscard]] std::string_view normalise(std::string_view kw) noexcept {
    if (kw == "diffuseColor")
        return "base_color";
    if (kw == "metallic")
        return "base_metalness";
    if (kw == "roughness")
        return "specular_roughness";
    if (kw == "base_roughness")
        return "base_diffuse_roughness";
    if (kw == "ior")
        return "specular_ior";
    if (kw == "emissiveColor")
        return "emission_color";
    if (kw == "emissiveLuminance")
        return "emission_luminance";
    if (kw == "clearcoat")
        return "coat_weight";
    if (kw == "clearcoatRoughness")
        return "coat_roughness";
    if (kw == "transmissionAmount")
        return "transmission_weight";
    if (kw == "specularColor")
        return "specular_color";
    // OpenPBR canonical geometry opacity name (Hyperion stores it as `opacity`).
    if (kw == "geometry_opacity")
        return "opacity";
    if (kw == "geometry_thin_walled")
        return "thin_walled";
    // OpenPBR canonical subsurface radius-scale name (Hyperion stores it as `subsurface_scale`).
    if (kw == "subsurface_radius_scale")
        return "subsurface_scale";
    // OpenPBR canonical geometry normal/tangent map names.
    if (kw == "geometry_normal")
        return "map_normal";
    if (kw == "geometry_coat_normal")
        return "map_coat_normal";
    if (kw == "geometry_tangent")
        return "map_tangent";
    if (kw == "geometry_coat_tangent")
        return "map_coat_tangent";
    return kw;
}

[[nodiscard]] bool parseVec3(std::string_view text, glm::vec3& out) {
    std::istringstream ss{std::string(text)};
    return static_cast<bool>(ss >> out.x >> out.y >> out.z);
}

[[nodiscard]] bool parseFloat(std::string_view text, float& out) {
    std::istringstream ss{std::string(text)};
    return static_cast<bool>(ss >> out);
}

// ── Parameter dispatch ────────────────────────────────────────────────────

void applyKw(MatParams& p, std::string_view rawKw, std::string_view rest, bool convertToRec2020) {
    const std::string_view kw = normalise(rawKw);

    // ── Texture map paths (string values) ────────────────────────────────
    if (kw == "map_base_color") {
        p.map_base_color.path = std::string(rest);
        return;
    }
    if (kw == "map_normal") {
        p.map_normal.path = std::string(rest);
        return;
    }
    if (kw == "map_orm") {
        p.map_orm.path = std::string(rest);
        return;
    }
    if (kw == "map_roughness") {
        p.map_roughness.path = std::string(rest);
        return;
    }
    if (kw == "map_metalness") {
        p.map_metalness.path = std::string(rest);
        return;
    }
    if (kw == "map_emission_color") {
        p.map_emission_color.path = std::string(rest);
        return;
    }
    if (kw == "map_coat_normal") {
        p.map_coat_normal.path = std::string(rest);
        return;
    }
    if (kw == "map_tangent") {
        p.map_tangent.path = std::string(rest);
        return;
    }
    if (kw == "map_coat_tangent") {
        p.map_coat_tangent.path = std::string(rest);
        return;
    }

    // ── Texture map color spaces (OCIO / OpenEXR IIF names) ──────────────
    if (kw == "map_base_color_colorspace") {
        p.map_base_color.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_normal_colorspace") {
        p.map_normal.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_orm_colorspace") {
        p.map_orm.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_roughness_colorspace") {
        p.map_roughness.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_metalness_colorspace") {
        p.map_metalness.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_emission_color_colorspace") {
        p.map_emission_color.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_coat_normal_colorspace") {
        p.map_coat_normal.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_tangent_colorspace") {
        p.map_tangent.colorSpace = parseTextureColorSpace(rest);
        return;
    }
    if (kw == "map_coat_tangent_colorspace") {
        p.map_coat_tangent.colorSpace = parseTextureColorSpace(rest);
        return;
    }

    // ── Colour keywords ───────────────────────────────────────────────────
    glm::vec3 c{};
    if (kw == "base_color" || kw == "specular_color" || kw == "transmission_color" || kw == "coat_color" ||
        kw == "fuzz_color" || kw == "emission_color" || kw == "subsurface_color" || kw == "subsurface_radius" ||
        kw == "transmission_scatter") {
        if (!parseVec3(rest, c))
            return;
        // Non-colour data: skip colour-space conversion.
        const bool isColour = (kw != "subsurface_radius" && kw != "transmission_scatter");
        if (isColour && convertToRec2020)
            c = ColorSpace::rec709ToRec2020(c);
        if (kw == "base_color")
            p.base_color = c;
        else if (kw == "specular_color")
            p.specular_color = c;
        else if (kw == "transmission_color")
            p.transmission_color = c;
        else if (kw == "transmission_scatter")
            p.transmission_scatter = c;
        else if (kw == "coat_color")
            p.coat_color = c;
        else if (kw == "fuzz_color")
            p.fuzz_color = c;
        else if (kw == "emission_color")
            p.emission_color = c;
        else if (kw == "subsurface_color")
            p.subsurface_color = c;
        else if (kw == "subsurface_radius")
            p.subsurface_radius = c;
        return;
    }

    // ── Scalar keywords ───────────────────────────────────────────────────
    float f = 0.0f;
    if (!parseFloat(rest, f))
        return;
    if (kw == "base_weight")
        p.base_weight = f;
    else if (kw == "base_metalness")
        p.base_metalness = f;
    else if (kw == "base_diffuse_roughness")
        p.base_diffuse_roughness = f;
    else if (kw == "specular_weight")
        p.specular_weight = f;
    else if (kw == "specular_roughness")
        p.specular_roughness = f;
    else if (kw == "specular_roughness_anisotropy")
        p.specular_roughness_anisotropy = f;
    else if (kw == "specular_ior")
        p.specular_ior = f;
    else if (kw == "transmission_weight")
        p.transmission_weight = f;
    else if (kw == "transmission_depth")
        p.transmission_depth = f;
    else if (kw == "transmission_scatter_anisotropy")
        p.transmission_scatter_anisotropy = std::clamp(f, -1.0f, 1.0f);
    else if (kw == "transmission_dispersion_scale")
        p.transmission_dispersion_scale = std::max(f, 0.0f);
    else if (kw == "transmission_dispersion_abbe_number")
        p.transmission_dispersion_abbe_number = f;
    else if (kw == "thin_film_weight")
        p.thin_film_weight = f;
    else if (kw == "thin_film_thickness")
        p.thin_film_thickness = f;
    else if (kw == "thin_film_ior")
        p.thin_film_ior = f;
    else if (kw == "coat_weight")
        p.coat_weight = f;
    else if (kw == "coat_roughness")
        p.coat_roughness = f;
    else if (kw == "coat_roughness_anisotropy")
        p.coat_roughness_anisotropy = f;
    else if (kw == "coat_ior")
        p.coat_ior = f;
    else if (kw == "coat_darkening")
        p.coat_darkening = f;
    else if (kw == "fuzz_weight")
        p.fuzz_weight = f;
    else if (kw == "fuzz_roughness")
        p.fuzz_roughness = f;
    else if (kw == "emission_luminance")
        p.emission_luminance = f;
    else if (kw == "subsurface_weight")
        p.subsurface_weight = f;
    else if (kw == "subsurface_scale")
        p.subsurface_scale = f;
    else if (kw == "subsurface_scatter_anisotropy")
        p.subsurface_scatter_anisotropy = std::clamp(f, -1.0f, 1.0f);
    else if (kw == "opacity")
        p.opacity = f;
    else if (kw == "thin_walled")
        p.thin_walled = (f != 0.0f);
}

// ── Build GpuMaterial from parsed OpenPBR params ─────────────────────────
// All lobes are set directly; no threshold-based type selection.
// flags: 0 = general layered, 2 = glass/dielectric, 3 = mirror (not set here).

[[nodiscard]] Material buildMaterial(const MatParams& p) {
    GpuMaterial g{};

    // Base layer
    g.baseColorWeight = glm::vec4(p.base_color, p.base_weight);
    g.baseMetalnessDiffRough = glm::vec4(p.base_metalness, p.base_diffuse_roughness, 0.0f, 0.0f);

    // Specular
    g.specularColorWeight = glm::vec4(p.specular_color, p.specular_weight);
    g.specularRoughAnisoIor =
        glm::vec4(p.specular_roughness, p.specular_roughness_anisotropy, std::max(p.specular_ior, 1.0f), 0.0f);

    // Transmission
    g.transmissionColorWeight = glm::vec4(p.transmission_color, p.transmission_weight);
    g.transmissionParams = glm::vec4(p.transmission_depth,
                                     p.specular_roughness,
                                     p.transmission_dispersion_scale,
                                     std::max(p.transmission_dispersion_abbe_number, 1.0f));
    g.transmissionScatter = glm::vec4(p.transmission_scatter, p.transmission_scatter_anisotropy);

    // Subsurface
    g.subsurfaceColorWeight = glm::vec4(p.subsurface_color, p.subsurface_weight);
    g.subsurfaceRadiusScale = glm::vec4(p.subsurface_radius, p.subsurface_scale);

    // Texture indices (filled by uploadTextures later; sentinel = kNoTexture)
    g.textureIndices = glm::uvec4(kNoTexture);
    g.textureIndices2 = glm::uvec4(kNoTexture);

    // Thin film
    g.thinFilmParams = glm::vec4(p.thin_film_thickness, std::max(p.thin_film_ior, 1.0f), p.thin_film_weight, 0.0f);

    // Coat
    g.coatColorWeight = glm::vec4(p.coat_color, p.coat_weight);
    g.coatRoughAnisoIorDark = glm::vec4(p.coat_roughness,
                                        p.coat_roughness_anisotropy,
                                        std::max(p.coat_ior, 1.0f),
                                        std::clamp(p.coat_darkening, 0.0f, 1.0f));

    // Fuzz / sheen
    g.fuzzColorWeight = glm::vec4(p.fuzz_color, p.fuzz_weight);
    g.fuzzRoughPad = glm::vec4(std::clamp(p.fuzz_roughness, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f);

    // Emission
    g.emissionColorLum = glm::vec4(p.emission_color, std::max(p.emission_luminance, 0.0f));

    // Opacity + flags: glass mode enables Fresnel split in sampleBSDF.
    const float flags = (p.transmission_weight >= 0.5f && p.base_metalness < 0.5f) ? 2.0f : 0.0f;
    g.opacityFlagsPad = glm::vec4(
        std::clamp(p.opacity, 0.0f, 1.0f), flags, p.subsurface_scatter_anisotropy, p.thin_walled ? 1.0f : 0.0f);

    return Material::fromGpu(g);
}

} // namespace

// ── MaterialLibrary ───────────────────────────────────────────────────────

bool MaterialLibrary::load(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        Logger::error("MaterialLibrary: cannot open '{}'", path.string());
        return false;
    }

    std::string currentName;
    MatParams currentParams;
    bool hasCurrent = false;
    bool convertToRec2020 = true; // default: lin_rec709 input → convert to lin_rec2020

    auto flush = [&] {
        if (hasCurrent && !currentName.empty()) {
            m_materials.insert_or_assign(currentName, buildMaterial(currentParams));
            // Store texture references so SceneLoader can pre-load them (one per slot).
            MaterialTextureRefs refs;
            refs.base_color = {currentParams.map_base_color.path, currentParams.map_base_color.colorSpace};
            refs.normal = {currentParams.map_normal.path, currentParams.map_normal.colorSpace};
            refs.orm = {currentParams.map_orm.path, currentParams.map_orm.colorSpace};
            refs.emission = {currentParams.map_emission_color.path, currentParams.map_emission_color.colorSpace};
            refs.coat_normal = {currentParams.map_coat_normal.path, currentParams.map_coat_normal.colorSpace};
            refs.tangent = {currentParams.map_tangent.path, currentParams.map_tangent.colorSpace};
            refs.coat_tangent = {currentParams.map_coat_tangent.path, currentParams.map_coat_tangent.colorSpace};
            m_textureRefs.insert_or_assign(currentName, std::move(refs));
        }
    };

    std::string line;
    while (std::getline(file, line)) {
        const std::string_view trimmed = stripComment(line);
        if (trimmed.empty())
            continue;

        std::istringstream ss{std::string(trimmed)};
        std::string kw;
        ss >> kw;

        if (kw == "colorspace") {
            // File-level color space declaration.
            std::string cs;
            std::getline(ss, cs);
            const std::string_view csv = trimSv(cs);
            convertToRec2020 = (csv != "lin_rec2020" && csv != "rec2020");
        } else if (kw == "newmtl") {
            flush();
            currentName = {};
            currentParams = {};
            hasCurrent = true;
            std::getline(ss, currentName);
            currentName = std::string(trimSv(currentName));
        } else if (hasCurrent) {
            // Rest of line after the keyword.
            std::string rest;
            std::getline(ss, rest);
            applyKw(currentParams, kw, trimSv(rest), convertToRec2020);
        }
    }
    flush();

    Logger::info("MaterialLibrary: loaded {} material(s) from '{}'", m_materials.size(), path.filename().string());
    return true;
}

std::optional<Material> MaterialLibrary::get(const std::string& name) const {
    const auto it = m_materials.find(name);
    return it != m_materials.end() ? std::optional{it->second} : std::nullopt;
}

Material MaterialLibrary::getOrDefault(const std::string& name) const {
    return get(name).value_or(Material::diffuse(glm::vec3(0.8f)));
}

std::optional<MaterialLibrary::MaterialTextureRefs> MaterialLibrary::textureRefs(const std::string& name) const {
    const auto it = m_textureRefs.find(name);
    return it != m_textureRefs.end() ? std::optional{it->second} : std::nullopt;
}

void MaterialLibrary::patchTextureIndex(const std::string& name, uint32_t slot, uint32_t idx) {
    const auto it = m_materials.find(name);
    if (it != m_materials.end())
        it->second.setTextureIndex(slot, idx);
}
