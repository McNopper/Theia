#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>

#include <algorithm>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "theia/renderer/IblPrecompute.hpp"

class Scene;

namespace theia {

/// GPU-driven forward renderer using Vulkan 1.4 features.
/// - Dynamic rendering (no render passes)
/// - Mesh shaders (VK_EXT_mesh_shader) for GPU-driven geometry
/// - Depth pre-pass ready (depth target owned here)
///
/// The HDR color target is owned by Application (same as Hyperion''s hdrImage)
/// and passed in via Config so ToneMapper can read it as a storage image.
class ForwardRenderer {
  public:
    struct Config {
        uint32_t width = 1024;
        uint32_t height = 768;
        VkFormat outputFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkImage hdrImage = VK_NULL_HANDLE; ///< externally owned HDR color target
        VkImageView hdrImageView = VK_NULL_HANDLE;
        VkImageView iblDiffuseView = VK_NULL_HANDLE;
        VkImageView iblSpecularView = VK_NULL_HANDLE;
        VkImageView sheenLutView = VK_NULL_HANDLE;
        VkSampler iblEnvSampler = VK_NULL_HANDLE;
        VkSampler iblLutSampler = VK_NULL_HANDLE;
    };

    /// Camera parameters. Uses the shared harmonia::Camera::Params so both
    /// Hyperion and Theia describe their cameras with the same type.
    /// ForwardRenderer reads: position, target, up, vfovDeg, nearPlane,
    /// farPlane, and physical.ev100() for exposure.
    using CameraParams = Camera::Params;

    ForwardRenderer() = default;
    ~ForwardRenderer();

    bool initialize(const DeviceContext& ctx, const Config& config);
    void shutdown();

    /// Resize to a new extent after the host (App) has recreated the HDR image.
    /// Recreates depth/GBuffer targets; the pipeline and descriptors are unchanged.
    bool resize(uint32_t width, uint32_t height, VkImage hdrImage, VkImageView hdrImageView) noexcept;

    void setScene(const Scene* scene) {
        m_scene = scene;
        m_texturesBoundFor = nullptr;
    }
    void setCamera(const CameraParams& cam) { m_camera = cam; }
    /// Bind IBL textures (set 2 / sky set 0). When a raw environment panorama is
    /// available it is bound to IBL binding 4 for the sky background and its
    /// physical scale (env_unit_nits) recorded; pass VK_NULL_HANDLE / 1.0f when the
    /// scene has no env map (binding 4 falls back to the specular view as a
    /// harmless placeholder).
    void setIbl(const IblResources& res, VkImageView rawEnvView = VK_NULL_HANDLE, float envUnitNits = 1.0f);

    /// Whether the current scene has an environment map (draws env sky vs black).
    void setHasEnvironment(bool hasEnv) noexcept { m_hasEnv = hasEnv; }

    /// Configure ray-traced directional ("sun") shadows from the dominant IBL light.
    /// dirToSun is the world-space direction pointing toward the sun; strength in [0,1]
    /// scales how dark the cast shadow is (0 disables sun shadows entirely).
    void setSunShadow(const glm::vec3& dirToSun, float strength) noexcept {
        m_sunDir = dirToSun;
        m_sunStrength = strength;
    }
    /// Presentation-only constant indirect ambient term (default off for parity).
    void setIndirectAmbient(float strength) noexcept { m_indirectAmbientStrength = std::max(0.0f, strength); }

    /// Update tile light list buffers (called by LightCuller each frame before recordFrame).
    void setTileBuffers(VkBuffer tileLightCounts, VkBuffer tileLightIndices, uint32_t tilesX, uint32_t tilesY);

    /// Record scene geometry rendering into cmd.
    /// Transitions hdrImage UNDEFINED/GENERAL -> ATTACHMENT_OPTIMAL, renders, leaves it there.
    /// Application::mainLoop() will barrier it to GENERAL before the ToneMapper.
    void recordFrame(VkCommandBuffer cmd);

    // Thin GBuffer (RGBA16F: xyz=view-space normal [0,1], w=roughness) for SSR.
    [[nodiscard]] VkImageView gbufferView() const noexcept { return m_gbufferTarget.view(); }
    [[nodiscard]] VkImage gbufferImage() const noexcept { return m_gbufferTarget.handle(); }
    [[nodiscard]] VkImageView depthView() const noexcept { return m_depthTarget.view(); }
    [[nodiscard]] VkImage depthImage() const noexcept { return m_depthTarget.handle(); }

  private:
    struct MeshPushConstants {
        glm::mat4 viewProj; ///< transposed for Slang mul(pos, mat) convention
        glm::mat4 view;
        glm::vec4 cameraPos;               ///< xyz = world-space camera position
        float     exposure             = 0.0f; ///< 1 / (1.2 * 2^EV100) — same as Hyperion
        uint32_t  lightCount           = 0;
        uint32_t  emissiveTriangleCount = 0;
        uint32_t  tilesX               = 0; ///< screen width / 16
        uint32_t  tilesY               = 0; ///< screen height / 16
        uint32_t  screenWidth          = 0;
        uint32_t  screenHeight         = 0;
        uint32_t  _pad                 = 0;
        glm::vec4 sunDirection; ///< xyz = world dir toward sun, w = shadow strength (0 disables)
        glm::vec4 shadowParams; ///< x = ray tMin (scene-scale bias), y = sky ambient floor
        glm::vec4 presentationParams; ///< x = indirect ambient strength (scene-linear), yzw reserved
    };
    static_assert(sizeof(MeshPushConstants) == 224);

    bool createDepthTarget();
    bool createPipeline();

    Config m_config{};
    CameraParams m_camera{};
    const DeviceContext* m_ctx = nullptr;
    const Scene* m_scene = nullptr;

    // Depth-only target (color target is externally owned)
    Image m_depthTarget;
    // Thin GBuffer target (RGBA16F: view-space normal + roughness)
    Image m_gbufferTarget;

    // Mesh + fragment graphics pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline m_graphicsPipelineTransparent = VK_NULL_HANDLE;

    // Sky/background fullscreen pipeline (samples env panorama, or black).
    struct SkyPushConstants {
        glm::mat4 invViewProj;
        glm::vec4 cameraPos;
        float    exposure  = 0.0f;
        uint32_t hasEnv    = 0;
        float    envScale  = 0.0f; ///< env_unit_nits — physical cd/m² per raw EXR unit
        uint32_t _pad1     = 0;
    };
    static_assert(sizeof(SkyPushConstants) == 96);
    VkPipelineLayout m_skyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_skyPipeline = VK_NULL_HANDLE;
    bool m_hasEnv = false;
    glm::vec3 m_sunDir{0.0f, 1.0f, 0.0f}; ///< world dir toward dominant IBL light
    float m_sunStrength = 0.0f;           ///< [0,1] ray-traced sun shadow strength
    float m_indirectAmbientStrength = 0.0f;

    // Set 0: geometry buffers (vertex/instance/index/meshlet data — task + mesh stages)
    VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_meshSet = VK_NULL_HANDLE;

    // Set 1: material/lighting buffers (materials/lights/emissive — fragment stage)
    VkDescriptorSetLayout m_matSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_matSet = VK_NULL_HANDLE;

    // Set 2: IBL textures + samplers (fragment stage)
    VkDescriptorSetLayout m_iblSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_iblSet = VK_NULL_HANDLE;

    // Set 3: bindless material textures (base_color/normal/ORM/emission — fragment stage)
    static constexpr uint32_t kMaxBindlessTextures = 256;
    VkDescriptorSetLayout m_textureSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_textureSet = VK_NULL_HANDLE;
    const Scene* m_texturesBoundFor = nullptr; ///< scene the bindless set was last written for

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    VkDescriptorImageInfo m_iblDiffuseInfo{};
    VkDescriptorImageInfo m_iblSpecularInfo{};
    VkDescriptorImageInfo m_sheenLutInfo{};
    VkDescriptorImageInfo m_brdfLutInfo{};
    VkDescriptorImageInfo m_iblEnvSamplerInfo{};
    VkDescriptorImageInfo m_iblEnvRawInfo{};
    float m_envUnitNits = 1.0f; ///< env_unit_nits for the raw-env sky background

    // Tile-based light culling buffers (set by LightCuller before each recordFrame).
    VkBuffer m_tileLightCountsBuf = VK_NULL_HANDLE;
    VkBuffer m_tileLightIndicesBuf = VK_NULL_HANDLE;
    uint32_t m_tilesX = 0;
    uint32_t m_tilesY = 0;
    // Dummy 1-element buffers bound when no tile data is available (fallback to full light loop).
    Buffer m_dummyTileCounts;
    Buffer m_dummyTileIndices;

    bool m_initialized = false;
    bool m_hdrFirstUse = true; ///< tracks whether HDR image is still in UNDEFINED layout
};

} // namespace theia
