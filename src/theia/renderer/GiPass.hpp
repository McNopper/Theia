#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>

#include "harmonia/DeviceContext.hpp"

class Scene;

namespace theia {

/// Ray-query global-illumination compute stage.
///
/// Runs one compute dispatch after the forward geometry pass. For every primary
/// hit recorded in the forward G-buffer it spawns a BSDF continuation ray and walks
/// 1..maxDepth path segments through the scene TLAS (inline VK_KHR_ray_query),
/// evaluating the shared Harmonia `integrateSurface` estimator at each secondary
/// vertex. The resulting indirect (bounce) radiance is additively composited into
/// the HDR render target, replacing the forward pass's flat-IBL approximation.
///
/// Toggleable and distinct from the forward raster stage: when active, the forward
/// pass emits direct + emission only (giEnabled push-constant disables IBL/ambient).
class GiPass {
  public:
    struct Config {
        uint32_t width = 0;
        uint32_t height = 0;
        VkImage hdrImage = VK_NULL_HANDLE;        ///< externally owned HDR render target (read+write)
        VkImageView hdrView = VK_NULL_HANDLE;     ///< storage view for the HDR image
        VkImage giBufferImage = VK_NULL_HANDLE;   ///< forward G-buffer: worldPos + materialIdx
        VkImageView giBufferView = VK_NULL_HANDLE;
        VkImage gbufferImage = VK_NULL_HANDLE;    ///< forward thin G-buffer: view-space normal + roughness
        VkImageView gbufferView = VK_NULL_HANDLE;
    };

    /// Per-frame scene + camera state consumed by record(). Most resources are
    /// scene-dependent (TLAS, material/geometry buffers, env CDF) and are rebound
    /// each frame, mirroring ForwardRenderer's descriptor update strategy.
    struct FrameParams {
        const Scene* scene = nullptr;
        VkImageView envMapView = VK_NULL_HANDLE;  ///< env panorama (or a harmless placeholder when no env)
        VkSampler envSampler = VK_NULL_HANDLE;
        VkBuffer envMarginalCdf = VK_NULL_HANDLE;     ///< may be VK_NULL_HANDLE (no env importance)
        VkBuffer envConditionalCdf = VK_NULL_HANDLE;  ///< may be VK_NULL_HANDLE
        uint32_t envImportanceWidth = 0;
        uint32_t envImportanceHeight = 0;
        bool hasEnvMap = false;
        float envLuminanceScale = 1.0f;

        glm::mat4 viewTransposed{1.0f}; ///< glm::transpose(view) — same convention as ForwardPC
        glm::vec3 cameraPos{0.0f};
        float exposure = 1.0f;
        uint32_t frameSampleIndex = 0;
        uint32_t rngBaseSeed = 0;
        uint32_t maxDepth = 3;
    };

    GiPass() = default;
    ~GiPass();
    GiPass(const GiPass&) = delete;
    GiPass& operator=(const GiPass&) = delete;

    [[nodiscard]] bool initialize(const DeviceContext& ctx, const Config& cfg, const char* giSpv = "gi.comp.spv");
    void shutdown();

    /// Dispatch the GI accumulation pass. Call AFTER ForwardRenderer::recordFrame()
    /// in the same command buffer. Transitions:
    ///   giBuffer / gbuffer : ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    ///   hdr                : ATTACHMENT_OPTIMAL -> GENERAL (stays GENERAL after this call)
    void record(VkCommandBuffer cmd, const FrameParams& params);

    [[nodiscard]] bool isInitialized() const noexcept { return m_pipeline != VK_NULL_HANDLE; }

  private:
    struct GiPushConstants {
        glm::mat4 view{1.0f};
        glm::vec4 cameraPos{0.0f};
        float exposure = 1.0f;
        uint32_t frameSampleIndex = 0;
        uint32_t rngBaseSeed = 0;
        uint32_t emissiveTriangleCount = 0;
        uint32_t envImportanceWidth = 0;
        uint32_t envImportanceHeight = 0;
        uint32_t hasEnvMap = 0;
        float envLuminanceScale = 1.0f;
        uint32_t maxDepth = 3;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t _pad0 = 0;
    };
    static_assert(sizeof(GiPushConstants) == 128);

    [[nodiscard]] bool createDescriptors();
    [[nodiscard]] bool createPipeline(const char* giSpv);
    void updateDescriptors(const FrameParams& params);
    [[nodiscard]] bool descriptorsDirty(const FrameParams& params) const;

    const DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    bool m_hdrFirstUse = true;
    const Scene* m_boundScene = nullptr;
    VkImageView m_boundEnvMapView = VK_NULL_HANDLE;
    VkSampler m_boundEnvSampler = VK_NULL_HANDLE;
    VkBuffer m_boundEnvMarginalCdf = VK_NULL_HANDLE;
    VkBuffer m_boundEnvConditionalCdf = VK_NULL_HANDLE;
};

} // namespace theia
