#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>
#include <vector>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Image.hpp"

namespace theia {

/// Screen-Space Reflections pass (linear ray march + composite).
///
/// Runs two compute dispatches after the forward geometry pass:
///   1. SSR ray march  : reads depth + thin GBuffer, writes RGBA16F hit buffer
///   2. SSR composite  : adds SSR hit buffer into the HDR render target
///
/// Also owns the barrier transitions that prepare the render targets for compute
/// (ATTACHMENT_OPTIMAL → SHADER_READ_ONLY / GENERAL) and leaves HDR in GENERAL
/// for the downstream ToneMapper.
///
/// Ref: McGuire & Mara — "Efficient GPU Screen-Space Ray Tracing" (JCGT 2014)
/// Ref: AMD FidelityFX SSSR (MIT) — github.com/GPUOpen-Effects/FidelityFX-SSSR
class SSRPass {
  public:
    struct Config {
        uint32_t width = 0;
        uint32_t height = 0;
        VkImage depthImage = VK_NULL_HANDLE;      ///< ForwardRenderer depth buffer image
        VkImageView depthView = VK_NULL_HANDLE;   ///< ForwardRenderer depth buffer view
        VkImage gbufferImage = VK_NULL_HANDLE;    ///< ForwardRenderer thin GBuffer image
        VkImageView gbufferView = VK_NULL_HANDLE; ///< ForwardRenderer thin GBuffer view
        VkImage hdrImage = VK_NULL_HANDLE;        ///< externally owned HDR render target
        VkImageView hdrView = VK_NULL_HANDLE;     ///< view for HDR image (sampled + storage)
        float ssrStrength = 1.0f;                 ///< global blend [0,1], adjustable at runtime
        float ssgiStrength = 0.0f;                ///< optional screen-space GI complement [0,1]
    };

    SSRPass() = default;
    ~SSRPass();
    SSRPass(const SSRPass&) = delete;
    SSRPass& operator=(const SSRPass&) = delete;

    /// SPIR-V filenames are resolved against THEIA_SHADER_DIR.
    [[nodiscard]] bool initialize(const DeviceContext& ctx,
                                  const Config& cfg,
                                  const char* ssrSpv = "ssr.comp.spv",
                                  const char* compositeSpv = "ssr_composite.comp.spv",
                                  const char* ssaoSpv = "ssao.comp.spv",
                                  const char* ssaoBlurSpv = "ssao_blur.comp.spv");

    void shutdown();

    /// Run the SSR ray march + composite into the HDR buffer.
    ///
    /// Call AFTER ForwardRenderer::recordFrame() in the same command buffer.
    /// Transitions:
    ///   depth    : ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    ///   gbuffer  : ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    ///   hdr      : ATTACHMENT_OPTIMAL → GENERAL (stays GENERAL after this call)
    ///
    /// @param proj     GLM perspective matrix (Y-flipped, GLM_FORCE_DEPTH_ZERO_TO_ONE)
    /// @param invProj  inverse(proj)
    void dispatch(VkCommandBuffer cmd, const glm::mat4& proj, const glm::mat4& invProj, bool afterGi = false);

    void setSSRStrength(float s) { m_cfg.ssrStrength = s; }
    void setSSAOStrength(float s) { m_ssaoStrength = s; }
    void setSSGIStrength(float s) { m_ssgiStrength = s; }
    [[nodiscard]] bool isInitialized() const noexcept { return m_ssrPipeline != VK_NULL_HANDLE; }

  private:
    struct SSRPushConstants {
        glm::mat4 proj;    ///< GLM projection (not transposed)
        glm::mat4 invProj; ///< inverse(proj)
    };
    static_assert(sizeof(SSRPushConstants) == 128);

    struct DepthPyramidPushConstants {
        uint32_t srcMip = 0;
        uint32_t _pad0  = 0;
        uint32_t _pad1  = 0;
        uint32_t _pad2  = 0;
    };
    static_assert(sizeof(DepthPyramidPushConstants) == 16);

    struct CompositePushConstants {
        float ssrStrength  = 0.0f;
        float roughnessMax = 0.0f;
        float _pad0        = 0.0f;
        float _pad1        = 0.0f;
    };
    static_assert(sizeof(CompositePushConstants) == 16);

    // SSAO / contact-shadow pass push constants.
    struct SSAOPushConstants {
        glm::mat4 proj;    ///< GLM projection (not transposed)
        glm::mat4 invProj; ///< inverse(proj)
        float radius    = 0.0f; ///< view-space sampling radius (scene units)
        float intensity = 0.0f; ///< occlusion strength multiplier
        float bias      = 0.0f; ///< depth bias to avoid self-occlusion
        float power     = 0.0f; ///< AO falloff exponent
    };
    static_assert(sizeof(SSAOPushConstants) == 144);

    // SSAO bilateral-blur push constants.
    struct SSAOBlurPushConstants {
        glm::mat4 invProj;   ///< inverse(proj), to linearise depth for edge-aware weighting
        glm::vec2 texelSize; ///< 1/width, 1/height
        float     _pad0 = 0.0f;
        float     _pad1 = 0.0f;
    };
    static_assert(sizeof(SSAOBlurPushConstants) == 80);

    [[nodiscard]] bool
    createPipelines(const char* ssrSpv,
                    const char* compositeSpv,
                    const char* ssaoSpv,
                    const char* ssaoBlurSpv,
                    const char* ssgiSpv);
    [[nodiscard]] bool createDescriptors();
    [[nodiscard]] bool createSamplers();
    void updateDescriptors();

    const DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    // Owned resources
    Image m_ssrResult; ///< RGBA16F: rgb=reflected color, a=confidence
    Image m_depthPyramid; ///< R32F min-depth pyramid for hierarchical SSR
    VkSampler m_samplerNearest = VK_NULL_HANDLE;
    VkSampler m_samplerLinear = VK_NULL_HANDLE;

    // Depth pyramid generation pipeline
    VkPipeline m_depthPyramidPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_depthPyramidLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_depthPyramidSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_depthPyramidPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_depthPyramidSets;
    std::vector<VkImageView> m_depthPyramidMipViews;
    uint32_t m_depthPyramidMipCount = 1;
    bool m_depthPyramidFirstUse = true;

    // SSR ray march pipeline
    VkPipeline m_ssrPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_ssrLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_ssrSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_ssrPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ssrSet = VK_NULL_HANDLE;

    // Composite pipeline
    VkPipeline m_compositePipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_compositeLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_compositeSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_compositePool = VK_NULL_HANDLE;
    VkDescriptorSet m_compositeSet = VK_NULL_HANDLE;

    // SSAO / contact-shadow pipeline (reads depth+gbuffer, writes AO factor to m_ssaoResult)
    VkPipeline m_ssaoPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_ssaoLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_ssaoSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_ssaoPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ssaoSet = VK_NULL_HANDLE;
    float m_ssaoStrength = 1.0f;

    // SSAO bilateral blur pipeline (denoises m_ssaoResult, then attenuates the
    // indirect/ambient fraction of HDR using the alpha channel as a weight)
    Image m_ssaoResult; ///< R16F: raw (noisy) AO factor written by SSAO pass
    VkPipeline m_ssaoBlurPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_ssaoBlurLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_ssaoBlurSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_ssaoBlurPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ssaoBlurSet = VK_NULL_HANDLE;
    bool m_ssaoResultFirstUse = true;

    // Optional SSGI pass (screen-space color bleed / low-frequency bounce).
    VkPipeline m_ssgiPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_ssgiLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_ssgiSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_ssgiPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ssgiSet = VK_NULL_HANDLE;
    float m_ssgiStrength = 0.0f;

    bool m_ssrResultFirstUse = true;
};

} // namespace theia
