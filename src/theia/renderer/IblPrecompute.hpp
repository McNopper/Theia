#ifndef THEIA_RENDERER_IBLPRECOMPUTE_HPP
#define THEIA_RENDERER_IBLPRECOMPUTE_HPP

#include <volk/volk.h>

#include <cstdint>
#include <vector>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/Image.hpp"

namespace theia {

struct IblResources {
    Image sheenLut;                        // 512×512 R16F  — Charlie sheen directional albedo
    Image brdfLut;                         // 512×512 RG16F — GGX split-sum BRDF integration (A/B)
    Image diffuseIrrad;                    // configurable RGBA16F — Lambertian-convolved irradiance
    Image specularMipped;                  // 512×256 RGBA16F, 8 mip levels — GGX prefiltered specular
    VkSampler lutSampler = VK_NULL_HANDLE; // clamp+linear, no mip (for sheenLut)
    VkSampler envSampler = VK_NULL_HANDLE; // clamp+linear-mip (diffuse + specular)
};

class IblPrecompute {
  public:
    IblPrecompute() = default;
    ~IblPrecompute();
    IblPrecompute(const IblPrecompute&) = delete;
    IblPrecompute& operator=(const IblPrecompute&) = delete;

    // Precompute all IBL textures. envImageView/envSampler may be VK_NULL_HANDLE
    // (no env map) — in that case, diffuse and specular textures will be black.
    //
    // marginalCdf / conditionalCdf are the GPU buffers built by IblProbe for env
    // importance sampling (PBR Book §12.5).  They must be provided together with
    // cdfWidth/cdfHeight when an env map is present; the diffuse irradiance pass
    // uses them instead of cosine-weighted sampling so that high-dynamic-range
    // panoramas (e.g. a visible sun disc) are represented correctly.
    bool initialize(const DeviceContext& ctx,
                    const CommandPool& pool,
                    VkImageView envImageView,
                    VkSampler envSampler,
                    float envUnitNits = 1.0f,
                    VkExtent2D diffuseExtent = VkExtent2D{256, 128},
                    VkBuffer marginalCdf = VK_NULL_HANDLE,
                    VkBuffer conditionalCdf = VK_NULL_HANDLE,
                    std::uint32_t cdfWidth = 0,
                    std::uint32_t cdfHeight = 0);
    void shutdown();

    const IblResources& resources() const { return m_res; }
    bool isValid() const { return m_initialized; }

  private:
    bool createTextures();
    bool createSamplers();
    bool runBrdfLutPass(VkCommandBuffer cmd);
    bool runSheenLutPass(VkCommandBuffer cmd);
    bool runLutPass(VkCommandBuffer cmd, Image& targetImage, const char* shaderName, const char* logLabel);
    bool runDiffusePass(VkCommandBuffer cmd);
    bool runSpecularPass(VkCommandBuffer cmd);
    bool createComputePipeline(const char* spirvPath,
                               VkDescriptorSetLayout layout,
                               std::uint32_t pushConstantSize,
                               VkPipeline& outPipeline,
                               VkPipelineLayout& outLayout);
    void destroyTemporaryObjects() noexcept;

    const DeviceContext* m_ctx = nullptr;
    const CommandPool* m_pool = nullptr;
    VkImageView m_envImageView = VK_NULL_HANDLE;
    VkSampler m_envSampler = VK_NULL_HANDLE;
    VkBuffer m_marginalCdf = VK_NULL_HANDLE;
    VkBuffer m_conditionalCdf = VK_NULL_HANDLE;
    std::uint32_t m_cdfWidth = 0;
    std::uint32_t m_cdfHeight = 0;
    VkExtent2D m_diffuseExtent{256, 128};
    IblResources m_res;
    float m_envUnitNits = 1.0f;

    bool m_initialized = false;
    std::vector<VkDescriptorPool> m_tempDescriptorPools;
    std::vector<VkDescriptorSetLayout> m_tempSetLayouts;
    std::vector<VkPipelineLayout> m_tempPipelineLayouts;
    std::vector<VkPipeline> m_tempPipelines;
    std::vector<VkImageView> m_tempImageViews;
};

} // namespace theia
#endif // THEIA_RENDERER_IBLPRECOMPUTE_HPP
