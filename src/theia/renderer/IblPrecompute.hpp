#pragma once

#include <volk/volk.h>

#include <vector>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/Image.hpp"

namespace theia {

struct IblResources {
    Image sheenLut;                        // 512×512 R16F  — Charlie sheen directional albedo
    Image diffuseIrrad;                    // 256×128 RGBA16F — Lambertian-convolved irradiance
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
    bool initialize(const DeviceContext& ctx,
                    const CommandPool& pool,
                    VkImageView envImageView,
                    VkSampler envSampler,
                    float envUnitNits = 1.0f);
    void shutdown();

    const IblResources& resources() const { return m_res; }
    bool isValid() const { return m_initialized; }

  private:
    bool createTextures();
    bool createSamplers();
    bool runSheenLutPass(VkCommandBuffer cmd);
    bool runDiffusePass(VkCommandBuffer cmd);
    bool runSpecularPass(VkCommandBuffer cmd);
    bool createComputePipeline(const char* spirvPath,
                               VkDescriptorSetLayout layout,
                               uint32_t pushConstantSize,
                               VkPipeline& outPipeline,
                               VkPipelineLayout& outLayout);
    void destroyTemporaryObjects() noexcept;

    const DeviceContext* m_ctx = nullptr;
    const CommandPool* m_pool = nullptr;
    VkImageView m_envImageView = VK_NULL_HANDLE;
    VkSampler m_envSampler = VK_NULL_HANDLE;
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
