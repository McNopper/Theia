#pragma once

#include <volk/volk.h>

#include <expected>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/GpuTypes.hpp"

class Scene;

class Descriptors {
  public:
    Descriptors() = default;
    ~Descriptors() noexcept;

    Descriptors(Descriptors&& other) noexcept;
    Descriptors& operator=(Descriptors&& other) noexcept;

    Descriptors(const Descriptors&) = delete;
    Descriptors& operator=(const Descriptors&) = delete;

    [[nodiscard]] static std::expected<Descriptors, VkResult> create(const DeviceContext& ctx);

    VkResult updateSceneSet(const DeviceContext& ctx, const Scene& scene);
    VkResult updateEnvMap(const DeviceContext& ctx, VkImageView view, VkSampler sampler);
    VkResult updateEnvImportance(const DeviceContext& ctx, VkBuffer marginalCdf, VkBuffer conditionalCdf);

    [[nodiscard]] VkDescriptorSetLayout set0Layout() const noexcept { return m_set0Layout; }
    [[nodiscard]] VkDescriptorSetLayout set1Layout() const noexcept { return m_set1Layout; }
    [[nodiscard]] VkDescriptorSet set1() const noexcept { return m_set1; }
    [[nodiscard]] VkPipelineLayout pipelineLayout() const noexcept { return m_pipelineLayout; }

  private:
    void reset() noexcept;

    const DeviceContext* m_ctx{};
    VkDescriptorSetLayout m_set0Layout{VK_NULL_HANDLE};
    VkDescriptorSetLayout m_set1Layout{VK_NULL_HANDLE};
    VkDescriptorPool m_pool{VK_NULL_HANDLE};
    VkDescriptorSet m_set1{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
};
