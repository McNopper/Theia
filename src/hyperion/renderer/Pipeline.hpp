#pragma once

#include <volk/volk.h>

#include <expected>
#include <filesystem>

#include "hyperion/DeviceContext.hpp"

class Descriptors;

class Pipeline {
  public:
    struct ShaderPaths {
        std::filesystem::path raygen;
        std::filesystem::path closesthitTriangle;
        std::filesystem::path closesthitSphere;
        std::filesystem::path intersection;
        std::filesystem::path miss;
        std::filesystem::path shadowMiss;
    };

    Pipeline() = default;
    ~Pipeline() noexcept;

    Pipeline(Pipeline&& other) noexcept;
    Pipeline& operator=(Pipeline&& other) noexcept;

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    [[nodiscard]] static std::expected<Pipeline, VkResult> create(const DeviceContext& ctx,
                                                                  const Descriptors& descriptors,
                                                                  const ShaderPaths& paths,
                                                                  uint32_t maxRayRecursion = 8);

    [[nodiscard]] VkPipeline rtPipeline() const noexcept { return m_rtPipeline; }

  private:
    void reset() noexcept;

    const DeviceContext* m_ctx{};
    VkPipeline m_rtPipeline{VK_NULL_HANDLE};
};
