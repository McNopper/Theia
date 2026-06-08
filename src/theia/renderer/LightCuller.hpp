#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/Buffer.hpp"

namespace theia {

/// Forward+ tile-based light culler.
/// Runs a compute pass each frame that bins lights into 16×16 pixel screen tiles.
/// Results are stored in GPU buffers and bound to ForwardRenderer set 1, bindings 3–4.
///
/// Ref: Olsson & Assarsson — "Tiled Shading" (JCGT 2011)
class LightCuller {
  public:
    static constexpr uint32_t kTileSize = 16;
    static constexpr uint32_t kMaxLightsPerTile = 128;

    LightCuller() = default;
    ~LightCuller();

    LightCuller(const LightCuller&) = delete;
    LightCuller& operator=(const LightCuller&) = delete;
    LightCuller(LightCuller&&) noexcept;
    LightCuller& operator=(LightCuller&&) noexcept;

    /// Initialize for a given render resolution.
    /// @param spvPath  Path to spirv/light_cull.comp.spv
    [[nodiscard]] bool initialize(const DeviceContext& ctx,
                                  uint32_t screenWidth,
                                  uint32_t screenHeight,
                                  const char* spvPath = "spirv/light_cull.comp.spv");

    void shutdown();

    /// Run the light culling compute pass.
    /// lightBuffer: scene light SSBO; lightCount: number of active lights.
    /// projView: projection and view matrices for this frame.
    void dispatch(VkCommandBuffer cmd,
                  VkBuffer lightBuffer,
                  uint32_t lightCount,
                  const glm::mat4& proj,
                  const glm::mat4& view,
                  float nearZ,
                  float farZ);

    // GPU buffers read by ForwardRenderer
    [[nodiscard]] VkBuffer tileLightCountsBuffer() const noexcept { return m_tileLightCountsBuf.handle(); }
    [[nodiscard]] VkBuffer tileLightIndicesBuffer() const noexcept { return m_tileLightIndicesBuf.handle(); }
    [[nodiscard]] uint32_t tilesX() const noexcept { return m_tilesX; }
    [[nodiscard]] uint32_t tilesY() const noexcept { return m_tilesY; }

  private:
    const DeviceContext* m_ctx = nullptr;

    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    Buffer m_tileLightCountsBuf;  // uint[tilesX*tilesY]
    Buffer m_tileLightIndicesBuf; // uint[tilesX*tilesY * kMaxLightsPerTile]

    uint32_t m_tilesX = 0;
    uint32_t m_tilesY = 0;
    uint32_t m_screenWidth = 0;
    uint32_t m_screenHeight = 0;
};

} // namespace theia
