#ifndef THEIA_RENDERER_LIGHTCULLER_HPP
#define THEIA_RENDERER_LIGHTCULLER_HPP

#include <volk/volk.h>

#include <cstdint>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/VulkanHandle.hpp"

namespace theia {

/// Forward+ tile-based light culler.
/// Runs a compute pass each frame that bins lights into 16×16 pixel screen tiles.
/// Results are stored in GPU buffers and bound to ForwardRenderer set 1, bindings 3–4.
///
/// Ref: Olsson & Assarsson — "Tiled Shading" (J. Graphics, GPU, and Game Tools 15(4), 2011)
class LightCuller {
  public:
    static constexpr std::uint32_t kTileSize = 16;
    static constexpr std::uint32_t kMaxLightsPerTile = 128;

    LightCuller() = default;
    ~LightCuller();

    LightCuller(const LightCuller&) = delete;
    LightCuller& operator=(const LightCuller&) = delete;
    LightCuller(LightCuller&&) noexcept = default;
    LightCuller& operator=(LightCuller&&) noexcept = default;

    /// Initialize for a given render resolution.
    /// @param spvFilename  SPIR-V filename resolved against THEIA_SHADER_DIR.
    [[nodiscard]] bool initialize(const harmonia::DeviceContext& ctx,
                                  std::uint32_t screenWidth,
                                  std::uint32_t screenHeight,
                                  const char* spvFilename = "light_cull.comp.spv");

    void shutdown();

    /// Run the light culling compute pass.
    /// lightBuffer: scene light SSBO; lightCount: number of active lights.
    /// projView: projection and view matrices for this frame.
    void dispatch(VkCommandBuffer cmd,
                  VkBuffer lightBuffer,
                  std::uint32_t lightCount,
                  const sm::float4x4& proj,
                  const sm::float4x4& view,
                  float nearZ,
                  float farZ);

    // GPU buffers read by ForwardRenderer
    [[nodiscard]] VkBuffer tileLightCountsBuffer() const noexcept { return m_tileLightCountsBuf.handle(); }
    [[nodiscard]] VkBuffer tileLightIndicesBuffer() const noexcept { return m_tileLightIndicesBuf.handle(); }
    [[nodiscard]] std::uint32_t tilesX() const noexcept { return m_tilesX; }
    [[nodiscard]] std::uint32_t tilesY() const noexcept { return m_tilesY; }

  private:
    const harmonia::DeviceContext* m_ctx = nullptr;

    harmonia::UniquePipeline m_pipeline;
    harmonia::UniquePipelineLayout m_pipelineLayout;
    harmonia::UniqueDescriptorSetLayout m_setLayout;
    harmonia::UniqueDescriptorPool m_pool;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    harmonia::Buffer m_tileLightCountsBuf;  // uint[tilesX*tilesY]
    harmonia::Buffer m_tileLightIndicesBuf; // uint[tilesX*tilesY * kMaxLightsPerTile]

    std::uint32_t m_tilesX = 0;
    std::uint32_t m_tilesY = 0;
    std::uint32_t m_screenWidth = 0;
    std::uint32_t m_screenHeight = 0;
};

} // namespace theia
#endif // THEIA_RENDERER_LIGHTCULLER_HPP
