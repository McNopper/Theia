#include "hyperion/scene/Mesh.hpp"

#include <volk/volk.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vma/vk_mem_alloc.h>

namespace {
[[nodiscard]] std::expected<Buffer, VkResult> createAndUpload(const DeviceContext& ctx,
                                                              const CommandPool& cmdPool,
                                                              std::span<const std::byte> bytes,
                                                              VkBufferUsageFlags usage,
                                                              std::string_view name) {
    auto staging = Buffer::create(ctx,
                                  static_cast<VkDeviceSize>(bytes.size()),
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  std::string(name).append(".staging"));
    if (!staging) {
        return std::unexpected(staging.error());
    }
    staging->uploadData(bytes.data(), bytes.size(), 0);

    auto deviceBuffer = Buffer::create(ctx,
                                       static_cast<VkDeviceSize>(bytes.size()),
                                       usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                       name);
    if (!deviceBuffer) {
        return std::unexpected(deviceBuffer.error());
    }

    auto cmd = cmdPool.beginOneShot();
    if (!cmd) {
        return std::unexpected(cmd.error());
    }

    const VkBufferCopy copy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = static_cast<VkDeviceSize>(bytes.size()),
    };
    vkCmdCopyBuffer(*cmd, staging->handle(), deviceBuffer->handle(), 1, &copy);

    if (const VkResult result = cmdPool.endOneShot(*cmd); result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    return std::move(*deviceBuffer);
}
} // namespace

std::expected<Mesh, VkResult>
Mesh::create(const DeviceContext& ctx, const CommandPool& cmdPool, MeshData data, std::string_view debugName) {
    if (data.vertices.empty() || data.indices.empty()) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    const auto vertexBytes = std::as_bytes(std::span<const GpuVertex>(data.vertices.data(), data.vertices.size()));
    const auto indexBytes = std::as_bytes(std::span<const uint32_t>(data.indices.data(), data.indices.size()));

    const std::string baseName = debugName.empty() ? std::string{"mesh"} : std::string{debugName};
    auto vertexBuffer = createAndUpload(ctx,
                                        cmdPool,
                                        vertexBytes,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                        baseName + ".vertices");
    if (!vertexBuffer) {
        return std::unexpected(vertexBuffer.error());
    }

    auto indexBuffer = createAndUpload(ctx,
                                       cmdPool,
                                       indexBytes,
                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                       baseName + ".indices");
    if (!indexBuffer) {
        return std::unexpected(indexBuffer.error());
    }

    Mesh mesh;
    mesh.m_vertexBuffer = std::move(*vertexBuffer);
    mesh.m_indexBuffer = std::move(*indexBuffer);
    mesh.m_vertexCount = static_cast<uint32_t>(data.vertices.size());
    mesh.m_indexCount = static_cast<uint32_t>(data.indices.size());
    return mesh;
}
