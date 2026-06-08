#pragma once

#include <volk/volk.h>

#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/GpuTypes.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/CommandPool.hpp"

struct MeshData {
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
};

class Mesh {
  public:
    Mesh() = default;
    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    ~Mesh() = default;

    [[nodiscard]] static std::expected<Mesh, VkResult>
    create(const DeviceContext& ctx, const CommandPool& cmdPool, MeshData data, std::string_view debugName = "");

    [[nodiscard]] const Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] uint32_t vertexCount() const noexcept { return m_vertexCount; }
    [[nodiscard]] uint32_t indexCount() const noexcept { return m_indexCount; }

  private:
    Buffer m_vertexBuffer{};
    Buffer m_indexBuffer{};
    uint32_t m_vertexCount{};
    uint32_t m_indexCount{};
};
