#include "hyperion/renderer/Descriptors.hpp"

#include <volk/volk.h>

#include <array>
#include <utility>

#include "hyperion/scene/Scene.hpp"

Descriptors::~Descriptors() noexcept {
    reset();
}

Descriptors::Descriptors(Descriptors&& other) noexcept
    : m_ctx(other.m_ctx),
      m_set0Layout(std::exchange(other.m_set0Layout, VK_NULL_HANDLE)),
      m_set1Layout(std::exchange(other.m_set1Layout, VK_NULL_HANDLE)),
      m_pool(std::exchange(other.m_pool, VK_NULL_HANDLE)),
      m_set1(std::exchange(other.m_set1, VK_NULL_HANDLE)),
      m_pipelineLayout(std::exchange(other.m_pipelineLayout, VK_NULL_HANDLE)) {
    other.m_ctx = nullptr;
}

Descriptors& Descriptors::operator=(Descriptors&& other) noexcept {
    if (this != &other) {
        reset();
        m_ctx = other.m_ctx;
        m_set0Layout = std::exchange(other.m_set0Layout, VK_NULL_HANDLE);
        m_set1Layout = std::exchange(other.m_set1Layout, VK_NULL_HANDLE);
        m_pool = std::exchange(other.m_pool, VK_NULL_HANDLE);
        m_set1 = std::exchange(other.m_set1, VK_NULL_HANDLE);
        m_pipelineLayout = std::exchange(other.m_pipelineLayout, VK_NULL_HANDLE);
        other.m_ctx = nullptr;
    }
    return *this;
}

std::expected<Descriptors, VkResult> Descriptors::create(const DeviceContext& ctx) {
    constexpr std::array set0Bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo set0Info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = static_cast<uint32_t>(set0Bindings.size()),
        .pBindings = set0Bindings.data(),
    };

    constexpr std::array set1Bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
        VkDescriptorSetLayoutBinding{9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
    };
    constexpr std::array bindingFlags{
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                 VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT),
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT),
        VkDescriptorBindingFlags{},
        VkDescriptorBindingFlags(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT), // env marginal CDF
        VkDescriptorBindingFlags(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT), // env conditional CDF
    };
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo set1Info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &bindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(set1Bindings.size()),
        .pBindings = set1Bindings.data(),
    };

    Descriptors descriptors;
    descriptors.m_ctx = &ctx;

    if (const VkResult result = vkCreateDescriptorSetLayout(ctx.device, &set0Info, nullptr, &descriptors.m_set0Layout);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    if (const VkResult result = vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &descriptors.m_set1Layout);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    constexpr std::array poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1025},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    if (const VkResult result = vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptors.m_pool);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptors.m_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptors.m_set1Layout,
    };
    if (const VkResult result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptors.m_set1);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    constexpr VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    const std::array layouts{descriptors.m_set0Layout, descriptors.m_set1Layout};
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    if (const VkResult result =
            vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &descriptors.m_pipelineLayout);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    ctx.setDebugName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                     reinterpret_cast<uint64_t>(descriptors.m_set0Layout),
                     "hyperion.set0.push");
    ctx.setDebugName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                     reinterpret_cast<uint64_t>(descriptors.m_set1Layout),
                     "hyperion.set1.scene");
    ctx.setDebugName(
        VK_OBJECT_TYPE_DESCRIPTOR_POOL, reinterpret_cast<uint64_t>(descriptors.m_pool), "hyperion.scene.pool");
    ctx.setDebugName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                     reinterpret_cast<uint64_t>(descriptors.m_pipelineLayout),
                     "hyperion.pipelineLayout");

    return descriptors;
}

VkResult Descriptors::updateSceneSet(const DeviceContext& ctx, const Scene& scene) {
    const std::array bufferInfos{
        VkDescriptorBufferInfo{scene.instanceBuffer().handle(), 0, VK_WHOLE_SIZE},
        VkDescriptorBufferInfo{scene.materialBuffer().handle(), 0, VK_WHOLE_SIZE},
        VkDescriptorBufferInfo{scene.vertexBuffer().handle(), 0, VK_WHOLE_SIZE},
        VkDescriptorBufferInfo{scene.indexBuffer().handle(), 0, VK_WHOLE_SIZE},
        VkDescriptorBufferInfo{scene.lightBuffer().handle(), 0, VK_WHOLE_SIZE},
        VkDescriptorBufferInfo{scene.emissiveTriangleBuffer().handle(), 0, VK_WHOLE_SIZE},
    };
    const std::array writes{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[0],
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[1],
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[2],
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             3,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[3],
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             5,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[4],
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             7,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &bufferInfos[5],
                             nullptr},
    };
    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Bind each scene texture to binding 4 (bindless combined image sampler array).
    for (uint32_t i = 0; i < static_cast<uint32_t>(scene.textures().size()); ++i) {
        const VkDescriptorImageInfo imageInfo{
            .sampler = scene.textures()[i].sampler(),
            .imageView = scene.textures()[i].image().view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = m_set1,
            .dstBinding = 4,
            .dstArrayElement = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &imageInfo,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }

    return VK_SUCCESS;
}

VkResult Descriptors::updateEnvMap(const DeviceContext& ctx, VkImageView view, VkSampler sampler) {
    const VkDescriptorImageInfo imageInfo{
        .sampler = sampler,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = m_set1,
        .dstBinding = 6,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &imageInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return VK_SUCCESS;
}

VkResult Descriptors::updateEnvImportance(const DeviceContext& ctx, VkBuffer marginalCdf, VkBuffer conditionalCdf) {
    const VkDescriptorBufferInfo marginalInfo{.buffer = marginalCdf, .offset = 0, .range = VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo conditionalInfo{.buffer = conditionalCdf, .offset = 0, .range = VK_WHOLE_SIZE};
    const std::array writes{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             8,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &marginalInfo,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set1,
                             9,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &conditionalInfo,
                             nullptr},
    };
    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return VK_SUCCESS;
}

void Descriptors::reset() noexcept {
    if (m_ctx != nullptr && m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_ctx != nullptr && m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx->device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    if (m_ctx != nullptr && m_set1Layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_set1Layout, nullptr);
        m_set1Layout = VK_NULL_HANDLE;
    }
    if (m_ctx != nullptr && m_set0Layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_set0Layout, nullptr);
        m_set0Layout = VK_NULL_HANDLE;
    }
    m_set1 = VK_NULL_HANDLE;
    m_ctx = nullptr;
}
