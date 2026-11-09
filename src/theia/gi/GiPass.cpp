#include "theia/gi/GiPass.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <harmonia/core/Barrier.hpp>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/PipelineBinaryCache.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <slang-math/slang-math.hpp>
#include <span>
#include <theia/gi/PairingTextures.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace theia {

namespace {
/// GI-ENH: deterministic per-frame pairing-texture permutation (ReSTIR PT Enhanced
/// §3.2 — flip/mirror/transpose/offset so the static maps' tiling never bakes a
/// fixed spatial pattern into the accumulated image). 19 packed bits:
/// 0-2 = mirrorY|mirrorX|transpose, 3-10 = offsetX, 11-18 = offsetY.
[[nodiscard]] std::uint32_t pairingPermFor(std::uint32_t frame, std::uint32_t texIdx) {
    std::uint64_t z = static_cast<std::uint64_t>(frame) * 0x9E3779B97F4A7C15ULL +
                      static_cast<std::uint64_t>(texIdx + 1U) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;
    return static_cast<std::uint32_t>(z & 0x7FFFFU);
}

constexpr std::uint64_t kPairingSeed = 0x50414952494E47ULL; // "PAIRING"
} // namespace

GiPass::~GiPass() {
    shutdown();
}

bool GiPass::initialize(const harmonia::DeviceContext& ctx,
                        const Config& cfg,
                        const harmonia::core::CommandPool& uploadPool,
                        const char* giSpv) {
    m_ctx = &ctx;
    m_cfg = cfg;
    m_frameError = VK_ERROR_INITIALIZATION_FAILED;
    m_frameRecorded = m_smsFrameActive = false;
    m_hdrFirstUse = true;
    m_boundScene = nullptr;
    m_texturesBoundFor = nullptr;
    m_boundEnvMapView = VK_NULL_HANDLE;
    m_boundEnvSampler = VK_NULL_HANDLE;
    m_boundEnvMarginalCdf = nullptr;
    m_boundEnvConditionalCdf = nullptr;
    m_boundGradientVarianceView = VK_NULL_HANDLE;
    // VK14/nullDescriptor: bindings 15 (gradient/variance) and 18 (motion vectors) no
    // longer need 1×1 dummy images — VK_NULL_HANDLE is bound and the shader reads return
    // zero (the push-constant flags hasGradientVariance=0 / restirHasMotion=0 already gate
    // the read paths). The dummy images, their layout transitions, and their state tracking
    // are all deleted.

    // A4: ReSTIR DI reservoir ping-pong buffers. Sized to hold one Reservoir per pixel
    // at a generous stride (kReservoirStride) covering the Slang struct layout.
    const VkDeviceSize reservoirBytes =
        static_cast<VkDeviceSize>(cfg.width) * static_cast<VkDeviceSize>(cfg.height) * kReservoirStride;
    if (reservoirBytes > 0) {
        for (auto& slot : m_reservoirBuf) {
            auto buf =
                harmonia::core::Buffer::create(ctx,
                                               reservoirBytes,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                               "theia.gi.restirReservoir");
            if (!buf) {
                harmonia::core::Logger::error("GiPass: failed to create ReSTIR reservoir buffer: VkResult {}",
                                              static_cast<int>(buf.error()));
                return false;
            }
            slot = std::move(*buf);
        }
        // GI2 full PT: path reservoir ping-pong buffers (same per-pixel stride policy).
        const VkDeviceSize pathReservoirBytes =
            static_cast<VkDeviceSize>(cfg.width) * static_cast<VkDeviceSize>(cfg.height) * kPathReservoirStride;
        for (auto& slot : m_pathReservoirBuf) {
            auto buf =
                harmonia::core::Buffer::create(ctx,
                                               pathReservoirBytes,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                               "theia.gi.restirPathReservoir");
            if (!buf) {
                harmonia::core::Logger::error("GiPass: failed to create ReSTIR path reservoir buffer: VkResult {}",
                                              static_cast<int>(buf.error()));
                return false;
            }
            slot = std::move(*buf);
        }
    }

    // GI-ENH: Gaussian paired-neighbor pairing textures (self-inverse offset maps
    // built deterministically on the CPU, uploaded once, read-only at runtime).
    {
        const std::vector<std::uint32_t> packed = theia::pairing::buildPackedOffsets(kPairingSeed);
        auto buf = harmonia::core::Buffer::upload(ctx,
                                                  uploadPool,
                                                  std::as_bytes(std::span<const std::uint32_t>(packed)),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                  "theia.gi.pairingOffsets");
        if (!buf) {
            harmonia::core::Logger::error("GiPass: failed to upload pairing offsets: VkResult {}",
                                          static_cast<int>(buf.error()));
            return false;
        }
        m_pairingOffsetBuf = std::move(*buf);
    }
    m_reservoirPingPong = 0;
    m_reservoirsCleared = false;
    m_pathReservoirPingPong = 0;
    m_pathReservoirsCleared = false;
    m_boundMotionVectorView = VK_NULL_HANDLE;

    if (!createDescriptors()) {
        return false;
    }
    if (!createPipelineLayout()) {
        return false;
    }
    // GI-ENH §6.2.2: emit megakernel + uniform replay pass + per-pixel combine.
    if (!createComputePipeline(giSpv, "main", "theia.gi", m_pipeline)) {
        return false;
    }
    if (!createComputePipeline("gi.comp.replayMain.spv", "replayMain", "theia.gi.replay", m_replayPipeline)) {
        return false;
    }
    if (!createComputePipeline("gi.comp.combineMain.spv", "combineMain", "theia.gi.combine", m_combinePipeline)) {
        return false;
    }
    // Replay-compaction buffers (bindings 27-30; see GiPass.hpp). The compaction
    // is a pure restructuring of the replay path — these exist whenever the GI
    // pass runs and stay dormant (count = 0) when the path reservoir is off.
    {
        const std::uint64_t pixels = static_cast<std::uint64_t>(m_cfg.width) * static_cast<std::uint64_t>(m_cfg.height);
        const std::uint64_t pixelSlots = pixels * 3u; // kPathSpatialNeighbors (shader constant)
        for (std::uint32_t inst = 0; inst < 2; ++inst) {
            const std::vector<std::pair<harmonia::core::Buffer*, std::pair<VkDeviceSize, const char*>>> bufs{
                {&m_replayRecordsBuf[inst], {pixelSlots * 4, "theia.gi.replayRecords"}},
                {&m_replayCountBuf[inst], {kReplayCountBytes, "theia.gi.replayCount"}},
                {&m_ptParamsBuf[inst], {pixels * kPtParamsStride, "theia.gi.ptParams"}},
                {&m_replayResultsBuf[inst], {pixelSlots * kReplayResultStride, "theia.gi.replayResults"}},
            };
            for (auto& [buf, spec] : bufs) {
                auto created = harmonia::core::Buffer::create(
                    ctx,
                    spec.first,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    spec.second);
                if (!created) {
                    harmonia::core::Logger::error(
                        "GiPass: failed to create replay-compaction buffer {}[{}]: VkResult {}",
                        spec.second,
                        inst,
                        static_cast<int>(created.error()));
                    return false;
                }
                *buf = std::move(*created);
            }
        }
    }
    if (!createSmsResources(uploadPool))
        return false;
    m_frameError = VK_SUCCESS;
    return true;
}

void GiPass::shutdown() {
    m_sms = harmonia::pipeline::SmsPass{};
    m_smsContext = {};
    m_smsPrefixes = {};
    m_smsPrior = {};
    m_frameRecorded = m_smsFrameActive = false;
    m_frameError = VK_SUCCESS;
    m_reservoirBuf[0] = {};
    m_reservoirBuf[1] = {};
    m_reservoirsCleared = false;
    m_reservoirPingPong = 0;
    m_pathReservoirBuf[0] = {};
    m_pathReservoirBuf[1] = {};
    m_pathReservoirsCleared = false;
    m_pathReservoirPingPong = 0;
    m_pairingOffsetBuf = {};
    m_replayRecordsBuf[0] = {};
    m_replayRecordsBuf[1] = {};
    m_replayCountBuf[0] = {};
    m_replayCountBuf[1] = {};
    m_ptParamsBuf[0] = {};
    m_ptParamsBuf[1] = {};
    m_replayResultsBuf[0] = {};
    m_replayResultsBuf[1] = {};
    m_boundMotionVectorView = VK_NULL_HANDLE;
    m_pipeline.reset();
    m_replayPipeline.reset();
    m_combinePipeline.reset();
    m_pipelineLayout.reset();
    for (auto& w : m_descWriters) {
        w.shutdown(); // MOD1: replaces pool + sets
    }
    m_setLayout.reset();
    m_boundScene = nullptr;
    m_texturesBoundFor = nullptr;
    m_boundEnvMapView = VK_NULL_HANDLE;
    m_boundEnvSampler = VK_NULL_HANDLE;
    m_boundEnvMarginalCdf = nullptr;
    m_boundEnvConditionalCdf = nullptr;
    m_boundGradientVarianceView = VK_NULL_HANDLE;
}

bool GiPass::createDescriptors() {
    const std::array<VkDescriptorSetLayoutBinding, 31> bindings{{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // TLAS
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // hdr (RW)
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // giBuffer
        {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // gbuffer
        {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // envMap
        {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                    // env sampler
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // materials
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // vertices
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // instances
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // indices
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // emissive tris
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // marginal CDF
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // conditional CDF
        {13,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         kMaxBindlessTextures,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},                                                                        // textures
        {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // emissive power CDF
        {15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // A3(b) gradient/variance
        {16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // A4 reservoirs (cur, RW)
        {17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // A4 reservoirs (prev, RO)
        {18, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // A4 motion vectors
        {19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // instance transforms
        {20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI2 PT path reservoirs
                                                                                          // (cur, RW)
        {21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI2 PT path reservoirs
                                                                                          // (prev, RO)
        {22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH pairing maps
        {23, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH shift share (cur,
                                                                                          // RW)
        {24, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH shift share (prev,
                                                                                          // RO)
        {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // immutable SMS producer
                                                                                          // context
        {26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // optional SMS counters
        {27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH §6.2.2 replay
                                                                                          // records (RW)
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH §6.2.2 replay
                                                                                          // count + indirect cmd (RW)
        {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH §6.2.2 per-pixel
                                                                                          // emit params (RW)
        {30, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // GI-ENH §6.2.2 replay
                                                                                          // results (RW)
    }};

    // MOD1: no UPDATE_AFTER_BIND — incompatible with DESCRIPTOR_BUFFER layouts.
    constexpr VkDescriptorBindingFlags kTextureFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const std::array<VkDescriptorBindingFlags, 31> bindingFlags{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kTextureFlags, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };

    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &bindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    VkDescriptorSetLayout setLayout{};
    if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &setLayout) != VK_SUCCESS) {
        harmonia::core::Logger::error("GiPass: failed to create descriptor set layout");
        return false;
    }
    m_setLayout = harmonia::core::UniqueDescriptorSetLayout{m_ctx->device, setLayout};

    // MOD1: descriptor buffers (replaces the pool + allocated sets).
    for (std::uint32_t si = 0; si < kDescriptorSlots; ++si) {
        if (!m_descWriters[si].init(*m_ctx,
                                    setLayout,
                                    static_cast<std::uint32_t>(bindings.size()),
                                    si == 0 ? "theia.gi.descBuf0" : "theia.gi.descBuf1")) {
            harmonia::core::Logger::error("GiPass: failed to create descriptor buffer slot {}", si);
            for (std::uint32_t j = 0; j <= si; ++j)
                m_descWriters[j].shutdown();
            return false;
        }
    }

    // GI-ENH pairing maps (binding 22) are static after creation — write once per slot.
    for (std::uint32_t si = 0; si < kDescriptorSlots; ++si) {
        m_descWriters[si].writeStorageBufferHandle(*m_ctx, 22, m_pairingOffsetBuf);
    }
    return true;
}

bool GiPass::createPipelineLayout() {
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(GiPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_setLayout.ptr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    VkPipelineLayout pipelineLayout{};
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        harmonia::core::Logger::error("GiPass: failed to create pipeline layout");
        return false;
    }
    m_pipelineLayout = harmonia::core::UniquePipelineLayout{m_ctx->device, pipelineLayout};
    return true;
}

bool GiPass::createComputePipeline(const char* spv,
                                   const char* /*entry*/,
                                   const char* cacheName,
                                   harmonia::core::UniquePipeline& out) {
    auto module = harmonia::core::createShaderModule(m_ctx->device, shaderPath(spv));
    if (!module) {
        harmonia::core::Logger::error("GiPass: cannot load shader: {}", spv);
        return false;
    }

    const VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, // MOD1
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = *module,
                  // slangc -entry emits the selected function named "main" in
                  // the SPIR-V, so every per-entry spv's entry point is "main".
                  .pName = "main"},
        .layout = m_pipelineLayout,
    };
    auto pipelineResult = harmonia::core::PipelineBinaryCache::computePipeline(*m_ctx, pipeInfo, cacheName);
    vkDestroyShaderModule(m_ctx->device, *module, nullptr);
    if (!pipelineResult) {
        harmonia::core::Logger::error("GiPass: failed to create compute pipeline {} (spv {})", cacheName, spv);
        return false;
    }
    out = harmonia::core::UniquePipeline{m_ctx->device, *pipelineResult};
    return true;
}

void GiPass::updateDescriptors(const FrameParams& params) {
    const Scene* scene = params.scene;

    VkAccelerationStructureKHR tlas = scene->tlas();

    // Env CDF buffers may be absent (no env map); bind the material buffer as a harmless
    // placeholder so the descriptor stays valid — the shader gates all reads on hasEnvMap.
    const harmonia::core::Buffer& cdfFallback = scene->materialBuffer();
    const harmonia::core::Buffer& marginalCdf =
        (params.envMarginalCdf != nullptr) ? *params.envMarginalCdf : cdfFallback;
    const harmonia::core::Buffer& conditionalCdf =
        (params.envConditionalCdf != nullptr) ? *params.envConditionalCdf : cdfFallback;

    // Scene bindings are frame-independent — write them into EVERY slot's descriptor buffer.
    const bool texturesDirty = (m_texturesBoundFor != scene);
    for (std::uint32_t si = 0; si < kDescriptorSlots; ++si) {
        auto& w = m_descWriters[si];
        // MOD1: typed descriptor buffer writes.
        w.writeAccelerationStructure(*m_ctx, 0, tlas);
        w.writeStorageImage(*m_ctx, 1, m_cfg.hdrView, VK_IMAGE_LAYOUT_GENERAL);
        w.writeSampledImage(*m_ctx, 2, m_cfg.giBufferView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        w.writeSampledImage(*m_ctx, 3, m_cfg.gbufferView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        w.writeSampledImage(*m_ctx, 4, params.envMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        w.writeSampler(*m_ctx, 5, params.envSampler);
        w.writeStorageBufferHandle(*m_ctx, 6, scene->materialBuffer());
        w.writeStorageBufferHandle(*m_ctx, 7, scene->vertexBuffer());
        w.writeStorageBufferHandle(*m_ctx, 8, scene->instanceBuffer());
        w.writeStorageBufferHandle(*m_ctx, 9, scene->indexBuffer());
        w.writeStorageBufferHandle(*m_ctx, 10, scene->emissiveTriangleBuffer());
        w.writeStorageBufferHandle(*m_ctx, 11, marginalCdf);
        w.writeStorageBufferHandle(*m_ctx, 12, conditionalCdf);
        w.writeStorageBufferHandle(*m_ctx, 14, scene->emissiveCdfBuffer());
        w.writeSampledImage(*m_ctx, 15, params.gradientVarianceView, VK_IMAGE_LAYOUT_GENERAL);
        w.writeStorageBufferHandle(*m_ctx, 19, scene->instanceTransformBuffer());

        // Bindless textures (binding 13) — write when scene changes.
        if (texturesDirty) {
            const auto& sceneTextures = scene->textures();
            const std::uint32_t texCount =
                std::min(static_cast<std::uint32_t>(sceneTextures.size()), kMaxBindlessTextures);
            for (std::uint32_t i = 0; i < texCount; ++i) {
                w.writeCombinedImageSampler(*m_ctx,
                                            13,
                                            i,
                                            sceneTextures[i].sampler(),
                                            sceneTextures[i].image().view(),
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }
    m_texturesBoundFor = scene;
}

void GiPass::updateRestirDescriptors(const FrameParams& params, std::uint32_t slot) {
    // Per-frame bindings are written ONLY into the given slot's descriptor buffer —
    // the other slot may hold a frame that is still in flight (the descriptor-race fix).
    auto& w = m_descWriters[slot % kDescriptorSlots];
    m_activeSlot = slot % kDescriptorSlots;
    // A4: bindings 16 (cur reservoir), 17 (prev reservoir), 18 (motion vectors).
    // GI2 full PT: bindings 20/21 mirror 16/17 for the path reservoir ping-pong.
    if (!m_reservoirBuf[0].isValid() || !m_reservoirBuf[1].isValid() || !m_pathReservoirBuf[0].isValid() ||
        !m_pathReservoirBuf[1].isValid()) {
        return;
    }
    const std::uint32_t cur = m_reservoirPingPong;
    const std::uint32_t prev = 1u - m_reservoirPingPong;
    const std::uint32_t pathCur = m_pathReservoirPingPong;
    const std::uint32_t pathPrev = 1u - m_pathReservoirPingPong;

    // MOD1: typed descriptor buffer writes.
    w.writeStorageBufferHandle(*m_ctx, 16, m_reservoirBuf[cur]);
    w.writeStorageBufferHandle(*m_ctx, 17, m_reservoirBuf[prev]);
    w.writeSampledImage(*m_ctx, 18, params.motionVectorView, VK_IMAGE_LAYOUT_GENERAL);
    w.writeStorageBufferHandle(*m_ctx, 20, m_pathReservoirBuf[pathCur]);
    w.writeStorageBufferHandle(*m_ctx, 21, m_pathReservoirBuf[pathPrev]);
    // GI-ENH §6.2.2 replay-compaction buffers — the ACTIVE slot's instances
    // (frame-transient; the other slot may hold the pipelined previous frame).
    if (m_replayRecordsBuf[0].isValid() && m_replayCountBuf[0].isValid() && m_ptParamsBuf[0].isValid() &&
        m_replayResultsBuf[0].isValid()) {
        const std::uint32_t inst = m_activeSlot;
        w.writeStorageBufferHandle(*m_ctx, 27, m_replayRecordsBuf[inst]);
        w.writeStorageBufferHandle(*m_ctx, 28, m_replayCountBuf[inst]);
        w.writeStorageBufferHandle(*m_ctx, 29, m_ptParamsBuf[inst]);
        w.writeStorageBufferHandle(*m_ctx, 30, m_replayResultsBuf[inst]);
    }
    m_boundMotionVectorView = params.motionVectorView;
}

bool GiPass::descriptorsDirty(const FrameParams& params) const {
    return m_boundScene != params.scene || m_boundEnvMapView != params.envMapView ||
           m_boundEnvSampler != params.envSampler || m_boundEnvMarginalCdf != params.envMarginalCdf ||
           m_boundEnvConditionalCdf != params.envConditionalCdf ||
           m_boundGradientVarianceView != params.gradientVarianceView;
}

void GiPass::record(VkCommandBuffer cmd, const FrameParams& params, bool skipPreBarriers) {
    if (m_frameError != VK_SUCCESS)
        return;
    m_frameRecorded = m_smsFrameActive = false;
    if (m_pipeline == VK_NULL_HANDLE || params.scene == nullptr || m_cfg.hdrImage == VK_NULL_HANDLE) {
        m_frameError = VK_ERROR_INITIALIZATION_FAILED;
        return;
    }
    m_smsFrameActive = usesSms(params.scene->emissiveTriangleCount(), params.maxDepth);
    m_smsDiagnosticsReported = false;
    if (cmd == VK_NULL_HANDLE || (m_smsFrameActive && skipPreBarriers)) {
        m_frameError = VK_ERROR_INITIALIZATION_FAILED;
        return;
    }

    if (descriptorsDirty(params)) {
        updateDescriptors(params);
        m_boundScene = params.scene;
        m_boundEnvMapView = params.envMapView;
        m_boundEnvSampler = params.envSampler;
        m_boundEnvMarginalCdf = params.envMarginalCdf;
        m_boundEnvConditionalCdf = params.envConditionalCdf;
        m_boundGradientVarianceView = params.gradientVarianceView;
    }

    // A4: one-time zero-fill of both reservoir buffers so uninitialized entries read as
    // empty (M == 0 → skipped by temporal reuse; no garbage samples injected).
    // GI2 full PT: same for the path reservoir buffers.
    if (!m_reservoirsCleared && m_reservoirBuf[0].isValid() && m_reservoirBuf[1].isValid() &&
        m_pathReservoirBuf[0].isValid() && m_pathReservoirBuf[1].isValid()) {
        vkCmdFillBuffer(cmd, m_reservoirBuf[0].handle(), 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, m_reservoirBuf[1].handle(), 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, m_pathReservoirBuf[0].handle(), 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, m_pathReservoirBuf[1].handle(), 0, VK_WHOLE_SIZE, 0u);
        const std::array<VkBufferMemoryBarrier2, 4> fillBarriers{{
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_reservoirBuf[0].handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_reservoirBuf[1].handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_pathReservoirBuf[0].handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_pathReservoirBuf[1].handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
        }};
        const VkDependencyInfo fillDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(fillBarriers.size()),
            .pBufferMemoryBarriers = fillBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &fillDep);
        m_reservoirsCleared = true;
        m_pathReservoirsCleared = true;
    }

    // A4/GI2: cross-frame reservoir ping-pong dependency — this frame's dispatch READS
    // as `Prev` exactly what the PREVIOUS frame's dispatch wrote as `Cur`. Without this
    // barrier that write→read pair has no execution or memory dependency (the one-time
    // fill barrier above covers only the initial CLEAR): on the async-compute path the two
    // dispatches overlap and the read races the write → run-to-run nondeterminism in
    // multi-frame captures (plain missing-barrier bug, surfaced 2026-09).
    {
        auto pingPongBarrier = [](VkBuffer buf) {
            return VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .buffer = buf,
                                          .offset = 0,
                                          .size = VK_WHOLE_SIZE};
        };
        const std::array<VkBufferMemoryBarrier2, 4> pingPongBarriers{{
            pingPongBarrier(m_reservoirBuf[0].handle()),
            pingPongBarrier(m_reservoirBuf[1].handle()),
            pingPongBarrier(m_pathReservoirBuf[0].handle()),
            pingPongBarrier(m_pathReservoirBuf[1].handle()),
        }};
        const VkDependencyInfo pingPongDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(pingPongBarriers.size()),
            .pBufferMemoryBarriers = pingPongBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &pingPongDep);
    }

    // A4: rebind the reservoir ping-pong + motion vectors for this frame's slot.
    updateRestirDescriptors(params, params.frameSampleIndex % kDescriptorSlots);
    // ---- Barriers: forward attachments -> compute inputs/outputs ----
    if (!skipPreBarriers) {
        const std::array<VkImageMemoryBarrier2, 3> preBarriers{{
            harmonia::core::imageBarrier(m_cfg.giBufferImage,
                                         VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_READ_BIT),
            harmonia::core::imageBarrier(m_cfg.gbufferImage,
                                         VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_READ_BIT),
            harmonia::core::imageBarrier(m_cfg.hdrImage,
                                         VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT),
        }};
        const VkDependencyInfo preDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<std::uint32_t>(preBarriers.size()),
            .pImageMemoryBarriers = preBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &preDep);
    }
    m_hdrFirstUse = false;

    const GiPushConstants pc{
        .view = params.view,
        .prevViewProj = params.prevViewProj,
        .cameraPos = sm::float4(params.cameraPos, 1.0f),
        .exposure = params.exposure,
        .frameSampleIndex = params.frameSampleIndex,
        .rngBaseSeed = params.rngBaseSeed,
        .emissiveTriangleCount = params.scene->emissiveTriangleCount(),
        .envImportanceWidth = params.envImportanceWidth,
        .envImportanceHeight = params.envImportanceHeight,
        .hasEnvMap = params.hasEnvMap ? 1u : 0u,
        .envLuminanceScale = params.envLuminanceScale,
        .maxDepth = std::max(1u, params.maxDepth),
        .screenWidth = m_cfg.width,
        .screenHeight = m_cfg.height,
        ._pad0 = 0u,
        .a3RegularizationEnabled = params.useA3Regularization ? 1u : 0u,
        .hasGradientVariance = (params.gradientVarianceView != VK_NULL_HANDLE) ? 1u : 0u,
        .giAdaptiveMaxSamples = std::max(1u, params.adaptiveMaxSamples),
        // GI2 Phase 2: the reservoir path serves both legacy ReSTIR DI and ReSTIR PT
        // (PT absorbs DI — sprint plan GI2 conflict resolution). restirDiEnabled here
        // means "reservoir active"; restirDiSpatial enables the Phase 2 neighbour-reuse
        // pass (on by default for PT).
        .restirDiEnabled = ((params.useRestirDi || params.useRestirPt) && m_reservoirBuf[0].isValid()) ? 1u : 0u,
        .restirDiSpatial = params.useRestirPt ? 1u : 0u,
        .restirHasMotion = (params.motionVectorView != VK_NULL_HANDLE) ? 1u : 0u,
        .restirPtEnabled = params.useRestirPt ? 1u : 0u,
        // GI2 full PT: path reservoir for the indirect term. Independent of
        // useRestirPt at this level (the application gates it), but requires the
        // path reservoir buffers.
        .restirPtPathEnabled = (params.useRestirPtPath && m_pathReservoirBuf[0].isValid()) ? 1u : 0u,
        .fireflyClampEnabled = params.fireflyClampEnabled ? 1u : 0u,
        .pairingPerm0 = pairingPermFor(params.frameSampleIndex, 0u),
        .pairingPerm1 = pairingPermFor(params.frameSampleIndex, 1u),
        .pairingPerm2 = pairingPermFor(params.frameSampleIndex, 2u),
        .pairingEnabled = m_pairingOffsetBuf.isValid() ? 1u : 0u,
        .reconnectShiftEnabled = params.reconnectShiftEnabled,
        .smsEnabled = m_smsFrameActive ? 1U : 0U,
    };

    const bool replayCompaction = (pc.restirPtPathEnabled != 0u) && m_replayCountBuf[0].isValid() &&
                                  m_ptParamsBuf[0].isValid() && m_replayRecordsBuf[0].isValid() &&
                                  m_replayResultsBuf[0].isValid();
    const std::uint32_t rInst = m_activeSlot; // this frame's transient-buffer instance
    if (replayCompaction) {
        // Per-frame reset of the indirect command + record counter (words 0-4).
        vkCmdFillBuffer(cmd, m_replayCountBuf[rInst].handle(), 0, kReplayCountBytes, 0u);
        const VkBufferMemoryBarrier2 countReset{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                                .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                .dstAccessMask =
                                                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .buffer = m_replayCountBuf[rInst].handle(),
                                                .offset = 0,
                                                .size = kReplayCountBytes};
        const VkDependencyInfo resetDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &countReset,
        };
        vkCmdPipelineBarrier2(cmd, &resetDep);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    // MOD1: bind the descriptor buffer for the active slot (replaces vkCmdBindDescriptorSets).
    m_descWriters[m_activeSlot].bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const std::uint32_t gx = (m_cfg.width + 7u) / 8u;
    const std::uint32_t gy = (m_cfg.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    if (replayCompaction) {
        // GI-ENH §6.2.2: emit → uniform replay → combine. The emit kernel's
        // replay jobs (atomically compacted stream) and per-pixel emit params
        // must be visible to the replay pass; the count words double as the
        // indirect dispatch command (DRAW_INDIRECT read).
        auto bufBarrier = [](VkBuffer buf, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            return VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                          .dstStageMask = dstStage,
                                          .dstAccessMask = dstAccess,
                                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .buffer = buf,
                                          .offset = 0,
                                          .size = VK_WHOLE_SIZE};
        };
        const std::array<VkBufferMemoryBarrier2, 4> emitToReplay{{
            bufBarrier(m_replayCountBuf[rInst].handle(),
                       VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT),
            bufBarrier(m_replayRecordsBuf[rInst].handle(),
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT),
            bufBarrier(
                m_ptParamsBuf[rInst].handle(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT),
            // combineMain re-merges from the local reservoir the emit stage stored.
            bufBarrier(m_pathReservoirBuf[m_pathReservoirPingPong].handle(),
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT),
        }};
        const VkDependencyInfo emitToReplayDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(emitToReplay.size()),
            .pBufferMemoryBarriers = emitToReplay.data(),
        };
        vkCmdPipelineBarrier2(cmd, &emitToReplayDep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_replayPipeline);
        vkCmdDispatchIndirect(cmd, m_replayCountBuf[rInst].handle(), 0);

        // Replay → combine: replay results + the emit stage's partial HDR
        // composite (combineMain reads and finishes the pixel's HDR write).
        const std::array<VkBufferMemoryBarrier2, 1> replayToCombine{{
            bufBarrier(m_replayResultsBuf[rInst].handle(),
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT),
        }};
        const VkImageMemoryBarrier2 hdrForCombine =
            harmonia::core::imageBarrier(m_cfg.hdrImage,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        const std::array<VkImageMemoryBarrier2, 1> hdrBarriers{hdrForCombine};
        const VkDependencyInfo replayToCombineDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(replayToCombine.size()),
            .pBufferMemoryBarriers = replayToCombine.data(),
            .imageMemoryBarrierCount = static_cast<std::uint32_t>(hdrBarriers.size()),
            .pImageMemoryBarriers = hdrBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &replayToCombineDep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_combinePipeline);
        vkCmdDispatch(cmd, gx, gy, 1);
    }
    if (m_smsFrameActive) {
        if (m_smsPrior.isValid())
            vkCmdFillBuffer(cmd, m_smsPrior.handle(), 0, VK_WHOLE_SIZE, 0);
        if (m_smsTemporalReservoir.isValid() && !m_smsReservoirsCleared) {
            // Probe pattern: 0xDEADBEEF proves the fill executed and lets the
            // host dump distinguish filled-but-unwritten entries from garbage.
            vkCmdFillBuffer(cmd, m_smsTemporalReservoir.handle(), 0, VK_WHOLE_SIZE, 0xDEADBEEFu);
            vkCmdFillBuffer(cmd, m_smsHistoryReservoir.handle(), 0, VK_WHOLE_SIZE, 0xDEADBEEFu);
            m_smsReservoirsCleared = true;
        }
        if (m_smsTemporalReservoir.isValid() && m_smsHistoryReservoir.isValid()) {
            // Publish last frame's pre-spatial merge as this frame's temporal
            // history BEFORE the passes run (the reference swaps its temporal
            // buffer in at the same point). Recorded ahead of the SMS passes,
            // like the one-time fill above.
            VkBufferCopy copy{0, 0, m_smsTemporalReservoir.size()};
            vkCmdCopyBuffer(cmd, m_smsTemporalReservoir.handle(), m_smsHistoryReservoir.handle(), 1, &copy);
        }
        m_frameError = m_sms.recordBegin(cmd);
        if (m_frameError == VK_SUCCESS)
            m_frameError = m_sms.recordBatch(cmd);
        if (m_frameError == VK_SUCCESS && m_smsTemporalStaging.isValid() && m_smsHistoryStaging.isValid()) {
            // Diagnostics ground truth: mirror the device-local reservoirs into
            // the host-visible staging pair after the passes (observation only).
            VkBufferCopy temporalCopy{0, 0, m_smsTemporalReservoir.size()};
            vkCmdCopyBuffer(cmd, m_smsTemporalReservoir.handle(), m_smsTemporalStaging.handle(), 1, &temporalCopy);
            VkBufferCopy historyCopy{0, 0, m_smsHistoryReservoir.size()};
            vkCmdCopyBuffer(cmd, m_smsHistoryReservoir.handle(), m_smsHistoryStaging.handle(), 1, &historyCopy);
        }
    }
    m_frameRecorded = m_frameError == VK_SUCCESS;

    // A4: flip the reservoir ping-pong so this frame's "current" becomes next frame's "prev".
    m_reservoirPingPong = 1u - m_reservoirPingPong;
    m_pathReservoirPingPong = 1u - m_pathReservoirPingPong;
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
