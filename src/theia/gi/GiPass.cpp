#include "theia/gi/GiPass.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <harmonia/core/Barrier.hpp>
#include <harmonia/core/Logger.hpp>
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

/// One compute pipeline from a compiled .spv against the shared layout
/// (descriptor-buffer flag per MOD1).
[[nodiscard]] bool
createComputePipeline(VkDevice device, VkPipelineLayout layout, const char* spv, harmonia::core::UniquePipeline& out) {
    auto module = harmonia::core::createShaderModule(device, shaderPath(spv));
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
                  .pName = "main"},
        .layout = layout,
    };
    VkPipeline pipeline{};
    const VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device, *module, nullptr);
    if (res != VK_SUCCESS) {
        harmonia::core::Logger::error("GiPass: failed to create compute pipeline ({})", spv);
        return false;
    }
    out = harmonia::core::UniquePipeline{device, pipeline};
    return true;
}
} // namespace

GiPass::~GiPass() {
    shutdown();
}

bool GiPass::initialize(const harmonia::DeviceContext& ctx,
                        const Config& cfg,
                        const harmonia::core::CommandPool& uploadPool,
                        const char* giSpv,
                        const char* generateSpv,
                        const char* traceSpv) {
    m_ctx = &ctx;
    m_cfg = cfg;
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

    // PERF5 10b: wavefront path buffers (design §2 byte math: pixels × 8 slots ×
    // stride; ray buffer ×2 for the wave ping-pong). No one-time fill — the
    // generate pass rewrites every consumed slot each frame (the records' alive
    // bit is the truth). These are structural now: the GI dispatch consumes the
    // wave-0 records, so a creation failure fails the pass (GI off), same as a
    // pipeline failure.
    {
        const VkDeviceSize pixels = static_cast<VkDeviceSize>(cfg.width) * static_cast<VkDeviceSize>(cfg.height);
        const VkDeviceSize wavePathBytes = pixels * kWaveSlotsPerPixel * kWavePathStride;
        const VkDeviceSize waveHitBytes = pixels * kWaveSlotsPerPixel * kWaveHitStride;
        const VkDeviceSize waveCollectBytes = pixels * kWaveCollectStride;
        for (auto& slot : m_wavePathBuf) {
            auto buf = harmonia::core::Buffer::create(ctx,
                                                      wavePathBytes,
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                      "theia.gi.wavePath");
            if (!buf) {
                harmonia::core::Logger::error("GiPass: failed to create wavefront path buffer: VkResult {}",
                                              static_cast<int>(buf.error()));
                return false;
            }
            slot = std::move(*buf);
        }
        auto hitBuf = harmonia::core::Buffer::create(ctx,
                                                     waveHitBytes,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                     "theia.gi.waveHit");
        if (!hitBuf) {
            harmonia::core::Logger::error("GiPass: failed to create wavefront hit buffer: VkResult {}",
                                          static_cast<int>(hitBuf.error()));
            return false;
        }
        m_waveHitBuf = std::move(*hitBuf);
        auto collectBuf = harmonia::core::Buffer::create(ctx,
                                                         waveCollectBytes,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                         "theia.gi.waveCollect");
        if (!collectBuf) {
            harmonia::core::Logger::error("GiPass: failed to create wavefront collect buffer: VkResult {}",
                                          static_cast<int>(collectBuf.error()));
            return false;
        }
        m_waveCollectBuf = std::move(*collectBuf);
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
    m_wavePathPingPong = 0;
    m_boundMotionVectorView = VK_NULL_HANDLE;

    if (!createDescriptors()) {
        return false;
    }
    if (!createPipelines(giSpv, generateSpv, traceSpv)) {
        return false;
    }
    return true;
}

void GiPass::shutdown() {
    m_reservoirBuf[0] = {};
    m_reservoirBuf[1] = {};
    m_reservoirsCleared = false;
    m_reservoirPingPong = 0;
    m_pathReservoirBuf[0] = {};
    m_pathReservoirBuf[1] = {};
    m_pathReservoirsCleared = false;
    m_pathReservoirPingPong = 0;
    m_wavePathBuf[0] = {};
    m_wavePathBuf[1] = {};
    m_wavePathPingPong = 0;
    m_waveHitBuf = {};
    m_waveCollectBuf = {};
    m_pairingOffsetBuf = {};
    m_boundMotionVectorView = VK_NULL_HANDLE;
    m_pipeline.reset();
    m_generatePipeline.reset();
    m_tracePipeline.reset();
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
    const std::array<VkDescriptorSetLayoutBinding, 29> bindings{{
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
        {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // PERF5 10b wavefront ray
                                                                                          // buffer (cur, RW)
        {26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // PERF5 wavefront ray buffer
                                                                                          // (prev, RW; 10d successors)
        {27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // PERF5 10c wavefront hits
                                                                                          // (trace writes; GI
                                                                                          // consumes since 10c)
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // PERF5 wavefront collect
                                                                                          // (10d)
    }};

    // MOD1: no UPDATE_AFTER_BIND — incompatible with DESCRIPTOR_BUFFER layouts.
    constexpr VkDescriptorBindingFlags kTextureFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const std::array<VkDescriptorBindingFlags, 29> bindingFlags{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kTextureFlags, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        if (!m_descWriters[si].init(*m_ctx, setLayout, 29, si == 0 ? "theia.gi.descBuf0" : "theia.gi.descBuf1")) {
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

bool GiPass::createPipelines(const char* giSpv, const char* generateSpv, const char* traceSpv) {
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

    // PERF5 wavefront: three compute pipelines against the shared layout — the
    // generate pass (camera-ray enqueue + collect init), the 10c trace pass
    // (verbatim RayQuery over the wave-0 records → hit buffer), and the GI
    // shade pass (consumes records + hits). Recorded in that order, with
    // buffer barriers between the dispatches.
    if (!createComputePipeline(m_ctx->device, m_pipelineLayout, generateSpv, m_generatePipeline)) {
        return false;
    }
    if (!createComputePipeline(m_ctx->device, m_pipelineLayout, traceSpv, m_tracePipeline)) {
        return false;
    }
    if (!createComputePipeline(m_ctx->device, m_pipelineLayout, giSpv, m_pipeline)) {
        return false;
    }
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
    // PERF5 10b: wavefront buffers — the ray buffer is the wave transport, the
    // hit buffer the per-wave trace output, the collect the per-pixel
    // termination record. 10d: the ray REGENERATE is in-place and ONE physical
    // buffer (m_wavePathBuf[0], binding 25) serves every wave — each shade
    // thread reads and writes only its own pixel's slots, so the inter-dispatch
    // wave barriers are the only ordering the scheme needs; a physical
    // ping-pong would force every fixed-width wave to re-write every dead
    // slot's 160-B record, which dominated the frame on unified memory.
    // m_wavePathBuf[1] stays allocated as the design's second ray half — the
    // 10e stream-compaction scratch (bound, but read by no wavefront stage
    // this stage). The collect carries the per-sample composite chains
    // (written across the waves, read by the last wave's composite). Written
    // every frame like the reservoir ping-pong bindings.
    w.writeStorageBufferHandle(*m_ctx, 25, m_wavePathBuf[0]);
    w.writeStorageBufferHandle(*m_ctx, 26, m_wavePathBuf[1]);
    w.writeStorageBufferHandle(*m_ctx, 27, m_waveHitBuf);
    w.writeStorageBufferHandle(*m_ctx, 28, m_waveCollectBuf);
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
    m_boundMotionVectorView = params.motionVectorView;
}

bool GiPass::descriptorsDirty(const FrameParams& params) const {
    return m_boundScene != params.scene || m_boundEnvMapView != params.envMapView ||
           m_boundEnvSampler != params.envSampler || m_boundEnvMarginalCdf != params.envMarginalCdf ||
           m_boundEnvConditionalCdf != params.envConditionalCdf ||
           m_boundGradientVarianceView != params.gradientVarianceView;
}

void GiPass::record(VkCommandBuffer cmd, const FrameParams& params, bool skipPreBarriers) {
    if (m_pipeline == VK_NULL_HANDLE || params.scene == nullptr || m_cfg.hdrImage == VK_NULL_HANDLE) {
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
        const std::array<VkBufferMemoryBarrier2, 8> pingPongBarriers{{
            pingPongBarrier(m_reservoirBuf[0].handle()),
            pingPongBarrier(m_reservoirBuf[1].handle()),
            pingPongBarrier(m_pathReservoirBuf[0].handle()),
            pingPongBarrier(m_pathReservoirBuf[1].handle()),
            // PERF5 10b: wavefront buffers. The same-frame generate→trace→GI
            // write→read edges are ordered by the dedicated barriers below; this
            // cross-frame barrier keeps frame N's accesses ahead of frame N+1's
            // (the wave path + hit buffers are rewritten every frame by
            // generate/trace, the collect buffer follows in 10d).
            pingPongBarrier(m_wavePathBuf[0].handle()),
            pingPongBarrier(m_wavePathBuf[1].handle()),
            pingPongBarrier(m_waveHitBuf.handle()),
            pingPongBarrier(m_waveCollectBuf.handle()),
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
    };

    const std::uint32_t gx = (m_cfg.width + 7u) / 8u;
    const std::uint32_t gy = (m_cfg.height + 7u) / 8u;

    // ---- PERF5 10b: generate pass — G-buffer load, camera-ray enqueue, collect ----
    // ---- init (design §1). Runs ONCE per frame (wave 0 only); same full-width  ----
    // ---- dispatch shape as the GI pass; shares the pipeline layout, so the GI ----
    // ---- push constants are pushed as-is (waveIndex = 0).                    ----
    {
        GiPushConstants genPc = pc;
        genPc.waveIndex = 0u;
        genPc.waveCount =
            pc.restirPtPathEnabled != 0u ? 1u : pc.maxDepth + 1u; // (unused in generate; keep the block consistent)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_generatePipeline);
        m_descWriters[m_activeSlot].bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(genPc), &genPc);
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    // generate → trace: this frame's wave-0 records must be visible to the trace
    // dispatch (write→read on the ray buffer it consumes — binding 25 aliases
    // m_wavePathBuf[0] since the 10d in-place regenerate); the collect init rides
    // along (read|write dst keeps it coherent for the 10d shade waves). Stages
    // and access flags follow the repo's compute-to-compute barrier convention.
    {
        const std::array<VkBufferMemoryBarrier2, 2> genBarriers{{
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_wavePathBuf[0].handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = m_waveCollectBuf.handle(),
             .offset = 0,
             .size = VK_WHOLE_SIZE},
        }};
        const VkDependencyInfo genDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(genBarriers.size()),
            .pBufferMemoryBarriers = genBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &genDep);
    }

    // ---- PERF5 10d: the wave loop — the (trace, shade) dispatch train.     ----
    // The train runs maxDepth + 1 pairs for the legacy per-sample walks (design
    // §1: closest-hit for every live path in the wave's ray buffer, then the
    // per-vertex shade + regenerate; dead slots exit on the alive flag,
    // fixed-width dispatches per the Q1 decision — no indirect dispatch; the
    // shade regenerates IN PLACE — one physical ray buffer, see the descriptor
    // comment — so each shade thread rewrites only its own pixel's slots and
    // the inter-dispatch barriers order everything). When the ReSTIR PT path
    // reservoir owns ALL multi-bounce transport (restirPtPathEnabled — the
    // default), every per-sample slot is PROVABLY dead after the wave-0 shade
    // (the shade kills it — no per-sample continuation exists), so waves >= 1
    // would dispatch nothing but alive-bit probes; on this machine a
    // fixed-width dispatch + barrier costs ~0.3 ms of pure launch overhead,
    // which is the frame at parity resolution. The CPU knows the mode (it
    // computes the push constant), so the train collapses to ONE wave there —
    // the composite runs in the wave-0 shade (waveCount = 1). No GPU readback,
    // no indirect dispatch: the decision is push-constant-constant per frame.
    // The full train remains for the legacy modes.
    const std::uint32_t waveCount = (pc.restirPtPathEnabled != 0u) ? 1u : (pc.maxDepth + 1u);
    for (std::uint32_t wave = 0u; wave < waveCount; ++wave) {
        GiPushConstants wavePc = pc;
        wavePc.waveIndex = wave;
        wavePc.waveCount = waveCount;

        // ---- trace pass: the verbatim traceClosestHit traversal over the ----
        // ---- live records of this wave's ray buffer → the hit buffer.    ----
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
            m_descWriters[m_activeSlot].bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(wavePc), &wavePc);
            vkCmdDispatch(cmd, gx, gy, 1);
        }

        // trace → shade: this wave's hit-record writes must be visible before the
        // shade dispatch reads them (the shade reconstructs each live slot's
        // surface from its record — record 0 for the wave-0 primary since 10c).
        // The records' own visibility chains through the barrier below (the shade
        // reads the half the PREVIOUS dispatch wrote).
        {
            const std::array<VkBufferMemoryBarrier2, 1> traceBarriers{{
                {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                 .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .buffer = m_waveHitBuf.handle(),
                 .offset = 0,
                 .size = VK_WHOLE_SIZE},
            }};
            const VkDependencyInfo traceDep{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(traceBarriers.size()),
                .pBufferMemoryBarriers = traceBarriers.data(),
            };
            vkCmdPipelineBarrier2(cmd, &traceDep);
        }

        // ---- shade dispatch: wave 0 = primary vertices + regenerate;     ----
        // ---- waves >= 1 = per-slot vertex shade + regenerate; the last   ----
        // ---- wave also composites the collect into the HDR target.       ----
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        m_descWriters[m_activeSlot].bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(wavePc), &wavePc);
        vkCmdDispatch(cmd, gx, gy, 1);

        if (wave + 1u < waveCount) {
            // shade → next trace: the regenerated successor records (written
            // IN PLACE — binding 25 serves every wave) must be visible
            // to the next wave's trace + shade dispatches, and the collect
            // writes (reservoir terms at wave 0, terminations at any wave) to
            // the later readers.
            const std::array<VkBufferMemoryBarrier2, 2> waveBarriers{{
                {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                 .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .buffer = m_wavePathBuf[0].handle(),
                 .offset = 0,
                 .size = VK_WHOLE_SIZE},
                {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                 .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .buffer = m_waveCollectBuf.handle(),
                 .offset = 0,
                 .size = VK_WHOLE_SIZE},
            }};
            const VkDependencyInfo waveDep{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(waveBarriers.size()),
                .pBufferMemoryBarriers = waveBarriers.data(),
            };
            vkCmdPipelineBarrier2(cmd, &waveDep);
        }
    }

    // A4: flip the reservoir ping-pong so this frame's "current" becomes next frame's "prev".
    m_reservoirPingPong = 1u - m_reservoirPingPong;
    m_pathReservoirPingPong = 1u - m_pathReservoirPingPong;
    // PERF5 10d: the ray regenerate is IN-PLACE (both ray bindings alias
    // m_wavePathBuf[0]), so the physical ping-pong is dissolved this stage — the
    // flip below is retained as bookkeeping for the 10e stream-compaction half
    // (m_wavePathBuf[1]); wave 0 is fully rewritten by the generate pass every
    // frame either way.
    m_wavePathPingPong = 1u - m_wavePathPingPong;
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
