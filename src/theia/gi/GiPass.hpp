#ifndef THEIA_RENDERER_GIPASS_HPP
#define THEIA_RENDERER_GIPASS_HPP

#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/DescriptorBufferWriter.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "theia/renderer/RendererConstants.hpp"

namespace theia {

class Scene;

/// Ray-query global-illumination compute stage.
///
/// PERF5 wavefront pipeline (PLAN.md "PERF5 wavefront design (stage 10a)"):
/// records the wavefront dispatch train after the forward geometry pass —
///   1. generate (wavefront_generate, once per frame): G-buffer load, wave-0
///      camera-ray enqueue (all eight slots initialized every frame), collect init;
///   2. maxDepth + 1 (trace, shade) pairs (10d wave loop):
///        trace (wavefront_trace) — the verbatim traceClosestHit RayQuery
///          traversal over the live records of the current wave's ray buffer →
///          the WaveHit buffer (miss sentinel for misses); at wave 0 only kind 0
///          is traced (the slot-0 single-trace optimization — all four sample
///          slots hold the identical camera ray, shade reuses record 0's hit);
///        shade (gi.comp) — wave 0 shades the primary vertices (reservoir
///          passes + per-sample primary blocks); waves >= 1 resume one record
///          per slot (surface reconstruction, integrateSurface, inline null
///          chain + medium walk) and REGENERATE the next wave's successors into
///          the non-read ping-pong half; the last wave composites the collect.
///      Dead slots exit on the alive flag (fixed-width dispatches, Q1 — no
///      indirect dispatch); every slot is rewritten every wave (alive
///      continuation or explicit kill — the buffers are never cleared).
///   The resulting indirect (bounce) radiance accumulates IN the path records
///   and lands in the per-pixel WaveCollect (no atomics anywhere); the last
///   wave's shade composites it into the HDR render target — RT-GI is the
///   renderer's single indirect provider (there is no split-sum IBL fallback).
/// The three pipelines share one layout (descriptor set + push-constant block).
/// 10c: the trace pass is the single source of wave-0 hits — the shade dispatch
/// reconstructs the primary surface from hit record 0 (reconstructGiHit; the
/// miss sentinel triggers the Q1 G-buffer fallback).
///
/// Always recorded after the forward geometry pass, which therefore emits direct +
/// emission only (the `giEnabled` push-constant bit 1 also tells the forward pass to
/// skip its emissive-derived rect lights when ReSTIR DI/PT owns that term).
class GiPass {
  public:
    struct Config {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        VkImage hdrImage = VK_NULL_HANDLE;      ///< externally owned HDR render target (read+write)
        VkImageView hdrView = VK_NULL_HANDLE;   ///< storage view for the HDR image
        VkImage giBufferImage = VK_NULL_HANDLE; ///< forward G-buffer: worldPos + materialIdx
        VkImageView giBufferView = VK_NULL_HANDLE;
        VkImage gbufferImage = VK_NULL_HANDLE; ///< forward thin G-buffer: view-space normal + roughness
        VkImageView gbufferView = VK_NULL_HANDLE;
    };

    /// Per-frame scene + camera state consumed by record(). Most resources are
    /// scene-dependent (TLAS, material/geometry buffers, env CDF) and are rebound
    /// each frame, mirroring ForwardRenderer's descriptor update strategy.
    struct FrameParams {
        const Scene* scene = nullptr;
        VkImageView envMapView = VK_NULL_HANDLE; ///< env panorama (or a harmless placeholder when no env)
        VkSampler envSampler = VK_NULL_HANDLE;
        const harmonia::core::Buffer* envMarginalCdf = nullptr;    ///< may be nullptr (no env importance)
        const harmonia::core::Buffer* envConditionalCdf = nullptr; ///< may be nullptr
        std::uint32_t envImportanceWidth = 0;
        std::uint32_t envImportanceHeight = 0;
        bool hasEnvMap = false;
        float envLuminanceScale = 1.0f;

        sm::float4x4 view{1.0f}; ///< view matrix
        sm::float4x4 prevViewProj{
            1.0f}; ///< GI2: previous-frame view-projection for inline motion-vector reprojection (temporal reuse)
        sm::float3 cameraPos{0.0f};
        float exposure = 1.0f;
        std::uint32_t frameSampleIndex = 0;
        std::uint32_t rngBaseSeed = 0;
        std::uint32_t maxDepth = 3;

        /// A3(b): optional A-SVGF gradient/variance guide (R32G32F, R = gradient,
        /// G = variance) for the variance-aware adaptive firefly clamp. When
        /// VK_NULL_HANDLE, a 1×1 dummy is bound and the shader falls back to the
        /// legacy fixed clamp (bit-identical behaviour). A provided view must be
        /// in VK_IMAGE_LAYOUT_GENERAL for the compute dispatch.
        VkImageView gradientVarianceView = VK_NULL_HANDLE;
        /// A3(a): secondary-bounce GGX roughness regularization (Theia-only bias;
        /// Hyperion stays unbiased). Off → estimator identical to before.
        bool useA3Regularization = true;
        /// c1: variance-guided adaptive sampling. Upper bound on GI samples per pixel; the
        /// per-pixel count ramps from 1 up to this with the A-SVGF variance guide. Averaging
        /// independent samples keeps the estimator unbiased. 1 disables adaptive sampling.
        /// Only takes effect when gradientVarianceView is bound; otherwise the shader forces 1
        /// (bit-identical to the legacy single-sample path).
        std::uint32_t adaptiveMaxSamples = 4;

        /// A4: ReSTIR DI — spatiotemporal reservoir resampling for emissive-triangle
        /// direct lighting at the primary vertex. When true the shader replaces the
        /// forward pass's emissive-derived rect-light term (the forward pass must be
        /// told to skip those lights to avoid double-counting). When false the reservoir
        /// buffers are untouched and behaviour is bit-identical to the pre-A4 path.
        /// Mutually exclusive with useRestirPt: when PT is on, DI is forced off.
        bool useRestirDi = true;
        /// GI2: ReSTIR PT Enhanced — unified DI+GI path reservoir (default on). When
        /// true the primary vertex's emissive-triangle NEE runs through the shared path
        /// integrator (the same estimator Hyperion uses → bit-identical in expectation).
        /// The forward pass still skips its emissive-derived rect lights (shared with
        /// DI's semantic). Temporal/spatial reuse of path reservoirs is layered on top
        /// (Phase 2+); the Phase 1 stub is unbiased but high-variance.
        bool useRestirPt = true;
        /// GI2 full PT: multi-bounce path reservoir for the INDIRECT term (GRIS
        /// random-replay shift). Replaces the per-sample estimateGiSample multi-bounce
        /// walk with RIS over replayable path candidates (local + temporal + spatial).
        /// Only meaningful with useRestirPt; the application gates it accordingly.
        bool useRestirPtPath = true;
        /// Two-tier output contract (extended): the firefly clamps are a
        /// presentation-stability aid for the interactive window. In an offscreen
        /// capture (--output) they are DISABLED so the capture is the unclamped
        /// scene-referred estimator result (parity with Hyperion, which has no
        /// render-path clamp by guardrail).
        bool fireflyClampEnabled = true;
        /// A4: screen-space motion vectors (R32G32F, pixel-space dx/dy) for temporal
        /// reprojection. VK_NULL_HANDLE → a 1×1 zero dummy is bound (static-history
        /// reuse: previous reservoir read at the same pixel).
        /// GI-ENH (a): reconnection shift feature mask (THEIA_NO_RECONNECTION=0 /
        /// THEIA_SHIFT_MASK=<n>). bit0 = k=2 shift, bit1 = k>=3 hybrid shift.
        /// 0 = replay-only (the clean baseline).
        std::uint32_t reconnectShiftEnabled = 3;
        VkImageView motionVectorView = VK_NULL_HANDLE;
    };

    GiPass() = default;
    ~GiPass();
    GiPass(const GiPass&) = delete;
    GiPass& operator=(const GiPass&) = delete;

    [[nodiscard]] bool initialize(const harmonia::DeviceContext& ctx,
                                  const Config& cfg,
                                  const harmonia::core::CommandPool& uploadPool,
                                  const char* giSpv = "gi.comp.spv",
                                  const char* generateSpv = "wavefront_generate.comp.spv",
                                  const char* traceSpv = "wavefront_trace.comp.spv");
    void shutdown();

    /// Dispatch the GI accumulation pass. Call AFTER ForwardRenderer::recordFrame()
    /// in the same command buffer. Transitions:
    ///   giBuffer / gbuffer : ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    ///   hdr                : ATTACHMENT_OPTIMAL -> GENERAL (stays GENERAL after this call)
    /// @param skipPreBarriers When true the caller has already issued the layout
    ///        transitions (e.g. via queue-family acquire barriers on an async compute
    ///        command buffer). The dispatch still runs; only the pre-barriers are skipped.
    void record(VkCommandBuffer cmd, const FrameParams& params, bool skipPreBarriers = false);

    [[nodiscard]] bool isInitialized() const noexcept { return m_pipeline != VK_NULL_HANDLE; }

  private:
    struct GiPushConstants {
        sm::float4x4 view{1.0f};
        sm::float4x4 prevViewProj{1.0f}; ///< GI2: prev-frame VP for inline temporal reprojection
        sm::float4 cameraPos{0.0f};
        float exposure = 1.0f;
        std::uint32_t frameSampleIndex = 0;
        std::uint32_t rngBaseSeed = 0;
        std::uint32_t emissiveTriangleCount = 0;
        std::uint32_t envImportanceWidth = 0;
        std::uint32_t envImportanceHeight = 0;
        std::uint32_t hasEnvMap = 0;
        float envLuminanceScale = 1.0f;
        std::uint32_t maxDepth = 3;
        std::uint32_t screenWidth = 0;
        std::uint32_t screenHeight = 0;
        std::uint32_t _pad0 = 0;
        std::uint32_t a3RegularizationEnabled = 1; ///< A3(a): secondary-bounce roughness regularization
        std::uint32_t hasGradientVariance = 0;     ///< A3(b): 1 when a real A-SVGF guide is bound
        std::uint32_t giAdaptiveMaxSamples =
            1; ///< c1: max GI samples/pixel (1 = adaptive sampling off / behavior-preserving)
        std::uint32_t restirDiEnabled = 0; ///< A4: 1 = spatiotemporal reservoir DI; 0 = bit-identical fallback
        std::uint32_t restirDiSpatial = 0; ///< A4: 1 = enable spatial reuse (default 0 = initial+temporal only)
        std::uint32_t restirHasMotion = 0; ///< A4: 1 = real motion image bound; 0 = dummy (skip reprojection)
        std::uint32_t restirPtEnabled =
            0; ///< GI2: 1 = ReSTIR PT (path integrator owns primary emissive NEE); 0 = legacy DI path
        std::uint32_t restirPtPathEnabled = 0; ///< GI2 full PT: 1 = multi-bounce path reservoir owns the indirect walk
        std::uint32_t fireflyClampEnabled = 1; ///< 1 = presentation clamp (default); 0 = estimator-pure capture
        std::uint32_t pairingPerm0 = 0;        ///< GI-ENH pairing-texture 0 per-frame transform (19 packed bits)
        std::uint32_t pairingPerm1 = 0;
        std::uint32_t pairingPerm2 = 0;
        std::uint32_t pairingEnabled = 0;        ///< 1 = Gaussian paired neighbours in path-reservoir spatial reuse
        std::uint32_t reconnectShiftEnabled = 1; ///< GI-ENH (a): reconnection shift feature mask
                                                 ///< bit0 = k=2, bit1 = k>=3. 0 = replay-only.
        std::uint32_t waveIndex = 0;             ///< PERF5 10d: current wave (0..waveCount-1)
        std::uint32_t waveCount = 1;             ///< PERF5 10d: dispatched wave count — 1 when the
                                                 ///< ReSTIR PT path reservoir owns ALL multi-bounce
                                                 ///< transport (per-sample slots are provably dead
                                                 ///< after wave 0); maxDepth + 1 for the legacy
                                                 ///< per-sample walks
    };
    // Layout guard: must equal the Slang `GiParams` struct exactly (2×float4x4 + float4
    // + 28 scalars = 144 + 112 = 256). Fails loudly on any accidental padding mismatch.
    static_assert(sizeof(GiPushConstants) == 256);

    [[nodiscard]] bool createDescriptors();
    [[nodiscard]] bool createPipelines(const char* giSpv, const char* generateSpv, const char* traceSpv);
    void updateDescriptors(const FrameParams& params);
    void
    updateRestirDescriptors(const FrameParams& params,
                            std::uint32_t slot); ///< A4: per-frame bindings 16/17/18 (ping-pong + motion), per-slot set
    [[nodiscard]] bool descriptorsDirty(const FrameParams& params) const;

    const harmonia::DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    /// One descriptor buffer writer PER FRAME SLOT (== harmonia::app::FrameSync::kFrameSlots):
    /// frames run pipelined, so a single mutable descriptor buffer would let frame N+1's
    /// per-frame Cur/Prev rebinding race frame N's in-flight dispatch (the run-to-run
    /// nondeterminism root-caused 2026-09). Each slot has its own descriptor buffer.
    static constexpr std::uint32_t kDescriptorSlots = 2;

    harmonia::core::UniqueDescriptorSetLayout m_setLayout;
    /// MOD1: descriptor buffer writers replace the pool + allocated sets (ping-pong slots).
    std::array<harmonia::core::DescriptorBufferWriter, kDescriptorSlots> m_descWriters{};
    std::uint32_t m_activeSlot = 0;

    harmonia::core::UniquePipelineLayout m_pipelineLayout;
    harmonia::core::UniquePipeline m_pipeline;
    /// PERF5 10b: generate pass (camera-ray enqueue + collect init), dispatched
    /// ahead of the GI dispatch every frame. Shares m_pipelineLayout (same
    /// descriptor set + push-constant block as the GI shader).
    harmonia::core::UniquePipeline m_generatePipeline;
    /// PERF5 10c: trace pass (the verbatim traceClosestHit RayQuery traversal over
    /// the wave-0 records → WaveHit buffer), dispatched between generate and GI.
    /// Shares m_pipelineLayout.
    harmonia::core::UniquePipeline m_tracePipeline;

    bool m_hdrFirstUse = true;
    const Scene* m_boundScene = nullptr;
    const Scene* m_texturesBoundFor = nullptr;
    VkImageView m_boundEnvMapView = VK_NULL_HANDLE;
    VkSampler m_boundEnvSampler = VK_NULL_HANDLE;
    const harmonia::core::Buffer* m_boundEnvMarginalCdf = nullptr;
    const harmonia::core::Buffer* m_boundEnvConditionalCdf = nullptr;
    VkImageView m_boundGradientVarianceView = VK_NULL_HANDLE;

    /// A3(b): 1×1 R32G32F placeholder bound to binding 15 when no A-SVGF
    /// gradient/variance guide is provided; hasGradientVariance=0 ensures the
    /// shader never reads it.

    // A4: ReSTIR DI reservoir ping-pong buffers (binding 16 = current write, 17 = prev
    // read) and a 1×1 zero motion-vector placeholder (binding 18). The reservoir stride
    // is allocated generously (kReservoirStride) so it covers whichever std430-ish struct
    // layout Slang picks — the CPU never indexes the contents.
    static constexpr VkDeviceSize kReservoirStride = 64; ///< >= Slang Reservoir stride (float3 may be 16-aligned)
    harmonia::core::Buffer m_reservoirBuf[2]{};
    std::uint32_t m_reservoirPingPong = 0; ///< index of the buffer written THIS frame
    bool m_reservoirsCleared = false;      ///< one-time zero-fill of both reservoir buffers
    /// GI2 full PT: path reservoir ping-pong buffers (bindings 20 = cur, 21 = prev).
    /// 384 B covers the Slang PathReservoir (40 B core + the GI-ENH (a) reconnection record:
    /// ~288 B with Slang's 16-byte float3 alignment — the CPU never indexes the contents, so
    /// generous over-allocation is the policy; too small = out-of-bounds struct reads).
    static constexpr VkDeviceSize kPathReservoirStride = 512; // GI-ENH k≥3: ReconnectionCapture grew ~64B
    harmonia::core::Buffer m_pathReservoirBuf[2]{};
    std::uint32_t m_pathReservoirPingPong = 0;
    bool m_pathReservoirsCleared = false;
    /// GI-ENH: Gaussian paired-neighbor pairing maps (binding 22, static after init).
    harmonia::core::Buffer m_pairingOffsetBuf{};

    /// PERF5 10b: wavefront path buffers (PLAN.md "PERF5 wavefront design" §2).
    /// Exact Slang std430 strides (float3 = 16 B align): WavePath 160, WaveHit 32,
    /// WaveCollect 176 (10d: the collect carries BOTH per-sample composite chains —
    /// indirect and primary env NEE — see wave_common.slang; was 112 B) — sized to
    /// the render resolution × kWaveSlotsPerPixel (recreated on resize via
    /// initialize()). The CPU never indexes the contents.
    static constexpr VkDeviceSize kWavePathStride = 160;
    static constexpr VkDeviceSize kWaveHitStride = 32;
    static constexpr VkDeviceSize kWaveCollectStride = 176;
    /// Fixed per-pixel slot window: kinds 0..3 = adaptive samples, 4 = path-reservoir
    /// local candidate, 5..7 = spatial-replay candidates (enqueue is a later stage;
    /// the generate pass initializes all eight slots every frame — Q2 — so every
    /// first read is defined). Slot = pixelIndex * kWaveSlotsPerPixel + kind (no
    /// allocation atomics — deterministic index space; the records' alive bit is
    /// the truth, buffers are never cleared).
    static constexpr std::uint32_t kWaveSlotsPerPixel = 8;
    /// Ray buffer ×2 per the design's byte math. 10d: the regenerate is IN-PLACE
    /// (both ray bindings alias m_wavePathBuf[0] — each shade thread reads and
    /// writes only its own pixel's slots, so the wave barriers order everything;
    /// a physical ping-pong would force every fixed-width wave to re-write every
    /// dead slot's 160-B record, which dominated the frame on unified memory).
    /// m_wavePathBuf[1] stays allocated as the design's second ray half — the
    /// 10e stream-compaction scratch. Wave 0 is fully rewritten by the generate
    /// pass every frame; the buffers are never cleared (the alive bit is the
    /// truth — dead records persist in their slots).
    harmonia::core::Buffer m_wavePathBuf[2]{};
    std::uint32_t m_wavePathPingPong = 0;      ///< 10d: inert bookkeeping (bindings alias buf[0]);
                                               ///< retained for the 10e compaction half
    harmonia::core::Buffer m_waveHitBuf{};     ///< trace-pass output (10c) — written by each wave's
                                               ///< trace dispatch, read by that wave's shade dispatch
    harmonia::core::Buffer m_waveCollectBuf{}; ///< per-pixel collect: generate-initialized, accumulated
                                               ///< by the shade waves, composited by the last wave (10d)

    VkImageView m_boundMotionVectorView = VK_NULL_HANDLE;
};

} // namespace theia
#endif // THEIA_RENDERER_GIPASS_HPP
