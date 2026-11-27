#ifndef THEIA_RENDERER_GIPASS_HPP
#define THEIA_RENDERER_GIPASS_HPP

#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/DescriptorBufferWriter.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/pipeline/SmsPass.hpp"
#include "theia/renderer/RendererConstants.hpp"

namespace theia {

class Scene;

/// Ray-query global-illumination compute stage.
///
/// Runs one compute dispatch after the forward geometry pass. For every primary
/// hit recorded in the forward G-buffer it spawns a BSDF continuation ray and walks
/// 1..maxDepth path segments through the scene TLAS (inline VK_KHR_ray_query),
/// evaluating the shared Harmonia `integrateSurface` estimator at each secondary
/// vertex. The resulting indirect (bounce) radiance is additively composited into
/// the HDR render target — RT-GI is the renderer's single indirect provider (there
/// is no split-sum IBL fallback path).
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
        bool smsEnabled = true;
        std::uint32_t smsPartitionResolution = 16;
        bool smsPriorEnabled = true;
        std::uint32_t smsTrialBudget = 32; ///< Per-dispatch budget, never a total reciprocal-trial cap.
        bool smsSpatialReuse = false; ///< Opt-in (--sms-mode reuse): qualified C1/C2/C3 + 2d.2 controls; default local.
    };

    /// Per-frame scene + camera state consumed by record(). The scene bindings
    /// (material/geometry buffers, env CDF) are written into every descriptor slot
    /// when the scene or the environment changes; the TLAS and the reservoirs are
    /// written into the slot of each frame, like ForwardRenderer's descriptor update.
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
        /// Distance haze of the scene ([haze]): extinction per metre at height zero
        /// and the scale height in metres. 0 = none.
        float hazeDensity = 0.0f;
        float hazeHeight = 0.0f;

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
        /// GI-ENH bias correction (THEIA_BIAS_CORRECTION): 0 = RTXDI-OFF alias of the
        /// default formula, 1 = default RTXDI-OFF-form `w_sum/(p_hat*M)`, 2 = that
        /// formula plus per-source conservative visibility in the DI merges.
        std::uint32_t biasCorrectionMode = 1;
        VkImageView motionVectorView = VK_NULL_HANDLE;
        /// PERF8 (RTXPT whitepaper §7, material LOD by ray class): reduced-BRDF +
        /// conservative-MIP evaluation for diffuse-secondary visits (full material
        /// for primary/near-specular/delta-spawned rays; visibility rays are
        /// opacity-only by architecture). Two-tier like the firefly clamps: ON for
        /// the interactive window, DISABLED in estimator-pure capture (--output) so
        /// the capture converges to the unmodified ground truth (§5.6 litmus —
        /// irreducible bias lives in the real-time-only layer only).
        /// THEIA_MATERIAL_LOD=0|1 forces the mode (debug A/B harness).
        bool materialLodEnabled = true;
        /// PERF8: tan(fovY/2) of the current projection — seeds the primary pixel
        /// world footprint (meters) the ray cone starts from.
        float tanHalfFovY = 0.5f;
        /// GI-NRC (world-space radiance cache): two-tier like the clamps/LOD — ON for
        /// the interactive window, DISABLED in estimator-pure capture (biased by
        /// construction, §5.6/§8 litmus). THEIA_RADIANCE_CACHE=0|1 forces the mode.
        /// Also forced off for animated scenes (world-static table); SMS frames use the
        /// per-surface smsPhysicalEvent exclusion instead of a blanket gate.
        bool radianceCacheEnabled = true;
    };

    GiPass() = default;
    ~GiPass();
    GiPass(const GiPass&) = delete;
    GiPass& operator=(const GiPass&) = delete;

    [[nodiscard]] bool initialize(const harmonia::DeviceContext& ctx,
                                  const Config& cfg,
                                  const harmonia::core::CommandPool& uploadPool,
                                  const char* giSpv = "gi.comp.spv");
    void shutdown();

    /// Dispatch the GI accumulation pass. Call AFTER ForwardRenderer::recordFrame()
    /// in the same command buffer. Transitions:
    ///   giBuffer / gbuffer : ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    ///   hdr                : ATTACHMENT_OPTIMAL -> GENERAL (stays GENERAL after this call)
    /// @param skipPreBarriers When true the caller has already issued the layout
    ///        transitions (e.g. via queue-family acquire barriers on an async compute
    ///        command buffer). The dispatch still runs; only the pre-barriers are skipped.
    void record(VkCommandBuffer cmd, const FrameParams& params, bool skipPreBarriers = false);

    [[nodiscard]] bool isInitialized() const noexcept {
        return m_pipeline != VK_NULL_HANDLE && m_frameError == VK_SUCCESS;
    }
    [[nodiscard]] bool usesSms(std::uint32_t emitterCount, std::uint32_t depth) const noexcept {
        return m_cfg.smsEnabled && emitterCount != 0 && depth >= 3;
    }
    [[nodiscard]] bool completionRequired() const noexcept {
        return m_frameError != VK_SUCCESS || (m_frameRecorded && (m_smsFrameActive || m_smsDiagEnabled));
    }
    [[nodiscard]] std::expected<bool, VkResult> frameReady() const noexcept;
    [[nodiscard]] VkResult recordContinuation(VkCommandBuffer cmd) noexcept;

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
        float tanHalfFovY = 0.5f;                  ///< PERF8: tan(fovY/2) — primary pixel-footprint seed (was _pad0)
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
        std::uint32_t reconnectShiftEnabled = 1; ///< GI-ENH (a): reconnection shift feature mask
                                                 ///< bit0 = k=2, bit1 = k>=3. 0 = replay-only.
        std::uint32_t biasCorrectionMode = 1;    ///< GI-ENH: 0 = RTXDI-OFF alias of the
                                                 ///< default, 1 = default RTXDI-OFF-form,
                                                 ///< 2 = + source conservative visibility
        std::uint32_t smsEnabled = 0;
        std::uint32_t materialLodEnabled = 1;   ///< PERF8: 1 = material LOD tier active (interactive);
                                                ///< 0 = estimator-pure capture (full material everywhere)
        std::uint32_t radianceCacheEnabled = 1; ///< GI-NRC: 1 = world-space radiance-cache tier
                                                ///< active (interactive); 0 = estimator-pure capture
        float hazeDensity = 0.0F;               ///< [haze] extinction per metre at height zero; 0 = no haze
        float hazeHeight = 0.0F;                ///< [haze] metres at which the extinction has fallen to 1/e
    };
    // Layout guard: must equal the Slang `GiPC` struct exactly (2×float4x4 + float4
    // + 28 scalars = 144 + 112 = 256). Stays within the existing 256-byte device budget.
    static_assert(sizeof(GiPushConstants) == 256);

    [[nodiscard]] bool createDescriptors();
    [[nodiscard]] bool createPipelineLayout();
    [[nodiscard]] bool createComputePipeline(const char* spv,
                                             const char* entry,
                                             const char* cacheName,
                                             harmonia::core::UniquePipeline& out);
    [[nodiscard]] bool createSmsResources(const harmonia::core::CommandPool& uploadPool);
    void reportSmsDiagnostics() const noexcept;
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
    /// GI-ENH §6.2.2: uniform replay + per-pixel combine passes over the same
    /// descriptor set / push-constant layout as the emit megakernel.
    harmonia::core::UniquePipeline m_replayPipeline;
    harmonia::core::UniquePipeline m_combinePipeline;
    harmonia::pipeline::SmsPass m_sms;
    harmonia::core::Buffer m_smsContext, m_smsPrefixes, m_smsPrior, m_smsDiagnostics;
    harmonia::core::Buffer m_smsTemporalReservoir, m_smsHistoryReservoir;
    harmonia::core::Buffer m_smsTemporalStaging, m_smsHistoryStaging;
    bool m_smsReservoirsCleared = false;
    bool m_smsDiagEnabled = false;
    mutable bool m_smsDiagnosticsReported = false;
    mutable VkResult m_frameError = VK_SUCCESS;
    bool m_frameRecorded = false, m_smsFrameActive = false;

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
    /// GI-NRC: world-space radiance-cache table (binding 31, 2 MiB, persistent
    /// across frames — zero-filled once with the reservoir one-time clear).
    harmonia::core::Buffer m_radCacheBuf{};
    /// GI-CG: the former Gaussian pairing maps (binding 22) were removed 2026-11-15 —
    /// the CGNS compatibility score supersedes their neighbor-selection role.

    // ── GI-NRC: world-space radiance-cache table (binding 31) ─────────────────
    // MUST match the shader's kRadCacheSlots (gi.comp.slang): 2^17 hashed world-cell
    // records x {float3 rgb, uint stamp} = 2 MiB, persistent across frames.
    static constexpr std::uint32_t kRadCacheSlots = 1u << 17;

    // ── GI-ENH §6.2.2 replay stream compaction (bindings 27-30) ──────────────
    // Three-kernel split: emit (main) → uniform replay (replayMain) → combine
    // (combineMain). Buffers are frame-transient; the count buffer doubles as
    // the vkCmdDispatchIndirect command (words 0-2 = x/y/z, word 4 = record
    // counter, reset by a per-frame fill).
    static constexpr VkDeviceSize kPtParamsStride = 128;    // PtEmitParams (8 × 16 B)
    static constexpr VkDeviceSize kReplayResultStride = 16; // float4 per (pixel, slot)
    static constexpr std::uint32_t kReplayCountBytes = 20;  // 4 command words + counter
    /// ONE INSTANCE PER FRAME SLOT (kDescriptorSlots): frames run pipelined, so a
    /// single instance would let frame N+1's emit kernel overwrite the params/
    /// records/results that frame N's replay/combine kernels are still reading —
    /// the same cross-frame race the per-slot descriptor buffers fix.
    harmonia::core::Buffer m_replayRecordsBuf[2]{}; // pixelSlots × uint
    harmonia::core::Buffer m_replayCountBuf[2]{};   // 20 B: indirect command + counter
    harmonia::core::Buffer m_ptParamsBuf[2]{};      // pixels × PtEmitParams
    harmonia::core::Buffer m_replayResultsBuf[2]{}; // pixelSlots × float4

    VkImageView m_boundMotionVectorView = VK_NULL_HANDLE;
};

} // namespace theia
#endif // THEIA_RENDERER_GIPASS_HPP
