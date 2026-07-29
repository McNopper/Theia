#ifndef THEIA_RENDERER_GIPASS_HPP
#define THEIA_RENDERER_GIPASS_HPP

#include <volk/volk.h>

#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"

class Scene;

namespace theia {

/// Ray-query global-illumination compute stage.
///
/// Runs one compute dispatch after the forward geometry pass. For every primary
/// hit recorded in the forward G-buffer it spawns a BSDF continuation ray and walks
/// 1..maxDepth path segments through the scene TLAS (inline VK_KHR_ray_query),
/// evaluating the shared Harmonia `integrateSurface` estimator at each secondary
/// vertex. The resulting indirect (bounce) radiance is additively composited into
/// the HDR render target, replacing the forward pass's flat-IBL approximation.
///
/// Toggleable and distinct from the forward raster stage: when active, the forward
/// pass emits direct + emission only (giEnabled push-constant disables IBL/ambient).
class GiPass {
  public:
    static constexpr uint32_t kMaxBindlessTextures = 256;

    struct Config {
        uint32_t width = 0;
        uint32_t height = 0;
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
        VkBuffer envMarginalCdf = VK_NULL_HANDLE;    ///< may be VK_NULL_HANDLE (no env importance)
        VkBuffer envConditionalCdf = VK_NULL_HANDLE; ///< may be VK_NULL_HANDLE
        uint32_t envImportanceWidth = 0;
        uint32_t envImportanceHeight = 0;
        bool hasEnvMap = false;
        float envLuminanceScale = 1.0f;

        sm::float4x4 view{1.0f}; ///< view matrix
        sm::float4x4 prevViewProj{
            1.0f}; ///< GI2: previous-frame view-projection for inline motion-vector reprojection (temporal reuse)
        sm::float3 cameraPos{0.0f};
        float exposure = 1.0f;
        uint32_t frameSampleIndex = 0;
        uint32_t rngBaseSeed = 0;
        uint32_t maxDepth = 3;

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
        uint32_t adaptiveMaxSamples = 4;

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
        /// A4: enable the (optional) spatial reuse follow-on. Default off — initial
        /// candidates + temporal reuse only, which avoids the single-pass spatial race.
        bool useRestirDiSpatial = false;
        /// A4: screen-space motion vectors (R32G32F, pixel-space dx/dy) for temporal
        /// reprojection. VK_NULL_HANDLE → a 1×1 zero dummy is bound (static-history
        /// reuse: previous reservoir read at the same pixel).
        VkImageView motionVectorView = VK_NULL_HANDLE;
    };

    GiPass() = default;
    ~GiPass();
    GiPass(const GiPass&) = delete;
    GiPass& operator=(const GiPass&) = delete;

    [[nodiscard]] bool initialize(const DeviceContext& ctx, const Config& cfg, const char* giSpv = "gi.comp.spv");
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
        uint32_t frameSampleIndex = 0;
        uint32_t rngBaseSeed = 0;
        uint32_t emissiveTriangleCount = 0;
        uint32_t envImportanceWidth = 0;
        uint32_t envImportanceHeight = 0;
        uint32_t hasEnvMap = 0;
        float envLuminanceScale = 1.0f;
        uint32_t maxDepth = 3;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t _pad0 = 0;
        uint32_t a3RegularizationEnabled = 1; ///< A3(a): secondary-bounce roughness regularization
        uint32_t hasGradientVariance = 0;     ///< A3(b): 1 when a real A-SVGF guide is bound
        uint32_t giAdaptiveMaxSamples =
            1;                        ///< c1: max GI samples/pixel (1 = adaptive sampling off / behavior-preserving)
        uint32_t restirDiEnabled = 0; ///< A4: 1 = spatiotemporal reservoir DI; 0 = bit-identical fallback
        uint32_t restirDiSpatial = 0; ///< A4: 1 = enable spatial reuse (default 0 = initial+temporal only)
        uint32_t restirHasMotion = 0; ///< A4: 1 = real motion image bound; 0 = dummy (skip reprojection)
        uint32_t restirPtEnabled =
            0; ///< GI2: 1 = ReSTIR PT (path integrator owns primary emissive NEE); 0 = legacy DI path
        uint32_t restirPtPathEnabled = 0; ///< GI2 full PT: 1 = multi-bounce path reservoir owns the indirect walk
    };
    static_assert(sizeof(GiPushConstants) == 224);

    [[nodiscard]] bool createDescriptors();
    [[nodiscard]] bool createPipeline(const char* giSpv);
    void updateDescriptors(const FrameParams& params);
    void updateRestirDescriptors(const FrameParams& params); ///< A4: per-frame bindings 16/17/18 (ping-pong + motion)
    [[nodiscard]] bool descriptorsDirty(const FrameParams& params) const;

    const DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    bool m_hdrFirstUse = true;
    const Scene* m_boundScene = nullptr;
    const Scene* m_texturesBoundFor = nullptr;
    VkImageView m_boundEnvMapView = VK_NULL_HANDLE;
    VkSampler m_boundEnvSampler = VK_NULL_HANDLE;
    VkBuffer m_boundEnvMarginalCdf = VK_NULL_HANDLE;
    VkBuffer m_boundEnvConditionalCdf = VK_NULL_HANDLE;
    VkImageView m_boundGradientVarianceView = VK_NULL_HANDLE;

    /// A3(b): 1×1 R32G32F placeholder bound to binding 15 when no A-SVGF
    /// gradient/variance guide is provided; hasGradientVariance=0 ensures the
    /// shader never reads it.
    Image m_dummyGradientVariance{};
    bool m_dummyGradientReady = false; ///< one-time UNDEFINED → GENERAL transition done

    // A4: ReSTIR DI reservoir ping-pong buffers (binding 16 = current write, 17 = prev
    // read) and a 1×1 zero motion-vector placeholder (binding 18). The reservoir stride
    // is allocated generously (kReservoirStride) so it covers whichever std430-ish struct
    // layout Slang picks — the CPU never indexes the contents.
    static constexpr VkDeviceSize kReservoirStride = 64; ///< >= Slang Reservoir stride (float3 may be 16-aligned)
    Buffer m_reservoirBuf[2]{};
    uint32_t m_reservoirPingPong = 0; ///< index of the buffer written THIS frame
    bool m_reservoirsCleared = false; ///< one-time zero-fill of both reservoir buffers
    /// GI2 full PT: path reservoir ping-pong buffers (bindings 20 = cur, 21 = prev).
    static constexpr VkDeviceSize kPathReservoirStride = 64; ///< >= Slang PathReservoir stride
    Buffer m_pathReservoirBuf[2]{};
    uint32_t m_pathReservoirPingPong = 0;
    bool m_pathReservoirsCleared = false;
    Image m_dummyMotionVectors{}; ///< 1×1 R32G32F zero placeholder for binding 18
    bool m_dummyMotionReady = false;
    VkImageView m_boundMotionVectorView = VK_NULL_HANDLE;
};

} // namespace theia
#endif // THEIA_RENDERER_GIPASS_HPP
