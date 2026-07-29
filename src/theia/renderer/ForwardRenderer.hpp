#ifndef THEIA_RENDERER_FORWARDRENDERER_HPP
#define THEIA_RENDERER_FORWARDRENDERER_HPP

#include <volk/volk.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "theia/renderer/GpuCullPass.hpp"
#include "theia/renderer/HiZPass.hpp"
#include "theia/renderer/IblPrecompute.hpp"

class Scene;

namespace theia {

/// GPU-driven forward renderer using Vulkan 1.4 features.
/// - Dynamic rendering (no render passes)
/// - Mesh shaders (VK_EXT_mesh_shader) for GPU-driven geometry
/// - Depth pre-pass ready (depth target owned here)
///
/// The HDR color target is owned by Application (same as Hyperion''s hdrImage)
/// and passed in via Config so ToneMapper can read it as a storage image.
class ForwardRenderer {
  public:
    struct Config {
        std::uint32_t width = 1024;
        std::uint32_t height = 768;
        VkFormat outputFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkImage hdrImage = VK_NULL_HANDLE; ///< externally owned HDR color target
        VkImageView hdrImageView = VK_NULL_HANDLE;
        VkImageView iblDiffuseView = VK_NULL_HANDLE;
        VkImageView iblSpecularView = VK_NULL_HANDLE;
        VkImageView sheenLutView = VK_NULL_HANDLE;
        VkSampler iblEnvSampler = VK_NULL_HANDLE;
        VkSampler iblLutSampler = VK_NULL_HANDLE;
    };

    /// Camera parameters. Uses the shared harmonia::Camera::Params so both
    /// Hyperion and Theia describe their cameras with the same type.
    /// ForwardRenderer reads: position, target, up, vfovDeg, nearPlane,
    /// farPlane, and physical.ev100() for exposure.
    using CameraParams = Camera::Params;

    ForwardRenderer() = default;
    ~ForwardRenderer();

    bool initialize(const DeviceContext& ctx, const Config& config);
    void shutdown();

    /// Resize to a new extent after the host (App) has recreated the HDR image.
    /// Recreates depth/GBuffer targets; the pipeline and descriptors are unchanged.
    bool resize(std::uint32_t width, std::uint32_t height, VkImage hdrImage, VkImageView hdrImageView) noexcept;

    void setScene(const Scene* scene) {
        m_scene = scene;
        m_texturesBoundFor = nullptr;
    }
    void setCamera(const CameraParams& cam) { m_camera = cam; }
    /// Bind IBL textures (set 2 / sky set 0). When a raw environment panorama is
    /// available it is bound to IBL binding 4 for the sky background and its
    /// physical scale (env_unit_nits) recorded; pass VK_NULL_HANDLE / 1.0f when the
    /// scene has no env map (binding 4 falls back to the specular view as a
    /// harmless placeholder).
    void setIbl(const IblResources& res, VkImageView rawEnvView = VK_NULL_HANDLE, float envUnitNits = 1.0f);
    /// Bind env-importance CDF buffers used by transparent-path stochastic env sampling.
    void setEnvImportanceSampling(VkBuffer marginalCdf,
                                  VkBuffer conditionalCdf,
                                  std::uint32_t width,
                                  std::uint32_t height) noexcept {
        m_envMarginalCdf = marginalCdf;
        m_envConditionalCdf = conditionalCdf;
        m_envImportanceWidth = width;
        m_envImportanceHeight = height;
    }

    /// Whether the current scene has an environment map (draws env sky vs black).
    void setHasEnvironment(bool hasEnv) noexcept { m_hasEnv = hasEnv; }

    /// Configure ray-traced directional ("sun") shadows from the dominant IBL light.
    /// dirToSun is the world-space direction pointing toward the sun; strength in [0,1]
    /// scales how dark the cast shadow is (0 disables sun shadows entirely).
    void setSunShadow(const sm::float3& dirToSun, float strength) noexcept {
        m_sunDir = dirToSun;
        m_sunStrength = strength;
    }
    /// Presentation-only constant indirect ambient term (default off for parity).
    void setIndirectAmbient(float strength) noexcept { m_indirectAmbientStrength = std::max(0.0f, strength); }
    /// When the ray-query GI compute stage (GiPass) is active it supplies indirect
    /// lighting, so the forward pass must skip its flat-IBL / ambient approximation.
    /// Encoded as bit 0 of the `giEnabled` push constant (bit 1 = ReSTIR DI active).
    void setGiEnabled(bool enabled) noexcept { m_giEnabled = (m_giEnabled & ~1U) | (enabled ? 1U : 0U); }
    /// A4: when ReSTIR DI owns emissive-triangle direct lighting (in GiPass), the forward
    /// pass must skip its emissive-derived rect/area lights to avoid double-counting.
    /// Encoded as bit 1 of the `giEnabled` push constant. Punctual lights are unaffected.
    void setRestirDiActive(bool active) noexcept { m_giEnabled = (m_giEnabled & ~2U) | (active ? 2U : 0U); }
    /// Max transparent gather depth for coverage/transmission rays (scene max_depth).
    void setTransparentMaxDepth(std::uint32_t depth) noexcept { m_transparentMaxDepth = std::max(1u, depth); }
    /// Per-frame RNG state plumbed into shader push constants.
    void setRngState(std::uint32_t frameSampleIndex, std::uint32_t baseSeed, bool deterministicReplay) noexcept {
        m_frameSampleIndex = frameSampleIndex;
        m_rngBaseSeed = baseSeed;
        m_deterministicReplay = deterministicReplay;
    }
    void setCameraJitterEnabled(bool enabled) noexcept { m_cameraJitterEnabled = enabled; }
    void setRngDebug(bool enabled) noexcept { m_rngDebug = enabled ? 1U : 0U; }

    /// Enable/disable the Hi-Z occlusion test for the NEXT frame. Disabled on a camera cut
    /// (large motion) so newly disoccluded geometry is drawn conservatively.
    void setHiZTestEnabled(bool enabled) noexcept { m_hiZTestEnabled = enabled; }

    /// Update tile light list buffers (called by LightCuller each frame before recordFrame).
    void
    setTileBuffers(VkBuffer tileLightCounts, VkBuffer tileLightIndices, std::uint32_t tilesX, std::uint32_t tilesY);

    /// Record scene geometry rendering into cmd.
    /// Transitions hdrImage UNDEFINED/GENERAL -> ATTACHMENT_OPTIMAL, renders, leaves it there.
    /// Application::mainLoop() will barrier it to GENERAL before the ToneMapper.
    void recordFrame(VkCommandBuffer cmd);

    // Thin GBuffer (RGBA16F: xyz=view-space normal [0,1], w=roughness) for SSR.
    [[nodiscard]] VkImageView gbufferView() const noexcept { return m_gbufferTarget.view(); }
    [[nodiscard]] VkImage gbufferImage() const noexcept { return m_gbufferTarget.handle(); }
    // GI GBuffer (RGBA32F: xyz=world-space primary hit position, w=asfloat(materialIdx+1)) for GiPass.
    [[nodiscard]] VkImageView giBufferView() const noexcept { return m_giBufferTarget.view(); }
    [[nodiscard]] VkImage giBufferImage() const noexcept { return m_giBufferTarget.handle(); }
    [[nodiscard]] VkImageView depthView() const noexcept { return m_depthTarget.view(); }
    [[nodiscard]] VkImage depthImage() const noexcept { return m_depthTarget.handle(); }

  private:
    struct alignas(16) MeshPushConstants {
        sm::float4x4 viewProj; ///< row-major — mul(pos, viewProj) = VP * pos directly
        sm::float4x4 view;
        sm::float4 cameraPos;  ///< xyz = world-space camera position
        float exposure = 0.0f; ///< 1 / (1.2 * 2^EV100) — same as Hyperion
        std::uint32_t lightCount = 0;
        std::uint32_t emissiveTriangleCount = 0;
        std::uint32_t tilesX = 0; ///< screen width / 16
        std::uint32_t tilesY = 0; ///< screen height / 16
        std::uint32_t screenWidth = 0;
        std::uint32_t screenHeight = 0;
        std::uint32_t transparentMaxDepth = 2; ///< transparent gather depth
        std::uint32_t frameSampleIndex = 0;    ///< per-frame sample counter for stochastic stages
        std::uint32_t rngBaseSeed = 0;         ///< base seed for composeRngSeed(pixel, frame, bounce, seed)
        std::uint32_t rngFlags = 0;            ///< bit0 = deterministic replay, bit1 = RNG debug view
        std::uint32_t cullPhase = 0;           ///< Hi-Z pass: 0 = draw all, 1 = prev-visible, 2 = remaining + Hi-Z
        std::uint32_t envImportanceWidth = 0;  ///< CDF width; 0 disables env importance sampling
        std::uint32_t envImportanceHeight = 0; ///< CDF height
        std::uint32_t hiZMipCount = 0;         ///< Hi-Z mip levels; 0 disables the occlusion test
        std::uint32_t giEnabled = 0;           ///< 1 when GiPass supplies indirect; disables forward IBL/ambient
        sm::float4 sunDirection;               ///< xyz = world dir toward sun, w = shadow strength (0 disables)
        sm::float4 shadowParams;       ///< x = ray tMin, y = sky ambient floor, z = env_unit_nits, w = |proj[0][0]|
        sm::float4 presentationParams; ///< x = indirect ambient, y = pass flag, z = debug ray-hit, w = |proj[1][1]|
    };
    static_assert(sizeof(MeshPushConstants) == 256);
    bool createDepthTarget();
    bool createPipeline();
    bool createDescriptorSetLayouts();
    bool createPipelineLayouts();
    bool createOpaquePipeline();
    bool createTransparentPipeline();
    bool createSkyPipeline();
    VkShaderModule loadShaderModule(const char* filename);
    /// (Re)create the ping-pong per-meshlet visibility buffers for the current scene and
    /// clear both to 0 on the first frame. Called when the bound scene changes.
    bool ensureVisibilityBuffers();
    /// Record one opaque meshlet draw for the given cull phase / Hi-Z mip count.
    void drawOpaque(VkCommandBuffer cmd,
                    const MeshPushConstants& pcBase,
                    std::uint32_t cullPhase,
                    std::uint32_t hiZMipCount);
    void prepareAttachments(VkCommandBuffer cmd);
    void updateSceneDescriptors(VkCommandBuffer cmd);
    void dispatchGpuCull(VkCommandBuffer cmd, std::uint32_t instanceCount, const sm::float4x4& viewProj);
    void beginSceneRendering(VkCommandBuffer cmd, VkAttachmentLoadOp loadOp);
    void bindMeshSets(VkCommandBuffer cmd);
    void recordSky(VkCommandBuffer cmd);
    void recordTransparent(VkCommandBuffer cmd, const MeshPushConstants& pcBase);
    void recordOpaquePass(VkCommandBuffer cmd, const MeshPushConstants& pcBase);

    struct PipelineBuildState {
        VkShaderModule taskModule = VK_NULL_HANDLE;
        VkShaderModule meshModule = VK_NULL_HANDLE;
        VkShaderModule fragModule = VK_NULL_HANDLE;
        VkPipelineViewportStateCreateInfo viewport{};
        VkPipelineRasterizationStateCreateInfo rasterization{};
        VkPipelineMultisampleStateCreateInfo multisample{};
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        std::array<VkFormat, 3> colorFormats{};
        VkPipelineRenderingCreateInfo rendering{};
    } m_pipelineBuild{};

    Config m_config{};
    CameraParams m_camera{};
    const DeviceContext* m_ctx = nullptr;
    const Scene* m_scene = nullptr;

    // Depth-only target (color target is externally owned)
    Image m_depthTarget;
    // Thin GBuffer target (RGBA16F: view-space normal + roughness)
    Image m_gbufferTarget;
    // GI GBuffer target (RGBA32F: world-space primary hit position + material index)
    Image m_giBufferTarget;

    // Mesh + fragment graphics pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline m_graphicsPipelineTransparent = VK_NULL_HANDLE;

    // Sky/background fullscreen pipeline (samples env panorama, or black).
    struct SkyPushConstants {
        sm::float4x4 invViewProj;
        sm::float4 cameraPos;
        float exposure = 0.0f;
        std::uint32_t hasEnv = 0;
        float envScale = 0.0f; ///< env_unit_nits — physical cd/m² per raw EXR unit
        std::uint32_t _pad1 = 0;
    };
    static_assert(sizeof(SkyPushConstants) == 96);
    VkPipelineLayout m_skyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_skyPipeline = VK_NULL_HANDLE;
    bool m_hasEnv = false;
    sm::float3 m_sunDir{0.0f, 1.0f, 0.0f}; ///< world dir toward dominant IBL light
    float m_sunStrength = 0.0f;            ///< [0,1] ray-traced sun shadow strength
    float m_indirectAmbientStrength = 0.0f;
    std::uint32_t m_giEnabled = 0;  ///< 1 when GiPass is active (forward skips IBL/ambient)
    float m_debugRayHitMode = 0.0f; ///< 0=off, 1=ray-hit albedo, 2=ray-hit radiance (debug only)
    std::uint32_t m_transparentMaxDepth = 2;
    std::uint32_t m_frameSampleIndex = 0;
    std::uint32_t m_rngBaseSeed = 0x12345678U;
    bool m_deterministicReplay = false;
    bool m_cameraJitterEnabled = true;
    std::uint32_t m_rngDebug = 0;

    // Set 0: geometry buffers (vertex/instance/index/meshlet data — task + mesh stages)
    VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_meshSet = VK_NULL_HANDLE;

    // Set 1: material/lighting buffers (materials/lights/emissive — fragment stage)
    VkDescriptorSetLayout m_matSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_matSet = VK_NULL_HANDLE;

    // Set 2: IBL textures + samplers (fragment stage)
    VkDescriptorSetLayout m_iblSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_iblSet = VK_NULL_HANDLE;

    // Set 3: bindless material textures (base_color/normal/ORM/emission — fragment stage)
    static constexpr std::uint32_t kMaxBindlessTextures = 256;
    VkDescriptorSetLayout m_textureSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_textureSet = VK_NULL_HANDLE;
    const Scene* m_texturesBoundFor = nullptr; ///< scene the bindless set was last written for

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    VkDescriptorImageInfo m_iblDiffuseInfo{};
    VkDescriptorImageInfo m_iblSpecularInfo{};
    VkDescriptorImageInfo m_sheenLutInfo{};
    VkDescriptorImageInfo m_brdfLutInfo{};
    VkDescriptorImageInfo m_iblEnvSamplerInfo{};
    VkDescriptorImageInfo m_iblEnvRawInfo{};
    VkBuffer m_envMarginalCdf = VK_NULL_HANDLE;
    VkBuffer m_envConditionalCdf = VK_NULL_HANDLE;
    std::uint32_t m_envImportanceWidth = 0;
    std::uint32_t m_envImportanceHeight = 0;
    float m_envUnitNits = 1.0f; ///< env_unit_nits for the raw-env sky background

    // Tile-based light culling buffers (set by LightCuller before each recordFrame).
    VkBuffer m_tileLightCountsBuf = VK_NULL_HANDLE;
    VkBuffer m_tileLightIndicesBuf = VK_NULL_HANDLE;
    std::uint32_t m_tilesX = 0;
    std::uint32_t m_tilesY = 0;
    // Dummy 1-element buffers bound when no tile data is available (fallback to full light loop).
    Buffer m_dummyTileCounts;
    Buffer m_dummyTileIndices;

    bool m_initialized = false;
    bool m_hdrFirstUse = true; ///< tracks whether HDR image is still in UNDEFINED layout

    // GPU-driven frustum cull pass (GD2/GD3/GD6). Writes compactInstanceList + a single
    // indirect draw command; ForwardRenderer uses vkCmdDrawMeshTasksIndirectEXT (GD3) or
    // vkCmdExecuteGeneratedCommandsEXT (GD6 DGC path).
    GpuCullPass m_gpuCullPass;
    /// Identity list [0,1,...,kMaxInstances-1] for binding 10 when GpuCullPass is unavailable.
    Buffer m_identityInstanceList;
    /// DGC commands layout: one DRAW_MESH_TASKS_EXT token, stride=12.
    /// Non-null when VK_EXT_device_generated_commands is available (ctx.dgcSupported).
    VkIndirectCommandsLayoutEXT m_dgcLayout = VK_NULL_HANDLE;
    /// Preprocess buffer required by vkCmdExecuteGeneratedCommandsEXT (driver scratch space).
    /// Allocated with VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT (64-bit usage flag).
    VkBuffer m_dgcPreprocessBuf = VK_NULL_HANDLE;
    VmaAllocation m_dgcPreprocessAlloc = VK_NULL_HANDLE;
    VkDeviceAddress m_dgcPreprocessAddr = 0;
    VkDeviceSize m_dgcPreprocessSize = 0;

    // Two-pass Hi-Z occlusion culling (B4).
    HiZPass m_hiZPass;                    ///< current-frame depth pyramid builder
    Buffer m_meshletVisibility[2];        ///< ping-pong per-meshlet visibility (uint per meshlet)
    std::uint32_t m_visFrame = 0;         ///< index of the buffer holding PREVIOUS-frame visibility
    std::uint32_t m_visMeshletCount = 0;  ///< meshlet count the visibility buffers were sized for
    const Scene* m_visBuiltFor = nullptr; ///< scene the visibility buffers were built for
    bool m_hiZTestEnabled = true;         ///< set false for one frame on a camera cut
    bool m_visClearPrev = false;          ///< clear PREV visibility next frame (freshly (re)built)
    bool m_hiZDebugDisabled = false;      ///< THEIA_DISABLE_HIZ: draw all meshlets (A-B debug)
    bool m_forceSinglePass = false;       ///< THEIA_SINGLE_PASS: bypass two-pass Hi-Z (A-B debug)
};

} // namespace theia
#endif // THEIA_RENDERER_FORWARDRENDERER_HPP
