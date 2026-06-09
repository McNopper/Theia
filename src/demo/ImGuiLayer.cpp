#include "demo/ImGuiLayer.hpp"

#define IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
#include <volk/volk.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include "harmonia/core/Logger.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

bool ImGuiLayer::initialize(const Config& cfg) {
    m_device = cfg.device;

    // Descriptor pool for ImGui (large enough for fonts + textures)
    const VkDescriptorPoolSize poolSizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 16,
        .poolSizeCount = 1,
        .pPoolSizes = poolSizes,
    };
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS) {
        Logger::error("ImGuiLayer: failed to create descriptor pool");
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dark style with a slight Theia tweak
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    // Muted blue accent
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.18f, 0.35f, 0.55f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.18f, 0.35f, 0.55f, 0.60f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.48f, 0.72f, 0.80f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.75f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.35f, 0.60f, 0.90f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.40f, 0.65f, 0.80f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.52f, 0.80f, 1.00f);

    if (!ImGui_ImplSDL3_InitForVulkan(cfg.window)) {
        Logger::error("ImGuiLayer: ImGui_ImplSDL3_InitForVulkan failed");
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        ImGui::DestroyContext();
        return false;
    }

    const VkFormat colorFmt = cfg.swapchainColorFormat;
    const VkPipelineRenderingCreateInfo dynRenderInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFmt,
    };
    ImGui_ImplVulkan_InitInfo vulkanInfo{};
    vulkanInfo.Instance = cfg.instance;
    vulkanInfo.PhysicalDevice = cfg.physDevice;
    vulkanInfo.Device = cfg.device;
    vulkanInfo.QueueFamily = cfg.graphicsQueueFamily;
    vulkanInfo.Queue = cfg.graphicsQueue;
    vulkanInfo.DescriptorPool = m_pool;
    vulkanInfo.RenderPass = VK_NULL_HANDLE; // dynamic rendering only
    vulkanInfo.MinImageCount = cfg.minImageCount;
    vulkanInfo.ImageCount = cfg.imageCount;
    vulkanInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vulkanInfo.UseDynamicRendering = true;
    vulkanInfo.PipelineRenderingCreateInfo = dynRenderInfo;

    if (!ImGui_ImplVulkan_Init(&vulkanInfo)) {
        Logger::error("ImGuiLayer: ImGui_ImplVulkan_Init failed");
        ImGui_ImplSDL3_Shutdown();
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        ImGui::DestroyContext();
        return false;
    }

    // Upload default font atlas
    ImGui_ImplVulkan_CreateFontsTexture();

    m_initialized = true;
    Logger::info("ImGuiLayer: initialized (Dear ImGui {})", IMGUI_VERSION);
    return true;
}

void ImGuiLayer::shutdown() {
    if (!m_initialized)
        return;
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    m_initialized = false;
}

void ImGuiLayer::processEvent(const SDL_Event& e) {
    if (m_initialized)
        ImGui_ImplSDL3_ProcessEvent(&e);
}

void ImGuiLayer::beginFrame() {
    if (!m_initialized)
        return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent) {
    if (!m_initialized)
        return;

    ImGui::Render();

    // Open a dynamic rendering pass over the already-tonemapped swapchain image.
    // LOAD to preserve the tonemapped content underneath the UI.
    const VkRenderingAttachmentInfo colorAttach{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = swapchainView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    const VkRenderingInfo renderInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttach,
    };
    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
