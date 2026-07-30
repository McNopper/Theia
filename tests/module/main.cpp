// Custom main for module tests.
// SDL3 requires SDL_SetMainReady() when GTest owns main().
// One Vulkan context is created here and shared across all tests via g_vulkanTestCtx.

#define SDL_MAIN_HANDLED
#include <volk/volk.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <gtest/gtest.h>
#include <memory>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/vulkan_init/Context.hpp"

namespace {
struct WindowDeleter {
    void operator()(SDL_Window* w) const noexcept {
        if (w != nullptr) {
            SDL_DestroyWindow(w);
        }
    }
};
} // namespace

int main(int argc, char* argv[]) {
    SDL_SetMainReady();
    ::testing::InitGoogleTest(&argc, argv);

    // Best-effort Vulkan setup — tests call GTEST_SKIP() if context is nullptr.
    VulkanTestContext testCtx;
    std::unique_ptr<SDL_Window, WindowDeleter> window;

    if (volkInitialize() == VK_SUCCESS && SDL_Init(SDL_INIT_VIDEO)) {
        window.reset(SDL_CreateWindow("Theia Module Tests", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN));
        if (window != nullptr) {
            harmonia::Context::Config cfg;
            cfg.appName = "TheiaModuleTests";
            cfg.enableValidation = false;
            cfg.window = window.get();

            auto ctx = harmonia::Context::create(cfg);
            if (ctx) {
                testCtx.context = std::make_unique<harmonia::Context>(std::move(*ctx));
                auto pool = harmonia::CommandPool::create(testCtx.context->deviceContext(),
                                                          testCtx.context->deviceContext().graphicsFamily);
                if (pool) {
                    testCtx.commandPool = std::make_unique<harmonia::CommandPool>(std::move(*pool));
                    testCtx.window = window.get();
                    g_vulkanTestCtx = &testCtx;
                }
            }
        }
    }

    const int result = RUN_ALL_TESTS();

    g_vulkanTestCtx = nullptr;
    testCtx.commandPool.reset();
    testCtx.context.reset();
    window.reset();
    SDL_Quit();
    return result;
}
