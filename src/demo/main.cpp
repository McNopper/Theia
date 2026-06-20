#include <utility>
#include <string_view>

#include "demo/Application.hpp"

int main(int argc, char* const argv[]) {
    harmonia::App::Config config;
    config.title = "Theia \xe2\x80\x94 Real-Time Renderer";
    config.width = 1024;
    config.height = 768;
    config.resizable = true;
    config.assetsDir = THEIA_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";

    bool cameraJitterEnabled = true;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-camera-jitter") {
            cameraJitterEnabled = false;
            continue;
        }
        static_cast<void>(harmonia::App::applyCommonArg(config, i, argc, argv));
    }

    theia::Application app;
    app.setCameraJitterEnabled(cameraJitterEnabled);
    return app.run(std::move(config));
}
