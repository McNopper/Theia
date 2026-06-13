#include <utility>

#include "demo/Application.hpp"

int main(int argc, char* argv[]) {
    harmonia::App::Config config;
    config.title = "Theia \xe2\x80\x94 Real-Time Renderer";
    config.width = 1024;
    config.height = 768;
    config.resizable = true;
    config.assetsDir = THEIA_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";

    for (int i = 1; i < argc; ++i) {
        static_cast<void>(harmonia::App::applyCommonArg(config, i, argc, argv));
    }

    theia::Application app;
    return app.run(std::move(config));
}
