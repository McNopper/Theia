#include <cstring>
#include <filesystem>

#include "demo/Application.hpp"

int main(int argc, char* argv[]) {
    theia::Application::Config config;
    config.title = "Theia \xe2\x80\x94 Real-Time Renderer";
    config.width = 1024;
    config.height = 768;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "--scene") == 0 || std::strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            config.initialScene = argv[++i];
        } else if (std::strcmp(argv[i], "--validation") == 0) {
            config.validation = true;
        } else if (std::strcmp(argv[i], "--no-ui") == 0 || std::strcmp(argv[i], "--hide-ui") == 0) {
            config.hideUi = true;
        } else if ((std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            config.outputFile = argv[++i];
        } else {
            // Bare argument treated as scene file path
            config.initialScene = argv[i];
        }
    }

    theia::Application app;
    return app.run(config);
}
