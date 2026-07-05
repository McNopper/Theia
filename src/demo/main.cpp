#include <utility>
#include <string_view>

#include "demo/Application.hpp"
#include "harmonia/core/Logger.hpp"

int main(int argc, char* const argv[]) {
    harmonia::App::Config config;
    config.title = "Theia \xe2\x80\x94 Real-Time Renderer";
    config.width = 1024;
    config.height = 768;
    config.resizable = true;
    config.assetsDir = THEIA_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";

    bool cameraJitterEnabled = true;
    bool restirDiEnabled = true;
    // TAA is a real-time *window* feature: it temporally reprojects fresh, non-accumulated
    // frames during camera motion. It is on by default for the interactive window but does
    // NOT run under offscreen capture, which uses progressive accumulation instead (the two
    // are mutually-exclusive temporal-integration approaches — see Application). `--taa`
    // force-enables it; `--no-taa` disables it.
    bool taaEnabled = true;
    bool taaForcedOn = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-camera-jitter") {
            cameraJitterEnabled = false;
            continue;
        }
        if (arg == "--no-restir-di") {
            restirDiEnabled = false;
            continue;
        }
        if (arg == "--no-taa") {
            taaEnabled = false;
            continue;
        }
        if (arg == "--taa") {
            taaEnabled = true;
            taaForcedOn = true;
            continue;
        }
        static_cast<void>(harmonia::App::applyCommonArg(config, i, argc, argv));
    }

    // Reject incompatible flag combinations rather than silently degrading the output.
    // Offscreen capture (--output) integrates over many jittered/stochastic samples via
    // progressive accumulation; layering TAA's temporal-reprojection blend on top double-
    // filters the result (blurring/ghosting). TAA and offscreen accumulation cannot both
    // be active, so an *explicit* --taa together with --output is not allowed.
    if (taaForcedOn && !config.outputFile.empty()) {
        Logger::error(
            "--taa is incompatible with offscreen capture (--output): offscreen rendering uses "
            "progressive accumulation for anti-aliasing, and TAA would double-filter it. Drop "
            "--taa for offscreen capture, or omit --output for interactive TAA.");
        return 2;
    }

    theia::Application app;
    app.setCameraJitterEnabled(cameraJitterEnabled);
    app.setRestirDiEnabled(restirDiEnabled);
    app.setTaaEnabled(taaEnabled);
    return app.run(std::move(config));
}
