#include <string_view>
#include <utility>

#include "demo/Application.hpp"
#include "harmonia/core/Logger.hpp"

namespace {
constexpr float kDefaultDenoiserStrength = 0.45F;
constexpr float kRestirPtDenoiserStrength = 0.55F;
} // namespace

int main(int argc, char* const argv[]) {
    harmonia::Logger::setTag("THEIA");
    harmonia::App::Config config;
    config.title = "Theia \xe2\x80\x94 Real-Time Renderer";
    config.width = 1024;
    config.height = 768;
    config.resizable = true;
    config.assetsDir = THEIA_ASSETS_DIR;
    config.sceneFile = "cornell_classic.scene.toml";

    bool cameraJitterEnabled = true;
    bool restirDiEnabled = true;
    // GI2: ReSTIR PT Enhanced — unified DI+GI path reservoir (default on). Mutually
    // exclusive with ReSTIR DI (the unified reservoir absorbs the separate DI path).
    // `--no-restir-pt` falls back to the legacy ReSTIR DI direct-illumination path.
    bool restirPtEnabled = true;
    // GI2 full PT: multi-bounce path reservoir for the indirect term (default on,
    // only effective with PT). `--no-restir-pt-path` keeps PT but runs the classic
    // per-sample multi-bounce walk instead.
    bool restirPtPathEnabled = true;
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
        if (arg == "--no-restir-pt") {
            restirPtEnabled = false;
            continue;
        }
        if (arg == "--no-restir-pt-path") {
            restirPtPathEnabled = false;
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
        if (!harmonia::CliParser::applyCommonArg(config, i, argc, argv)) {
            harmonia::Logger::error("Unknown argument: '{}'", argv[i]);
            return 2;
        }
    }

    // Reject incompatible flag combinations rather than silently degrading the output.
    // Offscreen capture (--output) integrates over many jittered/stochastic samples via
    // progressive accumulation; layering TAA's temporal-reprojection blend on top double-
    // filters the result (blurring/ghosting). TAA and offscreen accumulation cannot both
    // be active, so an *explicit* --taa together with --output is not allowed.
    if (taaForcedOn && !config.outputFile.empty()) {
        harmonia::Logger::error("--taa is incompatible with offscreen capture (--output): offscreen rendering uses "
                                "progressive accumulation for anti-aliasing, and TAA would double-filter it. Drop "
                                 "--taa for offscreen capture, or omit --output for interactive TAA.");
        return 2;
    }

    // --no-postfx forces off every post-processing stage: the shared A-SVGF denoiser
    // (gated in App::applySceneStageConfig) and TAA here. It wins over --taa so the
    // interactive window shows the raw estimator result.
    if (config.noPostfx) {
        taaEnabled = false;
    }

    theia::Application app;
    app.setCameraJitterEnabled(cameraJitterEnabled);
    app.setRestirDiEnabled(restirDiEnabled);
    app.setRestirPtEnabled(restirPtEnabled);
    app.setRestirPtPathEnabled(restirPtPathEnabled);
    // GI2 Phase 4: A-SVGF correlation awareness. ReSTIR PT's spatial reuse introduces
    // inter-pixel sample correlation (neighbours share reservoir samples). The à-trous
    // filter's per-pixel variance estimate (from the gradient/variance guide) under-counts
    // this shared-sample correlation, so a slightly stronger spatial filter compensates.
    // Temporal accumulation is unaffected — per-frame RNG reseeding keeps temporal paths
    // independent. Only tune the interactive-window denoiser; the parity/accumulation path
    // attenuates the denoiser to identity regardless.
    // M-aware denoising (GI2.7) is shipped: the denoiser reads the reservoirs' effective
    // sample count M via the HDR alpha channel.
    if (restirPtEnabled && config.denoiser.strength == kDefaultDenoiserStrength) {
        config.denoiser.strength = kRestirPtDenoiserStrength;
    }
    app.setTaaEnabled(taaEnabled);
    return app.run(std::move(config));
}
