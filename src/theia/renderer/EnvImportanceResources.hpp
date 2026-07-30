#ifndef THEIA_RENDERER_ENVIMPORTANCERESOURCES_HPP
#define THEIA_RENDERER_ENVIMPORTANCERESOURCES_HPP

#include <volk/volk.h>

#include <cstdint>

#include "theia/renderer/GiPass.hpp"

namespace theia {

/// Environment-map importance-sampling resources captured at scene-load and bound to the
/// per-frame GI dispatch. A cohesive grouping of the env view/sampler, the marginal +
/// conditional CDF buffers, and the env-presence / unit-nits flags that GiPass consumes.
struct EnvImportanceResources {
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer marginalCdf = VK_NULL_HANDLE;
    VkBuffer conditionalCdf = VK_NULL_HANDLE;
    std::uint32_t cdfWidth = 0;
    std::uint32_t cdfHeight = 0;
    bool hasEnv = false;
    float nits = 1.0f;

    /// Write the env-importance fields of a GiPass frame-parameter block.
    void fill(GiPass::FrameParams& gp) const noexcept {
        gp.envMapView = view;
        gp.envSampler = sampler;
        gp.envMarginalCdf = marginalCdf;
        gp.envConditionalCdf = conditionalCdf;
        gp.envImportanceWidth = cdfWidth;
        gp.envImportanceHeight = cdfHeight;
        gp.hasEnvMap = hasEnv;
        gp.envLuminanceScale = nits;
    }
};

} // namespace theia

#endif // THEIA_RENDERER_ENVIMPORTANCERESOURCES_HPP
