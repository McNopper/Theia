#pragma once

#include <volk/volk.h>

struct PassContext;

/// Common interface for GPU render / compute passes that operate on a frame.
class IRenderPass {
  public:
    virtual ~IRenderPass() = default;

    /// Record commands for this pass into ctx.cmd.
    virtual void record(const PassContext& ctx) noexcept = 0;

    /// Called when the render target is resized.  Implementations must recreate
    /// any extent-dependent resources.
    virtual void onResize(VkExtent2D extent) noexcept = 0;

    /// Debug name for logging and profiling.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};
