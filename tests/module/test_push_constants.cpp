// The push-constant block of the GI stage is the largest of the family: 256 bytes,
// twice what Vulkan guarantees. The device selection requires a limit that covers it
// (harmonia::vulkan_init::supportsPushConstants, enforced when the device is chosen).
// This ties the requirement to the real block: one that grows past it fails here, not
// on a device that reports exactly the requirement. No device needed.

#include <gtest/gtest.h>

#include "harmonia/vulkan_init/PhysicalDevice.hpp"
#include "theia/gi/GiPass.hpp"

TEST(TheiaPushConstants, GiStageBlockIsWhatTheDeviceSelectionRequires) {
    EXPECT_EQ(theia::GiPass::kPushConstantsSize, 256U) << "the block the requirement is calibrated for";
    EXPECT_LE(theia::GiPass::kPushConstantsSize, harmonia::vulkan_init::kRequiredPushConstantsSize)
        << "the block grew past the limit devices are selected for";
}
