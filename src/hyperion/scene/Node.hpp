#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hyperion/scene/Geometry.hpp"

class Node {
  public:
    std::string name;
    std::unique_ptr<Geometry> geometry;
    std::vector<std::shared_ptr<Node>> children;

    [[nodiscard]] const Xform& xform() const noexcept {
        static const Xform identity{};
        return geometry ? geometry->xform : identity;
    }
};
