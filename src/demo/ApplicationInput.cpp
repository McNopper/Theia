// ApplicationInput.cpp - the input handling of the Theia demo application: the keys,
// the mouse look and the EV100 keys of the interactive window. Methods of
// theia::Application split out of Application.cpp (TU split, no behavior change).
#include <cmath>

#include "demo/Application.hpp"
#include "harmonia/core/Logger.hpp"

namespace theia {

bool Application::onEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = (event.type == SDL_EVENT_KEY_DOWN);
        switch (event.key.scancode) {
        case SDL_SCANCODE_W:
            m_camCtrl.wDown = down;
            return true;
        case SDL_SCANCODE_A:
            m_camCtrl.aDown = down;
            return true;
        case SDL_SCANCODE_S:
            m_camCtrl.sDown = down;
            return true;
        case SDL_SCANCODE_D:
            m_camCtrl.dDown = down;
            return true;
        case SDL_SCANCODE_Q:
            m_camCtrl.qDown = down;
            return true;
        case SDL_SCANCODE_E:
            m_camCtrl.eDown = down;
            return true;
        case SDL_SCANCODE_ESCAPE:
            // ESC releases mouse capture first; quitting is the host default.
            if (down && m_camCtrl.captured) {
                SDL_SetWindowRelativeMouseMode(window(), false);
                m_camCtrl.captured = false;
                return true;
            }
            return false;
        // [ / ] keys: decrease / increase EV100 by 0.5 stops
        case SDL_SCANCODE_LEFTBRACKET:
            if (down) {
                const float ev100 = m_camera.physical.ev100() - 0.5f;
                m_camera.physical.shutterSpeedHz = std::pow(2.0f, ev100);
                harmonia::core::Logger::info("EV100 = {:.1f}", m_camera.physical.ev100());
            }
            return true;
        case SDL_SCANCODE_RIGHTBRACKET:
            if (down) {
                const float ev100 = m_camera.physical.ev100() + 0.5f;
                m_camera.physical.shutterSpeedHz = std::pow(2.0f, ev100);
                harmonia::core::Logger::info("EV100 = {:.1f}", m_camera.physical.ev100());
            }
            return true;
        default:
            return false;
        }
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (event.button.button == SDL_BUTTON_RIGHT) {
            m_camCtrl.captured = true;
            SDL_SetWindowRelativeMouseMode(window(), true);
            return true;
        }
        return false;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event.button.button == SDL_BUTTON_RIGHT) {
            m_camCtrl.captured = false;
            SDL_SetWindowRelativeMouseMode(window(), false);
            return true;
        }
        return false;

    case SDL_EVENT_MOUSE_MOTION:
        if (m_camCtrl.captured) {
            m_camCtrl.handleMotion(event.motion.xrel, event.motion.yrel);
            return true;
        }
        return false;

    case SDL_EVENT_MOUSE_WHEEL:
        // Scroll up = faster, scroll down = slower
        m_camCtrl.adjustSpeed(static_cast<float>(event.wheel.y));
        return true;

    default:
        return false;
    }
}

} // namespace theia
