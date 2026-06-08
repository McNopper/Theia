#include "hyperion/core/Timer.hpp"

#include <string>
#include <utility>

#include "hyperion/core/Logger.hpp"

void Timer::start() noexcept {
    m_start = Clock::now();
    m_end = m_start;
    m_running = true;
    m_started = true;
}

void Timer::stop() noexcept {
    if (!m_started) {
        return;
    }

    m_end = Clock::now();
    m_running = false;
}

[[nodiscard]] double Timer::elapsedMs() const noexcept {
    if (!m_started) {
        return 0.0;
    }

    const Clock::time_point end = m_running ? Clock::now() : m_end;
    return std::chrono::duration<double, std::milli>(end - m_start).count();
}

ScopedTimer::ScopedTimer(std::string_view label) : m_label(label) {
    m_timer.start();
}

ScopedTimer::~ScopedTimer() {
    m_timer.stop();

    try {
        Logger::info("{} took {:.3f} ms", m_label, m_timer.elapsedMs());
    } catch (...) {
    }
}
