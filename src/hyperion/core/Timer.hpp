#pragma once

#include <chrono>
#include <string>
#include <string_view>

class Timer {
  public:
    void start() noexcept;
    void stop() noexcept;
    [[nodiscard]] double elapsedMs() const noexcept;

  private:
    using Clock = std::chrono::high_resolution_clock;

    Clock::time_point m_start{};
    Clock::time_point m_end{};
    bool m_running = false;
    bool m_started = false;
};

class ScopedTimer {
  public:
    explicit ScopedTimer(std::string_view label);
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;
    ~ScopedTimer();

  private:
    std::string m_label;
    Timer m_timer;
};
