#include "hyperion/core/Logger.hpp"

#include <cstdio>
#include <mutex>
#include <print>

void Logger::log(Level level, std::string message) {
    static std::mutex mutex;
    std::lock_guard lock(mutex);

    const char* levelString = "UNKNOWN";
    switch (level) {
    case Level::Info:
        levelString = "INFO";
        break;
    case Level::Warning:
        levelString = "WARNING";
        break;
    case Level::Error:
        levelString = "ERROR";
        break;
    }

    const std::string formatted = std::format("[HYPERION][{}] {}", levelString, message);
    std::FILE* const stream = level == Level::Error ? stderr : stdout;
    std::print(stream, "{}\n", formatted);
    std::fflush(stream);
}
