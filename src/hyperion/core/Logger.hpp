#pragma once

#include <format>
#include <string>
#include <string_view>
#include <utility>

class Logger {
  public:
    template <typename... Args> static void info(std::format_string<Args...> fmt, Args&&... args) {
        log(Level::Info, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> static void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(Level::Warning, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> static void error(std::format_string<Args...> fmt, Args&&... args) {
        log(Level::Error, std::format(fmt, std::forward<Args>(args)...));
    }

  private:
    enum class Level {
        Info,
        Warning,
        Error,
    };

    static void log(Level level, std::string message);
};
