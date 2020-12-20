//
// Created by Zero on 2020/9/14.
//

#pragma once

#include <exception>
#include <iostream>
#include <filesystem>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "string_util.h"

namespace luminous {

inline namespace logging {

    spdlog::logger &logger() noexcept;

    inline void set_log_level(spdlog::level::level_enum lvl) noexcept {
        logger().set_level(lvl);
    }

    template<typename... Args>
    inline void debug(Args &&... args) noexcept {
        logger().debug(serialize(std::forward<Args>(args)...));
    }

    template<typename... Args>
    inline void info(Args &&... args) noexcept {
        logger().info(serialize(std::forward<Args>(args)...));
    }

    template<typename... Args>
    inline void warning(Args &&... args) noexcept {
        logger().warn(serialize(std::forward<Args>(args)...));
    }

    template<typename... Args>
    inline void warning_if(bool predicate, Args &&... args) noexcept {
        if (predicate) { warning(std::forward<Args>(args)...); }
    }

    template<typename... Args>
    inline void warning_if_not(bool predicate, Args &&... args) noexcept {
        warning_if(!predicate, std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[noreturn]] inline void exception(Args &&... args) {
        throw std::runtime_error{serialize(std::forward<Args>(args)...)};
    }

    template<typename... Args>
    inline void exception_if(bool predicate, Args &&... args) {
        if (predicate) { exception(std::forward<Args>(args)...); }
    }

    template<typename... Args>
    inline void exception_if_not(bool predicate, Args &&... args) {
        exception_if(!predicate, std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[noreturn]] inline void error(Args &&... args) {
        logger().error(serialize(std::forward<Args>(args)...));
        exit(-1);
    }

    template<typename... Args>
    inline void error_if(bool predicate, Args &&... args) {
        if (predicate) { error(std::forward<Args>(args)...); }
    }

    template<typename... Args>
    inline void error_if_not(bool predicate, Args &&... args) {
        error_if(!predicate, std::forward<Args>(args)...);
    }
}

}// namespace luminous::logging

#define LUMINOUS_SOURCE_LOCATION __FILE__ , ":", __LINE__

#define debug(...) \
    ::luminous::logging::debug(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);

#define SET_LOG_LEVEL(lv) \
    ::luminous::logging::set_log_level(spdlog::level::level_enum::lv);

#define LUMINOUS_INFO(...) \
    ::luminous::logging::info(__VA_ARGS__);

#define LUMINOUS_WARNING(...) \
    ::luminous::logging::warning(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_WARNING_IF(...) \
    ::luminous::logging::warning_if(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_WARNING_IF_NOT(...) \
    ::luminous::logging::warning_if_not(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);

#define LUMINOUS_EXCEPTION(...) \
    ::luminous::logging::exception(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_EXCEPTION_IF(...) \
    ::luminous::logging::exception_if(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_EXCEPTION_IF_NOT(...) \
    ::luminous::logging::exception_if_not(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);

#define LUMINOUS_ERROR(...) \
    ::luminous::logging::error(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_ERROR_IF(...) \
    ::luminous::logging::error_if(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);
#define LUMINOUS_ERROR_IF_NOT(...) \
    ::luminous::logging::error_if_not(__VA_ARGS__, "\n    Source: ", LUMINOUS_SOURCE_LOCATION);