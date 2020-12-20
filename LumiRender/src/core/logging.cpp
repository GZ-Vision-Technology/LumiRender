//
// Created by Zero on 2020/9/14.
//

#include "logging.h"

namespace luminous {
    inline namespace logging {
        spdlog::logger &logger() noexcept {
            static auto ret = spdlog::stdout_color_mt("console");
            return *ret;
        }
    }
}