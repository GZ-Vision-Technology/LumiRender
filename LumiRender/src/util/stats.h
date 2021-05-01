//
// Created by Zero on 2021/3/7.
//


#pragma once

#include "core/logging.h"
#include <chrono>
#include "clock.h"

namespace luminous {
    using namespace std::chrono;

    struct TaskTag {
        const char *task_name;
        Clock clock;

        TaskTag(const char *tn)
                : task_name(tn) {
            clock.tic();
            LUMINOUS_INFO(task_name, " start!")
        }

        ~TaskTag() {
            auto dt = clock.toc();
            LUMINOUS_INFO(string_printf("%s complete, elapsed time is %g s", task_name, dt / 1000));
        }
    };
}

#define TASK_TAG(task_name) TaskTag __(#task_name);