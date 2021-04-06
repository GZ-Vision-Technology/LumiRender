//
// Created by Zero on 2021/3/7.
//


#pragma once

#include "logging.h"
#include <chrono>

namespace luminous {
    using namespace std::chrono;

    struct TaskTag {
        const char *task_name;
        system_clock::time_point start_time_point;

        TaskTag(const char *tn)
                : task_name(tn) {
            start_time_point = std::chrono::system_clock::now();
            LUMINOUS_INFO(task_name, " start!")
        }

        double compute_dt() {
            auto cur_time = std::chrono::system_clock::now();
            auto duration = duration_cast<microseconds>(cur_time - start_time_point);
            return double(duration.count()) * microseconds::period::num / microseconds::period::den;
        }

        ~TaskTag() {
            double dt = compute_dt();
            LUMINOUS_INFO(string_printf("%s complete, elapsed time is %g", task_name, dt));
        }
    };
}

#define TASK_TAG(task_name) TaskTag __(#task_name);