//
// Created by Zero on 2021/3/7.
//


#pragma once

#include "logging.h"

namespace luminous {
    struct TaskTag {
        const char *task_name;

        TaskTag(const char *tn)
                : task_name(tn) {
            LUMINOUS_INFO(task_name, " start!")
        }

        ~TaskTag() {
            LUMINOUS_INFO(task_name, " complete!")
        }
    };
}

#define TASK_TAG(task_name) TaskTag __(#task_name);