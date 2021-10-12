//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "task.h"
#include "gpu/framework/cuda_impl.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {
        class CUDATask : public Task {
        public:
            explicit CUDATask(Context *context)
                : Task(create_cuda_device(), context) {}

            void init(const Parser &parser) override;

            void render_cli() override {}

        };
    }
}