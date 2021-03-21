//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/task.h"
#include "cuda_device.h"

namespace luminous {
    inline namespace gpu {
        class CUDATask : public Task {
        private:
            unique_ptr<Scene> _scene{nullptr};
        public:
            CUDATask(Context *context)
                : Task(create_cuda_device(), context) {}

            void init(const Parser &parser) override;

            void render_cli() override {}

            void render_gui() override;
        };
    }
}