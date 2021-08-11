//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/task.h"
#include "cuda_impl.h"
#include "render/films/film.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {
        class CUDATask : public Task {
        private:

            uint test_count = 0;

            double acc_t = 0;

        public:
            explicit CUDATask(Context *context)
                : Task(create_cuda_device(), context) {}

            void init(const Parser &parser) override;

            void update_device_buffer() override;

            void render_cli() override {}

            void update() override;

            FrameBufferType *download_frame_buffer() override;

            void render_gui(double dt) override;
        };
    }
}