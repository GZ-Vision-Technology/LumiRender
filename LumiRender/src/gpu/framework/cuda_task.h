//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/task.h"
#include "cuda_device.h"
#include "render/films/film_handle.h"
#include "../mega_kernel_pt.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {
        class CUDATask : public Task {
        private:

            vector<FrameBufferType> _host_frame_buffer;

            Buffer<float4> _accumulate_buffer{nullptr};

            Managed<FrameBufferType *> _frame_buffer;

            UP<Integrator> _integrator;

        public:
            CUDATask(Context *context)
                : Task(create_cuda_device(), context) {}

            void init(const Parser &parser) override;

            void update_device_buffer();

            void render_cli() override {}

            FrameBufferType *download_frame_buffer();

            int2 resolution();

            void render_gui() override;

            void on_key(int key,int scancode, int action, int mods);

            void update_camera_fov_y(float val);

            void update_camera_view(float d_yaw, float d_pitch);

            void update_film_resolution(int2 res);
        };
    }
}