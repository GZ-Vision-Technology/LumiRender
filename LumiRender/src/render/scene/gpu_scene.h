//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/concepts.h"
#include "gpu/accel/megakernel_optix_accel.h"
#include "scene.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene : public Scene {

        public:
            GPUScene(Device *device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            LM_NODISCARD size_t size_in_bytes() const override;

            void create_device_memory() override;

            void synchronize_to_gpu();

            void fill_scene_data() override;

            void clear() override;

            void build_accel();

            LM_NODISCARD std::string description() const override;
        };
    }
}