//
// Created by Zero on 2021/5/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "scene.h"
#include <embree3/rtcore.h>
#include "cpu/accel/embree_accel.h"
#include "scene_data.h"

namespace luminous {
    inline namespace cpu {

        class CPUScene : public Scene {
        public:
            CPUScene(Device *device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void create_device_memory() override;

            _NODISCARD RTCScene rtc_scene() const { return accel<EmbreeAccel>()->rtc_scene(); }

            _NODISCARD uint64_t scene_handle() const { return (uint64_t)rtc_scene(); }

            void fill_scene_data() override;
        };
    } // luminous::cpu
} // luminous