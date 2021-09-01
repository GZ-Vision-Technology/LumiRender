//
// Created by Zero on 2021/5/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/include/scene.h"
#include <embree3/rtcore.h>
#include "embree_accel.h"
#include "render/include/scene_data.h"

namespace luminous {
    inline namespace cpu {

        class CPUScene : public Scene {
        private:
            UP<EmbreeAccel> _embree_accel;
        public:
            CPUScene(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void create_device_memory() override;

            NDSC RTCScene rtc_scene() const { return _embree_accel->rtc_scene(); }

            NDSC uint64_t scene_handle() const { return (uint64_t)rtc_scene(); }

            void fill_scene_data() override;

            void init_accel() override;

            void build_accel();
        };
    } // luminous::cpu
} // luminous