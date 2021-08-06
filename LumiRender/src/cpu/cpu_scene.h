//
// Created by Zero on 2021/5/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "render/include/scene.h"
#include "embree_accel.h"

namespace luminous {
    inline namespace cpu {
        class CPUScene : public Scene {
        private:
            friend class EmbreeAccel;
            UP<EmbreeAccel> _embree_accel;
        public:
            CPUScene(Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void preload_textures(const SP<SceneGraph> &scene_graph);

            void init_accel() override;

            void build_accel();
        };
    } // luminous::cpu
} // luminous