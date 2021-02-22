//
// Created by Zero on 2021/2/21.
//


#pragma once

#include "sensor.h"
#include "shape.h"
#include "material.h"
#include "light.h"
#include "light_sampler.h"
#include "core/concepts.h"
#include <memory>
#include "model.h"

namespace luminous {
    inline namespace render {
        struct Config {
        };
        struct IntegratorConfig : Config {

        };

        struct SamplerConfig : Config {

        };

        struct ShapeConfig : Config {

        };

        struct MaterialConfig : Config {

        };

        struct CameraConfig : Config {

        };

        struct LightSamplerConfig : Config {

        };
        using namespace std;
        struct SceneGraph {
            vector<shared_ptr<const Mesh>> mesh_list;
            vector<MeshInstance> instance_list;
            void update_mesh_list(const shared_ptr<const Model> &model);

        };

    };
}