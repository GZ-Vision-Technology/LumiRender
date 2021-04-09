//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "render/include/scene_graph.h"
#include "render/sensors/sensor.h"

namespace luminous {
    class Integrator : public Noncopyable {
    public:
        virtual ~Integrator() {}

        virtual void init(const std::shared_ptr<SceneGraph> &scene_graph) = 0;

        virtual Sensor *camera() = 0;

        virtual void update() = 0;

        virtual void render() = 0;
    };
}