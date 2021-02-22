//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "device.h"
#include "render/include/parser.h"
#include "scene.h"
#include "render/include/sensor.h"

namespace luminous {
    using std::unique_ptr;
    class Pipeline : public Noncopyable {
    protected:
        shared_ptr<Device> _device{nullptr};
        Context * _context{nullptr};
        unique_ptr<Scene> _scene{nullptr};
        SensorHandle _camera;
    public:
        Pipeline(const shared_ptr<Device> &device, Context *context)
            : _device(device),
            _context(context) {}

        void init(const Parser &parser) {

        }

        virtual void render_gui() = 0;

        virtual void render_cli() = 0;
    };
}