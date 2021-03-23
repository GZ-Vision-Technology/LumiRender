//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "device.h"
#include "render/include/parser.h"
#include "scene.h"
#include "render/sensors/sensor_handle.h"

namespace luminous {
    using std::unique_ptr;
    class Task : public Noncopyable {
    protected:
        shared_ptr<Device> _device{nullptr};
        Context * _context{nullptr};
        SensorHandle _camera;
        SamplerHandle _sampler;
    public:
        Task(const shared_ptr<Device> &device, Context *context)
            : _device(device),
            _context(context) {}

        virtual void init(const Parser &parser) = 0;

        virtual void render_gui() = 0;

        virtual void render_cli() = 0;
    };
}