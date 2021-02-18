//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "device.h"
#include "core/parser.h"
#include "render/include/scene.h"
#include "../parser.h"

namespace luminous {
    using std::unique_ptr;
    class Pipeline : public Noncopyable {
    protected:
        unique_ptr<Device> _device{nullptr};
        Context * _context{nullptr};
        unique_ptr<Scene> _scene{nullptr};
    public:
        Pipeline(unique_ptr<Device> device, Context *context)
            : _device(move(device)),
            _context(context) {}

        void init(const Parser &parser) {

        }

        virtual void render_gui() = 0;

        virtual void render_cli() = 0;
    };
}