//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "device.h"
#include "core/parser.h"

namespace luminous {
    class Pipeline : public Noncopyable {
    protected:
        Device *_device;
        Context *_context;
    public:
        void parse_scene();
    };
}