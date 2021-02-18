//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/pipeline.h"
#include "cuda_device.h"

namespace luminous {
    inline namespace gpu {
        class CUDAPipeline : public Pipeline {

        public:
            CUDAPipeline(unique_ptr<CUDADevice> cuda_device, Context *context)
                : Pipeline(make_unique<Device>(move(cuda_device)),context) {}

//            CUDAPipeline(unique_ptr<Device> device, Context *context)
//                    : Pipeline(move(device), context) {}

            virtual void render_cli() override {

            }

            virtual void render_gui() override {

            }
        };
    }
}