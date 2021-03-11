//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/backend/dispatcher.h"
#include "helper/cuda.h"
#include <cuda.h>

namespace luminous {
    inline namespace gpu {
        class CUDADispatcher : public Dispatcher::Impl {
        public:
            CUstream stream;

            CUDADispatcher() {
                CU_CHECK(cuStreamCreate(&stream, 0));
            }

            void wait() override {CU_CHECK(cuStreamSynchronize(stream)); }

            void then(std::function<void(void)> F) override {
                using Func = std::function<void(void)>;
                Func *f = new Func(std::move(F));
                auto wrapper = [](void *p) {
                    auto f = reinterpret_cast<Func *>(p);
                    (*f)();
                    delete f;
                };
                CU_CHECK(cuLaunchHostFunc(stream, wrapper, (void *) f));
            }

            ~CUDADispatcher() {CU_CHECK(cuStreamDestroy(stream)); }
        };
    }
}