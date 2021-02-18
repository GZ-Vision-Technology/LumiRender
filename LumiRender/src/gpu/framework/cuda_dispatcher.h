//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "backend/dispatcher.h"
#include "helper/cuda.h"

namespace luminous {
    inline namespace gpu {
        class CUDADispatcher : public Dispatcher::Impl {
        public:
            cudaStream_t stream;

            CUDADispatcher(cudaStream_t stream) : stream(stream) {}

            void wait() override {CUDA_CHECK(cudaStreamSynchronize(stream)); }

            void then(std::function<void(void)> F) override {
                using Func = std::function<void(void)>;
                Func *f = new Func(std::move(F));
                auto wrapper = [](void *p) {
                    auto f = reinterpret_cast<Func *>(p);
                    (*f)();
                    delete f;
                };
                CUDA_CHECK(cudaLaunchHostFunc(stream, wrapper, (void *) f));
            }

            ~CUDADispatcher() {CUDA_CHECK(cudaStreamDestroy(stream)); }
        };
    }
}