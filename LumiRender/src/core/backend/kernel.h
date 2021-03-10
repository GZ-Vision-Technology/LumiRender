//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "dispatcher.h"

namespace luminous {
    class Kernel {
    public:
        class Impl {
        public:
            virtual void configure(uint3 grid_size,
                                   uint3 local_size) = 0;

            virtual void launch(Dispatcher &dispatcher,
                                std::vector<void *> args) = 0;

            virtual ~Impl() = default;
        };

        Kernel& configure(uint3 grid_size,
                       uint3 block_size) {
            impl->configure(grid_size, block_size);
            return *this;
        }

        void launch(Dispatcher &dispatcher,
                       std::vector<void *> args) {
            impl->launch(dispatcher, std::move(args));
        }

        explicit Kernel(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

//    private:
        std::unique_ptr<Impl> impl;
    };
}