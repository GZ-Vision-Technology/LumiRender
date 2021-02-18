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
            virtual void launch(Dispatcher &dispatcher,
                                std::vector<void *> args,
                                uint3 global_size,
                                uint3 local_size) = 0;

            virtual ~Impl() = default;
        };

        void launch(Dispatcher &dispatcher,
                    std::vector<void *> args,
                    uint3 global_size,
                    uint3 local_size) {
            impl->launch(dispatcher, std::move(args), global_size, local_size);
        }

        explicit Kernel(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

    private:
        std::unique_ptr<Impl> impl;
    };
}