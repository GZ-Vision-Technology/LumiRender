//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "dispatcher.h"

namespace luminous {
    class Kernel {
    public:
        class Impl {
        public:
            virtual void configure(uint3 grid_size,
                                   uint3 local_size,
                                   size_t sm) = 0;

            virtual void launch(Dispatcher &dispatcher,
                                std::vector<void *> &args) = 0;

            virtual void launch(Dispatcher &dispatcher, int n_items,
                                std::vector<void *> &args) = 0;

            virtual ~Impl() = default;
        };

        Kernel &configure(uint3 grid_size,
                          uint3 block_size,
                          size_t sm = 0) {
            _impl->configure(grid_size, block_size, sm);
            return *this;
        }

        void launch(Dispatcher &dispatcher,
                    std::vector<void *> args) {
            _impl->launch(dispatcher, args);
        }

        void launch(Dispatcher &dispatcher, int n_items,
                    std::vector<void *> &args) {
            _impl->launch(dispatcher, n_items, args);
        }

        explicit Kernel(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    protected:
        std::unique_ptr<Impl> _impl;
    };
}