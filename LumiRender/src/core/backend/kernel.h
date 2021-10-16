//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "dispatcher.h"
#include "base_libs/lstd/function_trait.h"

namespace luminous {
    
    class KernelOld {
    public:
        class Impl {
        public:
            virtual void configure(uint3 grid_size,
                                   uint3 local_size,
                                   size_t sm) = 0;

            virtual void launch(Dispatcher &dispatcher,
                                void *args[]) = 0;

            virtual ~Impl() = default;
        };

        KernelOld &configure(uint3 grid_size,
                             uint3 block_size,
                             size_t sm = 0) {
            _impl->configure(grid_size, block_size, sm);
            return *this;
        }

        /**
         * The first parameter must be the number of item
         * @tparam Args : n_item, ...
         * @param dispatcher
         * @param args
         */
        template<typename... Args>
        void launch(Dispatcher &dispatcher, Args &...args) {
            void *array[]{(&args)...};
            _impl->launch(dispatcher, array);
        }

        explicit KernelOld(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    protected:
        std::unique_ptr<Impl> _impl;
    };
}