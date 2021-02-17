//
// Created by Zero on 2021/2/17.
//


#pragma once


#include "core/concepts.h"

namespace luminous {
    class Dispatcher : Noncopyable {
    public:
        class Impl {
        public:
            virtual ~Impl() = default;

            virtual void then(std::function<void(void)> F) = 0;

            virtual void wait() = 0;
        };

        void wait() { impl->wait(); }

        Dispatcher &then(std::function<void(void)> F) {
            impl->then(std::move(F));
            return *this;
        }

        Dispatcher(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

        Impl *impl_mut() const { return impl.get(); }

    protected:
        std::unique_ptr<Impl> impl;
    };
}