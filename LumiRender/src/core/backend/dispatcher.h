//
// Created by Zero on 2021/2/17.
//


#pragma once


#include "core/concepts.h"
#include <functional>

namespace luminous {
    class Dispatcher : Noncopyable {
    public:
        class Impl {
        public:
            virtual ~Impl() = default;

            virtual void then(std::function<void(void)> F) = 0;

            virtual void wait() = 0;
        };

        void wait() { _impl->wait(); }

        Dispatcher &then(std::function<void(void)> F) {
            _impl->then(std::move(F));
            return *this;
        }

        explicit Dispatcher(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

        _NODISCARD Impl *impl_mut() const { return _impl.get(); }

    protected:
        std::unique_ptr<Impl> _impl;
    };
}