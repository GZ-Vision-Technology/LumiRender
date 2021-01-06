//
// Created by Zero on 2021/1/6.
//


#pragma once

#include <core/concepts.h>
#include <iostream>
#include <limits>
#include <functional>
#include <utility>

namespace luminous::backend {
    class Buffer : public Noncopyable, public std::enable_shared_from_this<Buffer>{
    public:
        static constexpr auto npos = std::numeric_limits<size_t>::max();

    protected:
        size_t _size;

    public:
        explicit Buffer(size_t size) noexcept
                : _size{size} {}
        virtual ~Buffer() noexcept = default;

        [[nodiscard]] size_t size() const noexcept { return _size; }

        template<typename T>
        [[nodiscard]] auto view(size_t offset = 0u, size_t size = npos) noexcept;

        virtual void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) = 0;
        virtual void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) = 0;
    };
}