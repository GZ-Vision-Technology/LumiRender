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

        virtual void upload(size_t offset, size_t size, const void *host_data) = 0;
        virtual void download(size_t offset, size_t size, void *host_buffer) = 0;
        virtual void clear_cache() = 0;
    };


    template<typename T>
    class BufferView {

    public:
        static constexpr auto npos = std::numeric_limits<size_t>::max();

    private:
        std::shared_ptr<Buffer> _buffer;
        size_t _offset{0u};
        size_t _size{0u};

    public:
        BufferView() noexcept = default;

        explicit BufferView(std::shared_ptr<Buffer> buffer, size_t offset = 0u, size_t size = npos) noexcept: _buffer{std::move(buffer)}, _offset{offset}, _size{size} {
            if (_size == npos) { _size = (_buffer->size() - byte_offset()) / sizeof(T); }
        }

        [[nodiscard]] BufferView subview(size_t offset, size_t size = npos) const noexcept { return BufferView{_buffer, _offset + offset, size}; }

        [[nodiscard]] bool empty() const noexcept { return _buffer == nullptr || _size == 0u; }
        [[nodiscard]] Buffer *buffer() const noexcept { return _buffer.get(); }
        [[nodiscard]] size_t offset() const noexcept { return _offset; }
        [[nodiscard]] size_t size() const noexcept { return _size; }
        [[nodiscard]] size_t byte_offset() const noexcept { return _offset * sizeof(T); }
        [[nodiscard]] size_t byte_size() const noexcept { return _size * sizeof(T); }

        void clear_cache() const noexcept { _buffer->clear_cache(); }

    };

    template<typename T>
    inline auto Buffer::view(size_t offset, size_t size) noexcept {
        return BufferView<T>{shared_from_this(), offset, size};
    }
}