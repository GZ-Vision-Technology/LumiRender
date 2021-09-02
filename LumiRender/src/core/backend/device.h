//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "dispatcher.h"
#include "buffer.h"
#include "texture.h"
#include "core/context.h"

namespace luminous {

    inline namespace render {
        class Scene;
    }

    class Device : public Noncopyable, std::enable_shared_from_this<Device> {
    public:
        class Impl {
        public:
            virtual RawBuffer allocate_buffer(size_t bytes) = 0;

            virtual DTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) = 0;

            NDSC virtual std::shared_ptr<Scene>
            create_scene(const std::shared_ptr<Device> &device, Context *context) = 0;

            virtual Dispatcher new_dispatcher() = 0;

            virtual ~Impl() = default;
        };

        template<typename T = std::byte>
        Buffer<T> allocate_buffer(size_t n_elements) {
            return Buffer<T>(_impl->allocate_buffer(n_elements * sizeof(T)));
        }

        NDSC std::shared_ptr<Scene> create_scene(Context *context) {
            return _impl->create_scene(shared_from_this(), context);
        }

        template<typename T = std::byte>
        BufferView<T> obtain_buffer(size_t n_elements) {
            RawBuffer raw_buffer = _impl->allocate_buffer(n_elements * sizeof(T));
            BufferView<T> ret(raw_buffer.ptr<T *>(), n_elements);
            _raw_buffers.push_back(std::move(raw_buffer));
            return ret;
        }

        template<typename T = std::byte>
        T *obtain_restrict_ptr(size_t n_elements) {
            RawBuffer raw_buffer = _impl->allocate_buffer(n_elements * sizeof(T));
            T *ret = raw_buffer.ptr<T *>();
            _raw_buffers.push_back(std::move(raw_buffer));
            return ret;
        }

        DTexture &allocate_texture(PixelFormat pixel_format, uint2 resolution) {
            size_t idx = _textures.size();
            DTexture texture = _impl->allocate_texture(pixel_format, resolution);
            _textures.push_back(std::move(texture));
            return _textures[idx];
        }

        Dispatcher new_dispatcher() { return _impl->new_dispatcher(); }

        explicit Device(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    protected:
        std::unique_ptr<Impl> _impl;
        std::vector<DTexture> _textures;
        std::vector<RawBuffer> _raw_buffers;
    };
}