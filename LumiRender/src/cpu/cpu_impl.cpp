//
// Created by Zero on 26/08/2021.
//

#include "cpu_impl.h"

namespace luminous {
    inline namespace cpu {

        void *CPUBuffer::ptr() const { return _ptr; }

        CPUBuffer::CPUBuffer(size_t bytes, void *ptr)
                : _size_in_bytes(bytes),
                  _is_external_ptr(bool(ptr)),
                  _ptr(ptr ? ptr : ::malloc(bytes)) {
            if (!_is_external_ptr) {
                this->memset(0);
            }
        }

        CPUBuffer::~CPUBuffer() {
            if (!_is_external_ptr) {
                ::free(_ptr);
            }
            _ptr = nullptr;
        }

        size_t CPUBuffer::size() const { return _size_in_bytes; }

        void *CPUBuffer::address(size_t offset) const {
            return reinterpret_cast<char *>(_ptr) + offset;
        }

        void CPUBuffer::memset(uint32_t val) {
            ::memset(_ptr, val, _size_in_bytes);
        }

        void CPUBuffer::download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) {
            if (host_ptr == _ptr) { return; }
            async(1, [&](uint, uint) {
                download(host_ptr, size, offset);
            });
        }

        void CPUBuffer::upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) {
            if (host_ptr == _ptr) { return; }
            async(1, [&](uint, uint) {
                upload(host_ptr, size, offset);
            });
        }

        void CPUBuffer::download(void *host_ptr, size_t size, size_t offset) {
            if (host_ptr == _ptr) { return; }
            ::memcpy(host_ptr, (void *) (reinterpret_cast<uint64_t>(_ptr) + offset), size);
        }

        void CPUBuffer::upload(const void *host_ptr, size_t size, size_t offset) {
            if (host_ptr == _ptr) { return; }
            ::memcpy((void *) (reinterpret_cast<uint64_t>(_ptr) + offset), host_ptr, size);
        }

        RawBuffer CPUDevice::create_buffer(size_t bytes, void *ptr) {
            return RawBuffer(std::make_unique<CPUBuffer>(bytes, ptr));
        }

        DTexture CPUDevice::allocate_texture(PixelFormat pixel_format, uint2 resolution) {
            return DTexture(std::make_unique<CPUTexture>(pixel_format, resolution));
        }

        Dispatcher CPUDevice::new_dispatcher() {
            return Dispatcher(std::make_unique<CPUDispatcher>());
        }

        CPUTexture::CPUTexture(PixelFormat pixel_format, uint2 resolution)
                : Impl(pixel_format, resolution) {
            init();
        }

        void CPUTexture::init() {

        }

        uint64_t CPUTexture::tex_handle() const {
            return reinterpret_cast<uint64_t>(&_mipmap);
        }

        void CPUTexture::copy_from(Dispatcher &dispatcher, const Image &image) {

        }

        Image CPUTexture::download() const {
            return Image();
        }

        void CPUTexture::copy_from(const Image &image) {

        }

        CPUTexture::~CPUTexture() {

        }

        CPUDispatcher::CPUDispatcher() {
            init_thread_pool();
            _work_pool = work_pool();
        }

        void CPUDispatcher::wait() {
            _work_pool->wait();
        }

        void CPUDispatcher::then(std::function<void(void)> F) {
            // todo
        }

        CPUDispatcher::~CPUDispatcher() {

        }
    }
}