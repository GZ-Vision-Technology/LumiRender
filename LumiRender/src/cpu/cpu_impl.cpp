//
// Created by Zero on 26/08/2021.
//

#include "cpu_impl.h"
#include "cpu_scene.h"
#include "util/parallel.h"

namespace luminous {
    inline namespace cpu {

        void *CPUBuffer::ptr() const { return _ptr; }

        CPUBuffer::CPUBuffer(size_t bytes)
                : _size_in_bytes(bytes) {
            DCHECK_GT(bytes, 0)
            _ptr = ::malloc(bytes);
        }

        CPUBuffer::~CPUBuffer() {
            ::free(_ptr);
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
            //todo
            DCHECK(0)
        }

        void CPUBuffer::upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) {
            //todo
            DCHECK(0)
        }

        void CPUBuffer::download(void *host_ptr, size_t size, size_t offset) {
            //todo
            DCHECK(0)
        }

        void CPUBuffer::upload(const void *host_ptr, size_t size, size_t offset) {
            //todo
            DCHECK(0)
        }

        RawBuffer CPUDevice::allocate_buffer(size_t bytes) {
            return RawBuffer(std::make_unique<CPUBuffer>(bytes));
        }

        DTexture CPUDevice::allocate_texture(PixelFormat pixel_format, uint2 resolution) {
            DCHECK(0)
            return DTexture(std::make_unique<CPUTexture>(pixel_format, resolution));
        }

        Dispatcher CPUDevice::new_dispatcher() {
            return Dispatcher(std::make_unique<CPUDispatcher>());
        }


        void CPUKernel::launch(Dispatcher &dispatcher, void **args) {
            parallel_for(1, [&](uint idx, uint tid) {
                _func(args, idx);
            });
        }

        void CPUKernel::launch(Dispatcher &dispatcher, int n_items, void **args) {
            parallel_for(n_items, [&](uint idx, uint tid) {
                _func(args, idx);
            });
        }

        std::shared_ptr<Scene> CPUDevice::create_scene(Device *device, Context *context) {
            return std::make_shared<CPUScene>(device, context);
        }

        CPUTexture::CPUTexture(PixelFormat pixel_format, uint2 resolution) : Impl(pixel_format, resolution) {
            init();
        }

        void CPUTexture::init() {

        }

        uint64_t CPUTexture::tex_handle() const {
            return _handle;
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

        }

        void CPUDispatcher::wait() {

        }

        void CPUDispatcher::then(std::function<void(void)> F) {

        }

        CPUDispatcher::~CPUDispatcher() {

        }
    }
}