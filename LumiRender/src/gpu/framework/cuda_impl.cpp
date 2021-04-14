//
// Created by Zero on 2021/4/14.
//

#include "cuda_impl.h"

namespace luminous {
    inline namespace gpu {
        // CUDADispatcher
        CUDADispatcher::CUDADispatcher() {
            CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        }

        void CUDADispatcher::wait() {
            CU_CHECK(cuStreamSynchronize(stream));
        }

        void CUDADispatcher::then(std::function<void(void)> F) {
            using Func = std::function<void(void)>;
            Func *f = new Func(std::move(F));
            auto wrapper = [](void *p) {
                auto f = reinterpret_cast<Func *>(p);
                (*f)();
                delete f;
            };
            CU_CHECK(cuLaunchHostFunc(stream, wrapper, (void *) f));
        }

        CUDADispatcher::~CUDADispatcher() {
            CU_CHECK(cuStreamDestroy(stream));
        }

        void *CUDABuffer::ptr() {
            return (void *) _ptr;
        }

        CUDABuffer::CUDABuffer(size_t bytes)
                : _size_in_bytes(bytes) {
            CU_CHECK(cuMemAlloc(&_ptr, bytes));
        }

        CUDABuffer::~CUDABuffer() {
            CU_CHECK(cuMemFree(_ptr));
        }

        size_t CUDABuffer::size() const {
            return _size_in_bytes;
        }

        void *CUDABuffer::address(size_t offset) const {
            return (void *) (_ptr + offset);
        }

        void CUDABuffer::download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            CU_CHECK(cuMemcpyDtoHAsync(host_ptr, _ptr + offset, size, stream));
        }

        void CUDABuffer::upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size, size_t offset) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            CU_CHECK(cuMemcpyHtoDAsync(_ptr + offset, host_ptr, size, stream));
        }

        void CUDABuffer::download(void *host_ptr, size_t size, size_t offset) {
            CU_CHECK(cuMemcpyDtoH(host_ptr, _ptr + offset, size));
        }

        void CUDABuffer::upload(const void *host_ptr, size_t size, size_t offset) {
            CU_CHECK(cuMemcpyHtoD(_ptr + offset, host_ptr, size));
        }

        CUDAKernel::CUDAKernel(CUfunction func)
                : _func(func) {
            compute_fit_size();
        }

        void CUDAKernel::compute_fit_size() {
            cuOccupancyMaxPotentialBlockSize(&_min_grid_size, &_auto_block_size, _func, 0, _shared_mem, 0);
        }

        void CUDAKernel::configure(uint3 grid_size, uint3 local_size, size_t sm) {
            _shared_mem = sm == 0 ? _shared_mem : sm;
            _grid_size = grid_size;
            _block_size = local_size;
        }

        void CUDAKernel::launch(Dispatcher &dispatcher, int n_items, std::vector<void *> &args) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            int grid_size = (n_items + _auto_block_size - 1) / _auto_block_size;
            CU_CHECK(cuLaunchKernel(_func, grid_size, 1, 1,
                                    _auto_block_size, 1, 1,
                                    _shared_mem, stream, args.data(), nullptr));
        }

        void CUDAKernel::launch(Dispatcher &dispatcher, std::vector<void *> &args) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            CU_CHECK(cuLaunchKernel(_func, _grid_size.x, _grid_size.y, _grid_size.z,
                                    _block_size.x, _block_size.y,_block_size.z,
                                    _shared_mem, stream, args.data(), nullptr));
        }

        CUDADevice::CUDADevice() {
            CU_CHECK(cuInit(0));
            CU_CHECK(cuDeviceGet(&_cu_device, 0));
            CU_CHECK(cuCtxCreate(&_cu_context, 0, _cu_device));
            CU_CHECK(cuCtxSetCurrent(_cu_context));
        }

        RawBuffer CUDADevice::allocate_buffer(size_t bytes) {
            return RawBuffer(std::make_unique<CUDABuffer>(bytes));
        }

        Dispatcher CUDADevice::new_dispatcher() {
            return Dispatcher(std::make_unique<CUDADispatcher>());
        }

        CUDADevice::~CUDADevice() {
            CU_CHECK(cuCtxDestroy(_cu_context));
        }

        CUDAModule::CUDAModule(const std::string &ptx_code) {
            CU_CHECK(cuModuleLoadData(&_module, ptx_code.c_str()));
        }

        SP<Kernel> CUDAModule::get_kernel(const std::string &name) {
            CUfunction func;
            CU_CHECK(cuModuleGetFunction(&func, _module, name.c_str()));
            return create_cuda_kernel(func);
        }
    }
}