//
// Created by Zero on 26/08/2021.
//

#include "cpu_impl.h"

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
    }
}