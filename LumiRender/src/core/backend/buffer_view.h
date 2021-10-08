//
// Created by Zero on 2021/4/10.
//


#pragma once

#include "base_libs/header.h"

namespace luminous {

    NDSC_XPU_INLINE size_t fix_count(size_t offset, size_t count, size_t size) {
        count = count < (size - offset) ? count : (size - offset);
        return count;
    }

    template<typename T = std::byte>
    struct BufferView {
    private:
        // point to device memory or host memory
        T *_ptr;
        // count of elements
        size_t _num;
    public:
        using value_type = T;
        using iterator = T *;
        using const_iterator = const T *;

        XPU BufferView()
                : _ptr(nullptr), _num(0) {}

        XPU BufferView(T *ptr, size_t num)
                : _ptr(ptr), _num(num) {}

        NDSC_XPU T *ptr() { return _ptr; }

        template<typename target_type>
        NDSC_XPU BufferView<target_type> cast() const {
            size_t num = _num * float(sizeof(value_type)) / sizeof(target_type);
            return BufferView<target_type>(reinterpret_cast<target_type *>(_ptr), num);
        }

        NDSC_XPU iterator begin() { return _ptr; }

        NDSC_XPU iterator end() { return _ptr + _num; }

        NDSC_XPU const_iterator cbegin() const { return _ptr; }

        NDSC_XPU const_iterator cend() const { return _ptr + _num; }

        template<typename Index>
        NDSC_XPU T &operator[](Index i) {
            EXE_DEBUG(i >= size(), printf("ptr = %p, size:%d,index:%d\n", _ptr, int(size()), int(i)));
#ifndef NDEBUG
            if (i >= size()) {
                volatile int a = 0;
            }
#endif
            DCHECK_LT(i, size());
            return _ptr[i];
        }

        template<typename Index>
        NDSC_XPU const T &operator[](Index i) const {
            EXE_DEBUG(i >= size(), printf("ptr = %p, size:%d,index:%d\n", _ptr, int(size()), int(i)));
#ifndef NDEBUG
            if (i >= size()) {
                volatile int a = 0;
            }
#endif
            DCHECK_LT(i, size());
            return _ptr[i];
        }

        NDSC_XPU bool empty() const { return _num == 0; }

        NDSC_XPU T front() const { return _ptr[0]; }

        NDSC_XPU T back() const { return _ptr[_num - 1]; }

        NDSC_XPU BufferView sub_view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, size());
            return BufferView(_ptr + offset, count);
        }

        NDSC_XPU size_t size() const { return _num; }
    };
}