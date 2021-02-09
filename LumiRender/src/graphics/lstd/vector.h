//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "common.h"
#include "memory_resource.h"

namespace luminous {
    namespace lstd {

        template <typename T, class Allocator = polymorphic_allocator<T>>
        class vector {
        public:
            using value_type = T;
            using allocator_type = Allocator;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using reference = value_type &;
            using const_reference = const value_type &;
            using pointer = T *;
            using const_pointer = const T *;
            using iterator = T *;
            using const_iterator = const T *;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const iterator>;

            
        private:
            Allocator alloc;
            T *ptr = nullptr;
            size_t nAlloc = 0, nStored = 0;

        public:
            vector(const Allocator &alloc = {}) : alloc(alloc) {}
            vector(size_t count, const T &value, const Allocator &alloc = {}) : alloc(alloc) {
                reserve(count);
                for (size_t i = 0; i < count; ++i)
                    this->alloc.template construct<T>(ptr + i, value);
                nStored = count;
            }
            vector(size_t count, const Allocator &alloc = {}) : vector(count, T{}, alloc) {}
            vector(const vector &other, const Allocator &alloc = {}) : alloc(alloc) {
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    this->alloc.template construct<T>(ptr + i, other[i]);
                nStored = other.size();
            }
            template <class InputIt>
            vector(InputIt first, InputIt last, const Allocator &alloc = {}) : alloc(alloc) {
                reserve(last - first);
                size_t i = 0;
                for (InputIt iter = first; iter != last; ++iter, ++i)
                    this->alloc.template construct<T>(ptr + i, *iter);
                nStored = nAlloc;
            }
            vector(vector &&other) : alloc(other.alloc) {
                nStored = other.nStored;
                nAlloc = other.nAlloc;
                ptr = other.ptr;

                other.nStored = other.nAlloc = 0;
                other.ptr = nullptr;
            }
            vector(vector &&other, const Allocator &alloc) {
                if (alloc == other.alloc) {
                    ptr = other.ptr;
                    nAlloc = other.nAlloc;
                    nStored = other.nStored;

                    other.ptr = nullptr;
                    other.nAlloc = other.nStored = 0;
                } else {
                    reserve(other.size());
                    for (size_t i = 0; i < other.size(); ++i)
                        alloc.template construct<T>(ptr + i, std::move(other[i]));
                    nStored = other.size();
                }
            }
            vector(std::initializer_list<T> init, const Allocator &alloc = {})
                : vector(init.begin(), init.end(), alloc) {}

            vector &operator=(const vector &other) {
                if (this == &other)
                    return *this;

                clear();
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    alloc.template construct<T>(ptr + i, other[i]);
                nStored = other.size();

                return *this;
            }
            vector &operator=(vector &&other) {
                if (this == &other)
                    return *this;

                if (alloc == other.alloc) {
                    pstd::swap(ptr, other.ptr);
                    pstd::swap(nAlloc, other.nAlloc);
                    pstd::swap(nStored, other.nStored);
                } else {
                    clear();
                    reserve(other.size());
                    for (size_t i = 0; i < other.size(); ++i)
                        alloc.template construct<T>(ptr + i, std::move(other[i]));
                    nStored = other.size();
                }

                return *this;
            }
            vector &operator=(std::initializer_list<T> &init) {
                reserve(init.size());
                clear();
                iterator iter = begin();
                for (const auto &value : init) {
                    *iter = value;
                    ++iter;
                }
                return *this;
            }

            void assign(size_type count, const T &value) {
                clear();
                reserve(count);
                for (size_t i = 0; i < count; ++i)
                    push_back(value);
            }
            template <class InputIt>
            void assign(InputIt first, InputIt last) {
                assert(0);
                // TODO
            }
            void assign(std::initializer_list<T> &init) { assign(init.begin(), init.end()); }

            ~vector() {
                clear();
                alloc.deallocate_object(ptr, nAlloc);
            }

            XPU iterator begin() { return ptr; }
            XPU iterator end() { return ptr + nStored; }
            XPU const_iterator begin() const { return ptr; }
            XPU const_iterator end() const { return ptr + nStored; }
            XPU const_iterator cbegin() const { return ptr; }
            XPU const_iterator cend() const { return ptr + nStored; }

            XPU reverse_iterator rbegin() { return reverse_iterator(end()); }
            XPU reverse_iterator rend() { return reverse_iterator(begin()); }
            XPU const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
            XPU const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

            allocator_type get_allocator() const { return alloc; }
            XPU size_t size() const { return nStored; }
            XPU bool empty() const { return size() == 0; }
            XPU size_t max_size() const { return (size_t)-1; }
            XPU size_t capacity() const { return nAlloc; }
            void reserve(size_t n) {
                if (nAlloc >= n)
                    return;

                T *ra = alloc.template allocate_object<T>(n);
                for (int i = 0; i < nStored; ++i) {
                    alloc.template construct<T>(ra + i, std::move(begin()[i]));
                    alloc.destroy(begin() + i);
                }

                alloc.deallocate_object(ptr, nAlloc);
                nAlloc = n;
                ptr = ra;
            }
            // TODO: shrink_to_fit

            XPU reference operator[](size_type index) {
                // DCHECK_LT(index, size());
                return ptr[index];
            }
            XPU const_reference operator[](size_type index) const {
                // DCHECK_LT(index, size());
                return ptr[index];
            }
            XPU reference front() { return ptr[0]; }
            XPU const_reference front() const { return ptr[0]; }
            XPU reference back() { return ptr[nStored - 1]; }
            XPU const_reference back() const { return ptr[nStored - 1]; }
            XPU pointer data() { return ptr; }
            XPU const_pointer data() const { return ptr; }

            void clear() {
                for (int i = 0; i < nStored; ++i)
                    alloc.destroy(&ptr[i]);
                nStored = 0;
            }

            iterator insert(const_iterator, const T &value) {
                // TODO
                assert(0);
            }
            iterator insert(const_iterator, T &&value) {
                // TODO
                assert(0);
            }
            iterator insert(const_iterator pos, size_type count, const T &value) {
                // TODO
                assert(0);
            }
            template <class InputIt>
            iterator insert(const_iterator pos, InputIt first, InputIt last) {
                if (pos == end()) {
                    size_t firstOffset = size();
                    for (auto iter = first; iter != last; ++iter)
                        push_back(*iter);
                    return begin() + firstOffset;
                } else
                    assert(0);
            }
            iterator insert(const_iterator pos, std::initializer_list<T> init) {
                // TODO
                assert(0);
            }

            template <class... Args>
            iterator emplace(const_iterator pos, Args &&... args) {
                // TODO
                assert(0);
            }
            template <class... Args>
            void emplace_back(Args &&... args) {
                if (nAlloc == nStored)
                    reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

                alloc.construct(ptr + nStored, std::forward<Args>(args)...);
                ++nStored;
            }

            iterator erase(const_iterator pos) {
                // TODO
                assert(0);
            }
            iterator erase(const_iterator first, const_iterator last) {
                // TODO
                assert(0);
            }

            void push_back(const T &value) {
                if (nAlloc == nStored)
                    reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

                alloc.construct(ptr + nStored, value);
                ++nStored;
            }
            void push_back(T &&value) {
                if (nAlloc == nStored)
                    reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

                alloc.construct(ptr + nStored, std::move(value));
                ++nStored;
            }
            void pop_back() {
                // DCHECK(!empty());
                alloc.destroy(ptr + nStored - 1);
                --nStored;
            }

            void resize(size_type n) {
                if (n < size()) {
                    for (size_t i = n; i < size(); ++i)
                        alloc.destroy(ptr + i);
                    if (n == 0) {
                        alloc.deallocate_object(ptr, nAlloc);
                        ptr = nullptr;
                        nAlloc = 0;
                    }
                } else if (n > size()) {
                    reserve(n);
                    for (size_t i = nStored; i < n; ++i)
                        alloc.construct(ptr + i);
                }
                nStored = n;
            }
            void resize(size_type count, const value_type &value) {
                // TODO
                assert(0);
            }

            void swap(vector &other) {
                // TODO
                assert(0);
            }

        };
    }
}