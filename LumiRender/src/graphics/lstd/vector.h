//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "common.h"
#include "pmr.h"

namespace lstd {

    template<typename T, class Allocator = pmr::polymorphic_allocator<T>>
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
        Allocator _allocator;
        T *ptr = nullptr;
        size_t _n_alloc = 0, _n_stored = 0;

    public:
        vector(const Allocator &alloc = {}) : _allocator(alloc) {}

        vector(size_t count, const T &value, const Allocator &alloc = {}) : _allocator(alloc) {
            reserve(count);
            for (size_t i = 0; i < count; ++i)
                this->_allocator.template construct<T>(ptr + i, value);
            _n_stored = count;
        }

        vector(size_t count, const Allocator &alloc = {}) : vector(count, T{}, alloc) {}

        vector(const vector &other, const Allocator &alloc = {}) : _allocator(alloc) {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                this->_allocator.template construct<T>(ptr + i, other[i]);
            _n_stored = other.size();
        }

        template<class InputIt>
        vector(InputIt first, InputIt last, const Allocator &alloc = {}) : _allocator(alloc) {
            reserve(last - first);
            size_t i = 0;
            for (InputIt iter = first; iter != last; ++iter, ++i)
                this->_allocator.template construct<T>(ptr + i, *iter);
            _n_stored = _n_alloc;
        }

        vector(vector &&other) : _allocator(other._allocator) {
            _n_stored = other._n_stored;
            _n_alloc = other._n_alloc;
            ptr = other.ptr;

            other._n_stored = other._n_alloc = 0;
            other.ptr = nullptr;
        }

        vector(vector &&other, const Allocator &alloc) {
            if (alloc == other._allocator) {
                ptr = other.ptr;
                _n_alloc = other._n_alloc;
                _n_stored = other._n_stored;

                other.ptr = nullptr;
                other._n_alloc = other._n_stored = 0;
            } else {
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    alloc.template construct<T>(ptr + i, std::move(other[i]));
                _n_stored = other.size();
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
                _allocator.template construct<T>(ptr + i, other[i]);
            _n_stored = other.size();

            return *this;
        }

        vector &operator=(vector &&other) {
            if (this == &other)
                return *this;

            if (_allocator == other._allocator) {
                lstd::swap(ptr, other.ptr);
                lstd::swap(_n_alloc, other._n_alloc);
                lstd::swap(_n_stored, other._n_stored);
            } else {
                clear();
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    _allocator.template construct<T>(ptr + i, std::move(other[i]));
                _n_stored = other.size();
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

        template<class InputIt>
        void assign(InputIt first, InputIt last) {
            assert(0);
            // TODO
        }

        void assign(std::initializer_list<T> &init) { assign(init.begin(), init.end()); }

        ~vector() {
            clear();
            _allocator.deallocate_object(ptr, _n_alloc);
        }

        XPU iterator begin() { return ptr; }

        XPU iterator end() { return ptr + _n_stored; }

        XPU const_iterator begin() const { return ptr; }

        XPU const_iterator end() const { return ptr + _n_stored; }

        XPU const_iterator cbegin() const { return ptr; }

        XPU const_iterator cend() const { return ptr + _n_stored; }

        XPU reverse_iterator rbegin() { return reverse_iterator(end()); }

        XPU reverse_iterator rend() { return reverse_iterator(begin()); }

        XPU const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

        XPU const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        allocator_type get_allocator() const { return _allocator; }

        XPU size_t size() const { return _n_stored; }

        XPU bool empty() const { return size() == 0; }

        XPU size_t max_size() const { return (size_t) -1; }

        XPU size_t capacity() const { return _n_alloc; }

        void reserve(size_t n) {
            if (_n_alloc >= n)
                return;

            T *ra = _allocator.template allocate_object<T>(n);
            for (int i = 0; i < _n_stored; ++i) {
                _allocator.template construct<T>(ra + i, std::move(begin()[i]));
                _allocator.destroy(begin() + i);
            }

            _allocator.deallocate_object(ptr, _n_alloc);
            _n_alloc = n;
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

        XPU reference back() { return ptr[_n_stored - 1]; }

        XPU const_reference back() const { return ptr[_n_stored - 1]; }

        XPU pointer data() { return ptr; }

        XPU const_pointer data() const { return ptr; }

        void clear() {
            for (int i = 0; i < _n_stored; ++i)
                _allocator.destroy(&ptr[i]);
            _n_stored = 0;
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

        template<class InputIt>
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

        template<class... Args>
        iterator emplace(const_iterator pos, Args &&... args) {
            // TODO
            assert(0);
        }

        template<class... Args>
        void emplace_back(Args &&... args) {
            if (_n_alloc == _n_stored)
                reserve(_n_alloc == 0 ? 4 : 2 * _n_alloc);

            _allocator.construct(ptr + _n_stored, std::forward<Args>(args)...);
            ++_n_stored;
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
            if (_n_alloc == _n_stored)
                reserve(_n_alloc == 0 ? 4 : 2 * _n_alloc);

            _allocator.construct(ptr + _n_stored, value);
            ++_n_stored;
        }

        void push_back(T &&value) {
            if (_n_alloc == _n_stored)
                reserve(_n_alloc == 0 ? 4 : 2 * _n_alloc);

            _allocator.construct(ptr + _n_stored, std::move(value));
            ++_n_stored;
        }

        void pop_back() {
            // DCHECK(!empty());
            _allocator.destroy(ptr + _n_stored - 1);
            --_n_stored;
        }

        void resize(size_type n) {
            if (n < size()) {
                for (size_t i = n; i < size(); ++i)
                    _allocator.destroy(ptr + i);
                if (n == 0) {
                    _allocator.deallocate_object(ptr, _n_alloc);
                    ptr = nullptr;
                    _n_alloc = 0;
                }
            } else if (n > size()) {
                reserve(n);
                for (size_t i = _n_stored; i < n; ++i)
                    _allocator.construct(ptr + i);
            }
            _n_stored = n;
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
