//
// Created by Zero on 2021/1/31.
//


#pragma once

#include <typeinfo>
#include <variant>
#include "../math/vector_types.h"
#include "../header.h"

namespace luminous {
    inline namespace lstd {

        template<typename T>
        LM_XPU inline void luminous_swap(T &a, T &b) {
            T tmp = std::move(a);
            a = std::move(b);
            b = std::move(tmp);
        }

        template<typename T>
        void append(T &v1, const T &v2) {
            v1.insert(v1.cend(), v2.cbegin(), v2.cend());
        }

        template<typename Iter, typename Predict>
        LM_ND_XPU Iter find_if(const Iter begin, const Iter end, Predict predict) {
            Iter iter;
            for (iter = begin; iter != end; ++iter) {
                if (predict(*iter)) {
                    break;
                }
            }
            return iter;
        }

        /**
         * @tparam T iterator
         * @tparam Predict
         * @param v
         * @param predict
         * @return element index , if not found return -1
         */
        template<typename T, typename Predict>
        LM_ND_XPU int64_t find_index_if(const T &v, Predict predict) {
            auto iter = lstd::find_if(v.cbegin(), v.cend(), predict);
            if (iter == v.cend()) {
                return -1;
            }
            return iter - v.cbegin();
        }

        LM_ND_XPU inline constexpr size_t max(size_t a, size_t b) { return a < b ? b : a; }

        struct nullopt_t {
        };
        inline constexpr nullopt_t nullopt{};

        template<typename T>
        class optional {
        private:
            LM_XPU T *ptr() { return reinterpret_cast<T *>(&_optional_value); }

            LM_XPU const T *ptr() const { return reinterpret_cast<const T *>(&_optional_value); }

            T _optional_value;
            bool set = false;
        public:
            using value_type = T;
            LM_XPU optional(nullopt_t) : optional() {}

            optional() = default;

            LM_XPU optional(const T &v) : set(true) { new(ptr()) T(v); }

            LM_XPU optional(T &&v) : set(true) { new(ptr()) T(std::move(v)); }

            LM_XPU optional(const optional &v) : set(v.has_value()) {
                if (v.has_value()) { new(ptr()) T(v.value()); }
            }

            LM_XPU optional(optional &&v) : set(v.has_value()) {
                if (v.has_value()) {
                    new(ptr()) T(std::move(v.value()));
                    v.reset();
                }
            }

            LM_XPU optional &operator=(const T &v) {
                reset();
                new(ptr()) T(v);
                set = true;
                return *this;
            }

            LM_XPU optional &operator=(T &&v) {
                reset();
                new(ptr()) T(std::move(v));
                set = true;
                return *this;
            }

            LM_XPU optional &operator=(const optional &v) {
                reset();
                if (v.has_value()) {
                    new(ptr()) T(v.value());
                    set = true;
                }
                return *this;
            }

            template<typename... Ts>
            LM_XPU void emplace(Ts &&...args) {
                reset();
                new(ptr()) T(std::forward<Ts>(args)...);
                set = true;
            }

            LM_XPU optional &operator=(optional &&v) {
                reset();
                if (v.has_value()) {
                    new(ptr()) T(std::move(v.value()));
                    set = true;
                    v.reset();
                }
                return *this;
            }

            LM_XPU ~optional() { reset(); }

            LM_XPU explicit operator bool() const { return set; }

            LM_XPU T value_or(const T &alt) const { return set ? value() : alt; }

            LM_XPU T *operator->() { return &value(); }

            LM_XPU const T *operator->() const { return &value(); }

            LM_XPU T &operator*() { return value(); }

            LM_XPU const T &operator*() const { return value(); }

            LM_XPU T &value() {
                DCHECK(set);
                return *ptr();
            }

            LM_XPU const T &value() const {
                DCHECK(set);
                return *ptr();
            }

            LM_XPU void reset() {
                if (set) {
                    value().~T();
                    set = false;
                }
            }

            LM_XPU bool has_value() const { return set; }
        };

        template<typename T, int N>
        class array;

        template<typename T>
        class array<T, 0> {
        public:
            using value_type = T;
            using iterator = value_type *;
            using const_iterator = const value_type *;
            using size_t = std::size_t;

            LM_XPU array() = default;

            LM_XPU void fill(const T &v) { DCHECK(!"should never be called"); }

            LM_ND_XPU bool operator==(const array<T, 0> &a) const { return true; }

            LM_ND_XPU bool operator!=(const array<T, 0> &a) const { return false; }

            LM_ND_XPU iterator begin() { return nullptr; }

            LM_ND_XPU iterator end() { return nullptr; }

            LM_ND_XPU const_iterator begin() const { return nullptr; }

            LM_ND_XPU const_iterator end() const { return nullptr; }

            LM_ND_XPU size_t size() const { return 0; }

            LM_ND_XPU T &operator[](size_t i) {
                DCHECK(!"should never be called");
                static T t;
                return t;
            }

            LM_ND_XPU const T &operator[](size_t i) const {
                DCHECK(!"should never be called");
                static T t;
                return t;
            }

            LM_ND_XPU T *data() { return nullptr; }

            LM_ND_XPU const T *data() const { return nullptr; }
        };

        template<typename T, int N>
        class array {
        private:
            T values[N] = {};
        public:
            using value_type = T;
            using iterator = value_type *;
            using const_iterator = const value_type *;
            using size_t = std::size_t;

            array() = default;

            LM_XPU array(std::initializer_list<T> v) {
                size_t i = 0;
                for (const T &val : v)
                    values[i++] = val;
            }

            LM_XPU void fill(const T &v) {
                for (int i = 0; i < N; ++i) { values[i] = v; }
            }

            LM_ND_XPU bool operator==(const array<T, N> &a) const {
                for (int i = 0; i < N; ++i) {
                    if (values[i] != a.values[i]) {
                        return false;
                    }
                }
                return true;
            }

            LM_ND_XPU bool operator!=(const array<T, N> &a) const { return !(*this == a); }

            LM_ND_XPU iterator begin() { return values; }

            LM_ND_XPU iterator end() { return values + N; }

            LM_ND_XPU const_iterator begin() const { return values; }

            LM_ND_XPU const_iterator end() const { return values + N; }

            LM_ND_XPU size_t size() const { return N; }

            LM_ND_XPU T &operator[](size_t i) { return values[i]; }

            LM_ND_XPU const T &operator[](size_t i) const { return values[i]; }

            LM_ND_XPU T *data() { return values; }

            LM_ND_XPU const T *data() const { return values; }

        };

        template<typename T, int Row, int Col = Row>
        class Array2D {
        private:
            T value[Row][Col]{};
        public:
            using value_type = T;
            using iterator = value_type *;
            using const_iterator = const value_type *;
            using size_t = std::size_t;
            static constexpr int row = Row;
            static constexpr int col = Col;
        public:
            Array2D() = default;

            LM_XPU void fill(const T *array) {
                std::memcpy(value, array, size() * sizeof(T));
            }

            LM_XPU void fill(const T array[row][col]) {
                for (int i = 0; i < row; ++i) {
                    for (int j = 0; j < col; ++j) {
                        value[i][j] = array[i][j];
                    }
                }
            }

            LM_ND_XPU static constexpr int size() { return row * col; }

            LM_ND_XPU bool operator==(const Array2D<T, Row, Col> &array) const {
                for (int i = 0; i < row; ++i) {
                    for (int j = 0; j < col; ++j) {
                        if (value[i][j] != array[i][j]) {
                            return false;
                        }
                    }
                }
                return true;
            }

            LM_ND_XPU bool operator!=(const Array2D<T, Row, Col> &array) const { return !(*this == array); }

            LM_ND_XPU iterator begin() { return value; }

            LM_ND_XPU iterator end() { return begin() + size(); }

            LM_ND_XPU const_iterator begin() const { return reinterpret_cast<const_iterator>(value); }

            LM_ND_XPU const_iterator end() const { return begin() + size(); }

            LM_ND_XPU const_iterator cbegin() const { return reinterpret_cast<const_iterator>(value); }

            LM_ND_XPU const_iterator cend() const { return cbegin() + size(); }

            LM_ND_XPU value_type operator()(int x, int y) const { return value[y][x]; }

            LM_ND_XPU value_type operator()(int2 coord) const { return (*this)(coord.x, coord.y); }

            LM_ND_XPU value_type& operator()(int x, int y) { return value[y][x]; }

            LM_ND_XPU value_type& operator()(int2 coord) { return (*this)(coord.x, coord.y); }

            LM_ND_XPU const_iterator operator[](int y) const { return value[y]; }

            LM_ND_XPU iterator operator[](int y) { return value[y]; }
        };
    } // lstd
}