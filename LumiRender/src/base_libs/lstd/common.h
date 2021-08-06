//
// Created by Zero on 2021/1/31.
//


#pragma once

#include <typeinfo>
#include <variant>
#include <utility>
#include "../header.h"

namespace lstd {

    template<typename T>
    XPU inline void luminous_swap(T &a, T &b) {
        T tmp = std::move(a);
        a = std::move(b);
        b = std::move(tmp);
    }

    template<typename Iter, typename Predict>
    NDSC_XPU Iter find_if(const Iter begin, const Iter end, Predict predict) {
        Iter iter;
        for(iter = begin; iter != end; ++iter) {
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
    NDSC_XPU int64_t find_index_if(const T& v, Predict predict) {
        auto iter = lstd::find_if(v.cbegin(), v.cend(), predict);
        if (iter == v.cend()) {
            return -1;
        }
        return iter - v.cbegin();
    }

    NDSC_XPU inline constexpr size_t max(size_t a, size_t b) { return a < b ? b : a; }

    template<typename T1, typename T2>
    struct alignas(max(alignof(T1), alignof(T2))) pair {
        T1 first;
        T2 second;
    };
    template<typename T1, typename T2>
    pair(T1 &&, T2 &&) -> pair<T1, T2>;

    template<typename T1, typename T2>
    XPU pair<T1, T2> make_pair(T1 a, T2 b) {
        return pair<T1, T2>{a, b};
    }

    template<class To, class From>
    XPU typename std::enable_if_t<sizeof(To) == sizeof(From) &&
                                  std::is_trivially_copyable_v<From> &&
                                  std::is_trivially_copyable_v<To>,
            To>
    bit_cast(const From &src) noexcept {
        static_assert(std::is_trivially_constructible_v<To>,
                      "This implementation requires the destination type to be trivially "
                      "constructible");
        To dst;
        std::memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    struct nullopt_t {
    };
    inline constexpr nullopt_t nullopt{};

    template<typename T>
    class optional {
    private:
        XPU T *ptr() { return reinterpret_cast<T *>(&_optional_value); }

        XPU const T *ptr() const { return reinterpret_cast<const T *>(&_optional_value); }

        T _optional_value;
        bool set = false;
    public:
        using value_type = T;
        XPU optional(nullopt_t) : optional() {}

        optional() = default;

        XPU optional(const T &v) : set(true) { new(ptr()) T(v); }

        XPU optional(T &&v) : set(true) { new(ptr()) T(std::move(v)); }

        XPU optional(const optional &v) : set(v.has_value()) {
            if (v.has_value()) { new(ptr()) T(v.value()); }
        }

        XPU optional(optional &&v) : set(v.has_value()) {
            if (v.has_value()) {
                new(ptr()) T(std::move(v.value()));
                v.reset();
            }
        }

        XPU optional &operator=(const T &v) {
            reset();
            new(ptr()) T(v);
            set = true;
            return *this;
        }

        XPU optional &operator=(T &&v) {
            reset();
            new(ptr()) T(std::move(v));
            set = true;
            return *this;
        }

        XPU optional &operator=(const optional &v) {
            reset();
            if (v.has_value()) {
                new(ptr()) T(v.value());
                set = true;
            }
            return *this;
        }

        template<typename... Ts>
        XPU void emplace(Ts &&...args) {
            reset();
            new(ptr()) T(std::forward<Ts>(args)...);
            set = true;
        }

        XPU optional &operator=(optional &&v) {
            reset();
            if (v.has_value()) {
                new(ptr()) T(std::move(v.value()));
                set = true;
                v.reset();
            }
            return *this;
        }

        XPU ~optional() { reset(); }

        XPU explicit operator bool() const { return set; }

        XPU T value_or(const T &alt) const { return set ? value() : alt; }

        XPU T *operator->() { return &value(); }

        XPU const T *operator->() const { return &value(); }

        XPU T &operator*() { return value(); }

        XPU const T &operator*() const { return value(); }

        XPU T &value() {
            DCHECK(set);
            return *ptr();
        }

        XPU const T &value() const {
            DCHECK(set);
            return *ptr();
        }

        XPU void reset() {
            if (set) {
                value().~T();
                set = false;
            }
        }

        XPU bool has_value() const { return set; }
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

        array() = default;

        XPU void fill(const T &v) { DCHECK(!"should never be called"); }

        XPU bool operator==(const array<T, 0> &a) const { return true; }

        XPU bool operator!=(const array<T, 0> &a) const { return false; }

        XPU iterator begin() { return nullptr; }

        XPU iterator end() { return nullptr; }

        XPU const_iterator begin() const { return nullptr; }

        XPU const_iterator end() const { return nullptr; }

        XPU size_t size() const { return 0; }

        XPU T &operator[](size_t i) {
            DCHECK(!"should never be called");
            static T t;
            return t;
        }

        XPU const T &operator[](size_t i) const {
            DCHECK(!"should never be called");
            static T t;
            return t;
        }

        XPU T *data() { return nullptr; }

        XPU const T *data() const { return nullptr; }
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

        XPU array(std::initializer_list<T> v) {
            size_t i = 0;
            for (const T &val : v)
                values[i++] = val;
        }

        XPU void fill(const T &v) {
            for (int i = 0; i < N; ++i) { values[i] = v; }
        }

        XPU bool operator==(const array<T, N> &a) const {
            for (int i = 0; i < N; ++i) {
                if (values[i] != a.values[i]) {
                    return false;
                }
            }
            return true;
        }

        XPU bool operator!=(const array<T, N> &a) const { return !(*this == a); }

        XPU iterator begin() { return values; }

        XPU iterator end() { return values + N; }

        XPU const_iterator begin() const { return values; }

        XPU const_iterator end() const { return values + N; }

        XPU size_t size() const { return N; }

        XPU T &operator[](size_t i) { return values[i]; }

        XPU const T &operator[](size_t i) const { return values[i]; }

        XPU T *data() { return values; }

        XPU const T *data() const { return values; }

    };

} // lstd
