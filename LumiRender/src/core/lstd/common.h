//
// Created by Zero on 2021/1/31.
//


#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <list>
#include <new>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <cstring>
#include "core/header.h"

namespace luminous {
    namespace lstd {

        template <typename T>
        XPU inline void swap(T &a, T &b) {
            T tmp = std::move(a);
            a = std::move(b);
            b = std::move(tmp);
        }

        template <class To, class From>
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
        public:
            using value_type = T;
            XPU optional(nullopt_t) : optional() {}

            optional() = default;

            XPU optional(const T &v) : set(true) { new(ptr()) T(v); }

            XPU optional(T &&v) : set(true) { new(ptr()) T(std::move(v)); }

            XPU optional(const optional &v) : set(v.has_value()) {
                if (v.has_value())
                    new(ptr()) T(v.value());
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
            CPU void emplace(Ts &&...args) {
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
                AKR_CHECK(set);
                return *ptr();
            }

            XPU const T &value() const {
                AKR_CHECK(set);
                return *ptr();
            }

            XPU void reset() {
                if (set) {
                    value().~T();
                    set = false;
                }
            }

            XPU bool has_value() const { return set; }

        private:
            // #ifdef __NVCC__
            // Work-around NVCC bug
            XPU T *ptr() { return reinterpret_cast<T *>(&optionalValue); }

            XPU const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
            // #else
            //         XPU
            //         T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
            //         XPU
            //         const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
            // #endif

            std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
            bool set = false;
        };

        template <typename T, int N>
        class array;

    } // luminous::lstd
} // luminous