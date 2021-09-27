//
// Created by Zero on 2021/2/6.
//


#pragma once

#include "constants.h"

namespace luminous {
    inline namespace math {
        template<typename T>
        struct interval {
            using scalar_t = T;

            T begin;
            T end;

            inline XPU interval()
                    : begin(empty_bounds_lower<T>()),
                      end(empty_bounds_upper<T>()) {}

            inline XPU interval(T begin, T end) : begin(begin), end(end) {}

            NDSC_XPU bool contains(const T &t) const { return t >= begin && t <= end; }

            NDSC_XPU bool is_empty() const { return begin > end; }

            NDSC_XPU T center() const { return (begin + end) / 2; }

            NDSC_XPU T span() const { return end - begin; }

            NDSC_XPU T diagonal() const { return end - begin; }

            NDSC_XPU interval<T> &extend(const T &t) {
                begin = min(begin, t);
                end = max(end, t);
                return *this;
            }

            NDSC_XPU interval<T> &extend(const interval<T> &t) {
                begin = min(begin, t.begin);
                end = max(end, t.end);
                return *this;
            }

            NDSC_XPU interval<T> including(const T &t) { return interval<T>(min(begin, t), max(end, t)); }

            NDSC_XPU static interval<T> positive() {
                return interval<T>(0.f, open_range_upper<T>());
            }
        };

        template<typename T>
        NDSC_XPU_INLINE interval<T> build_interval(const T &a, const T &b) { return interval<T>(min(a, b), max(a, b)); }

        template<typename T>
        NDSC_XPU_INLINE interval<T> intersect(const interval<T> &a, const interval<T> &b) {
            return interval<T>(max(a.begin, b.begin), min(a.end, b.end));
        }

        template<typename T>
        NDSC_XPU_INLINE interval<T> operator-(const interval<T> &a, const T &b) {
            return interval<T>(a.begin - b, a.end - b);
        }

        template<typename T>
        NDSC_XPU_INLINE interval<T> operator*(const interval<T> &a, const T &b) {
            return build_interval<T>(a.begin * b, a.end * b);
        }

        template<typename T>
        NDSC_XPU_INLINE bool operator==(const interval<T> &a, const interval<T> &b) {
            return a.begin == b.begin && a.end == b.end;
        }

        template<typename T>
        NDSC_XPU_INLINE bool operator!=(const interval<T> &a, const interval<T> &b) { return !(a == b); }

        using PtrInterval = interval<uint64_t>;
    } // luminous::math
} // luminous