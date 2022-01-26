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

            inline LM_XPU interval()
                    : begin(empty_bounds_lower<T>()),
                      end(empty_bounds_upper<T>()) {}

            inline LM_XPU interval(T begin, T end) : begin(begin), end(end) {}

            LM_ND_XPU bool contains(const T &t) const { return t >= begin && t <= end; }

            LM_ND_XPU bool is_empty() const { return begin > end; }

            LM_ND_XPU T center() const { return (begin + end) / 2; }

            LM_ND_XPU T span() const { return end - begin; }

            LM_ND_XPU T diagonal() const { return end - begin; }

            LM_ND_XPU interval<T> &extend(const T &t) {
                begin = min(begin, t);
                end = max(end, t);
                return *this;
            }

            LM_ND_XPU interval<T> &extend(const interval<T> &t) {
                begin = min(begin, t.begin);
                end = max(end, t.end);
                return *this;
            }

            LM_ND_XPU interval<T> including(const T &t) { return interval<T>(std::min(begin, t), std::max(end, t)); }

            LM_ND_XPU static interval<T> positive() {
                return interval<T>(0.f, open_range_upper<T>());
            }
        };

        template<typename T>
        ND_XPU_INLINE interval<T> build_interval(const T &a, const T &b) { return interval<T>(std::min(a, b), std::max(a, b)); }

        template<typename T>
        ND_XPU_INLINE interval<T> intersect(const interval<T> &a, const interval<T> &b) {
            return interval<T>(max(a.begin, b.begin), min(a.end, b.end));
        }

        template<typename T>
        ND_XPU_INLINE interval<T> operator-(const interval<T> &a, const T &b) {
            return interval<T>(a.begin - b, a.end - b);
        }

        template<typename T>
        ND_XPU_INLINE interval<T> operator*(const interval<T> &a, const T &b) {
            return build_interval<T>(a.begin * b, a.end * b);
        }

        template<typename T>
        ND_XPU_INLINE bool operator==(const interval<T> &a, const interval<T> &b) {
            return a.begin == b.begin && a.end == b.end;
        }

        template<typename T>
        ND_XPU_INLINE bool operator!=(const interval<T> &a, const interval<T> &b) { return !(a == b); }

        using PtrInterval = interval<uint64_t>;
    } // luminous::math
} // luminous