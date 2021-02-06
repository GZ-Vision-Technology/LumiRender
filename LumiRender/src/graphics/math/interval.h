//
// Created by Zero on 2021/2/6.
//


#pragma once

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

            inline XPU bool contains(const T &t) const { return t >= lower && t <= upper; }

            inline XPU bool is_empty() const { return begin > end; }

            inline XPU T center() const { return (begin + end) / 2; }

            inline XPU T span() const { return end - begin; }

            inline XPU T diagonal() const { return end - begin; }

            inline XPU interval<T> &extend(const T &t) {
                lower = min(lower, t);
                upper = max(upper, t);
                return *this;
            }

            inline XPU interval<T> &extend(const interval<T> &t) {
                lower = min(lower, t.lower);
                upper = max(upper, t.upper);
                return *this;
            }

            inline XPU interval<T> including(const T &t) { return interval<T>(min(lower, t), max(upper, t)); }

            static inline XPU interval<T> positive() {
                return interval<T>(0.f, open_range_upper<T>());
            }
        };

        template<typename T>
        inline XPU interval<T> build_interval(const T &a, const T &b) { return interval<T>(min(a, b), max(a, b)); }

        template<typename T>
        inline XPU interval<T> intersect(const interval<T> &a, const interval<T> &b) {
            return interval<T>(max(a.lower, b.lower), min(a.upper, b.upper));
        }

        template<typename T>
        inline XPU interval<T> operator-(const interval<T> &a, const T &b) {
            return interval<T>(a.lower - b, a.upper - b);
        }

        template<typename T>
        inline XPU interval<T> operator*(const interval<T> &a, const T &b) {
            return build_interval<T>(a.lower * b, a.upper * b);
        }

        template<typename T>
        inline XPU bool operator==(const interval<T> &a, const interval<T> &b) {
            return a.lower == b.lower && a.upper == b.upper;
        }

        template<typename T>
        inline XPU bool operator!=(const interval<T> &a, const interval<T> &b) { return !(a == b); }
    } // luminous::math
} // luminous