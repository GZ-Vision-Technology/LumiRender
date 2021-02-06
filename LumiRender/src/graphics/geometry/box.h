//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    template<typename T, uint N>
    struct TBox {
        using scalar_t = T;
        using vector_t = Vector<T, N>;

        vector_t lower, upper;

        XPU TBox() :
                lower(empty_bounds_lower<scalar_t>()),
                upper(empty_bounds_upper<scalar_t>()) {}

        explicit inline XPU TBox(const vector_t &v)
                : lower(v),
                  upper(v) {}

        /*! construct a new, origin-oriented box of given size */
        inline XPU TBox(const vector_t &lo, const vector_t &hi)
                : lower(lo),
                  upper(hi) {}

        /*! returns new box including both ourselves _and_ the given point */
        inline XPU TBox including(const vector_t &other) const {
            return TBox(min(lower, other), max(upper, other));
        }

        /*! returns new box including both ourselves _and_ the given point */
        inline XPU TBox including(const TBox &other) const {
            return TBox(min(lower, other.lower), max(upper, other.upper));
        }

        /*! returns new box including both ourselves _and_ the given point */
        inline XPU TBox &extend(const vector_t &other) {
            lower = min(lower, other);
            upper = max(upper, other);
            return *this;
        }

        /*! returns new box including both ourselves _and_ the given point */
        inline XPU TBox &extend(const TBox &other) {
            lower = min(lower, other.lower);
            upper = max(upper, other.upper);
            return *this;
        }

        /*! get the d-th dimensional slab (lo[dim]..hi[dim] */
        inline XPU interval <scalar_t> get_slab(const uint32_t dim) {
            return interval<scalar_t>(lower[dim], upper[dim]);
        }

        inline XPU bool contains(const vector_t &point) const {
            return all(point >= lower) && all(upper >= point);
        }

        inline XPU bool contains(const TBox &other) const {
            return all(other.lower >= lower) && all(upper >= other.upper);
        }

        inline XPU bool overlap(const TBox &other) const {
            return contains(other.lower) || contains(other.upper);
        }

        inline XPU vector_t center() const { return (lower + upper) / (scalar_t) 2; }

        inline XPU vector_t span() const { return upper - lower; }

        inline XPU vector_t size() const { return upper - lower; }
    };
}