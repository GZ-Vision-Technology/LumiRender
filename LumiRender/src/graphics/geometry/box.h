//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    inline namespace geometry {
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
            [[nodiscard]] inline XPU TBox including(const vector_t &other) const {
                return TBox(min(lower, other), max(upper, other));
            }

            /*! returns new box including both ourselves _and_ the given point */
            [[nodiscard]] inline XPU TBox including(const TBox &other) const {
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
            [[nodiscard]] inline XPU interval<scalar_t> get_slab(const uint32_t dim) {
                return interval<scalar_t>(lower[dim], upper[dim]);
            }

            [[nodiscard]] inline XPU vector_t offset(vector_t p) const {
                return (p - lower) / span();
            }

            [[nodiscard]] inline XPU bool contains(const vector_t &point) const {
                return all(point >= lower) && all(upper >= point);
            }

            [[nodiscard]] inline XPU bool contains(const TBox &other) const {
                return all(other.lower >= lower) && all(upper >= other.upper);
            }

            [[nodiscard]] inline XPU bool overlap(const TBox &other) const {
                return contains(other.lower) || contains(other.upper);
            }

            [[nodiscard]] inline XPU vector_t center() const {
                return (lower + upper) / (scalar_t) 2;
            }

            [[nodiscard]] inline XPU vector_t span() const {
                return upper - lower;
            }

            [[nodiscard]] inline XPU vector_t size() const {
                return upper - lower;
            }

            [[nodiscard]] inline XPU scalar_t volume() const {
                return luminous::functor::volume(upper - lower);
            }

            [[nodiscard]] inline XPU scalar_t area() const {
                static_assert(N == 2 || N == 3);
                vector_t diag = upper - lower;
                if constexpr (N == 2) {
                    return diag.x * diag.y;
                } else if constexpr (N == 3) {
                    return 2 * (diag.x * diag.y
                                + diag.x * diag.z
                                + diag.y * diag.z);
                }
            }

            [[nodiscard]] inline XPU bool empty() const {
                return any(upper < lower);
            }

            [[nodiscard]] inline std::string to_string() const {
                return string_printf("box : {lower: %s, upper : %s }",
                                     lower.to_string().c_str(),
                                     upper.to_string().c_str());
            }
        };

        template<typename T, uint N>
        [[nodiscard]] XPU auto intersection(const TBox<T, N> &a, const TBox<T, N> &b) {
            return TBox<T, N>(max(a.lower, b.lower), min(a.upper, b.upper));
        }

        template<typename T, uint N>
        [[nodiscard]] XPU bool operator == (const TBox<T, N> &a, const TBox<T, N> &b) {
            return a.lower == b.lower && a.upper == b.upper;
        }

        template<typename T, uint N>
        [[nodiscard]] XPU bool operator != (const TBox<T, N> &a, const TBox<T, N> &b) {
            return !(a == b);
        }

#define _define_box(scalar_t, suffix)             \
        using Box##2##suffix = TBox<scalar_t, 2>; \
        using Box##3##suffix = TBox<scalar_t, 3>; \
        using Box##4##suffix = TBox<scalar_t, 4>;

        _define_box(int32_t , i);
        _define_box(float, f);
        _define_box(double, d);
        _define_box(int64_t , l);

#undef _define_box

    }
}