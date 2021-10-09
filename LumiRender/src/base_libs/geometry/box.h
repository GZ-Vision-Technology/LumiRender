//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace geometry {
        template<typename T, uint N>
        struct TBox {
            using scalar_t = T;
            using vector_t = Vector<T, N>;

            vector_t lower, upper;

            LM_XPU TBox() :
                    lower(empty_bounds_lower<scalar_t>()),
                    upper(empty_bounds_upper<scalar_t>()) {}

            explicit inline LM_XPU TBox(const vector_t &v)
                    : lower(v),
                      upper(v) {}

            /*! construct a new, origin-oriented box of given size */
            ND_XPU_INLINE TBox(const vector_t &lo, const vector_t &hi)
                    : lower(lo),
                      upper(hi) {}

            /*! returns new box including both ourselves _and_ the given point */
            ND_XPU_INLINE TBox including(const vector_t &other) const {
                return TBox(min(lower, other), max(upper, other));
            }

            /*! returns new box including both ourselves _and_ the given point */
            ND_XPU_INLINE TBox including(const TBox &other) const {
                return TBox(min(lower, other.lower), max(upper, other.upper));
            }

            /*! returns new box including both ourselves _and_ the given point */
            LM_XPU_INLINE TBox &extend(const vector_t &other) {
                lower = min(lower, other);
                upper = max(upper, other);
                return *this;
            }

            /*! returns new box including both ourselves _and_ the given point */
            LM_XPU_INLINE TBox &extend(const TBox &other) {
                lower = min(lower, other.lower);
                upper = max(upper, other.upper);
                return *this;
            }

            /*! get the d-th dimensional slab (lo[dim]..hi[dim] */
            ND_XPU_INLINE interval <scalar_t> get_slab(const uint32_t dim) {
                return interval<scalar_t>(lower[dim], upper[dim]);
            }

            ND_XPU_INLINE vector_t offset(vector_t p) const {
                return (p - lower) / span();
            }

            ND_XPU_INLINE bool contains(const vector_t &point) const {
                return all(point >= lower) && all(upper >= point);
            }

            ND_XPU_INLINE bool contains(const TBox &other) const {
                return all(other.lower >= lower) && all(upper >= other.upper);
            }

            ND_XPU_INLINE bool overlap(const TBox &other) const {
                return contains(other.lower) || contains(other.upper);
            }

            ND_XPU_INLINE scalar_t radius() const {
                return length(upper - lower) * 0.5f;
            }

            ND_XPU_INLINE vector_t center() const {
                return (lower + upper) * 0.5f;
            }

            ND_XPU_INLINE vector_t span() const {
                return upper - lower;
            }

            ND_XPU_INLINE vector_t size() const {
                return upper - lower;
            }

            ND_XPU_INLINE scalar_t volume() const {
                return luminous::functor::volume(upper - lower);
            }

            ND_XPU_INLINE scalar_t area() const {
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

            ND_XPU_INLINE bool empty() const {
                return any(upper < lower);
            }

            ND_XPU_INLINE auto advance(Vector<scalar_t, 2> p) const {
                ++p.x;
                if (p.x == upper.x) {
                    p.x = lower.x;
                    ++p.y;
                }
                return p;
            }

            /**
             * for each every point in [lower, upper)
             * @tparam Func
             * @param func
             */
            template<typename Func>
            LM_XPU_INLINE void for_each(Func func) const {
                static_assert(std::is_integral_v<scalar_t> || std::is_unsigned_v<scalar_t>,
                        "scalar_t must be unsigned or integral!");
                auto p = lower;
                do {
                    func(p);
                    p = advance(p);
                } while (all(p < upper));
            }

            GEN_STRING_FUNC({
                                return string_printf("box : {lower: %s, upper : %s }",
                                                     lower.to_string().c_str(),
                                                     upper.to_string().c_str());
                            })
        };

        template<typename T, uint N>
        LM_ND_XPU auto intersection(const TBox<T, N> &a, const TBox<T, N> &b) {
            return TBox<T, N>(max(a.lower, b.lower), min(a.upper, b.upper));
        }

        template<typename T, uint N>
        LM_ND_XPU bool operator==(const TBox<T, N> &a, const TBox<T, N> &b) {
            return a.lower == b.lower && a.upper == b.upper;
        }

        template<typename T, uint N>
        LM_ND_XPU bool operator!=(const TBox<T, N> &a, const TBox<T, N> &b) {
            return !(a == b);
        }

#define _define_box(scalar_t, suffix)             \
        using Box##2##suffix = TBox<scalar_t, 2>; \
        using Box##3##suffix = TBox<scalar_t, 3>; \
        using Box##4##suffix = TBox<scalar_t, 4>;

        _define_box(int32_t, i);
        _define_box(uint32_t, u);
        _define_box(float, f);
        _define_box(double, d);
        _define_box(int64_t, l);

#undef _define_box
    }
}