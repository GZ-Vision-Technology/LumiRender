//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../lstd/vector.h"
#include "../math/common.h"

// from pbrt-v4

namespace luminous {
    inline namespace sampling {

#if USE_LSTD == 1
        using ::lstd::vector;
        using ::lstd::span;
        using ::lstd::Allocator;
#else
        using std::vector;
        using std::span;
        using std::Allocator;
#endif
        class PiecewiseConstant1D {
        private:
            vector<float> func, cdf;
            float min, max;
            float funcInt = 0;
        public:
            XPU size_t bytes_used() const {
                return (func.capacity() + cdf.capacity()) * sizeof(float);
            }


            PiecewiseConstant1D() = default;

            PiecewiseConstant1D(Allocator alloc) : func(alloc), cdf(alloc) {}

            PiecewiseConstant1D(span<const float> f, Allocator alloc = {})
                    : PiecewiseConstant1D(f, 0., 1., alloc) {}

            PiecewiseConstant1D(span<const float> f, float min, float max,
                                Allocator alloc = {})
                    : func(f.begin(), f.end(), alloc), cdf(f.size() + 1, alloc), min(min), max(max) {
                DCHECK_GT(max, min);
                // Take absolute value of _func_
                for (float &f : func)
                    f = std::abs(f);

                // Compute integral of step function at $x_i$
                cdf[0] = 0;
                size_t n = f.size();
                for (size_t i = 1; i < n + 1; ++i) {
                    DCHECK_GE(func[i - 1], 0);
                    cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
                }

                // Transform step function integral into CDF
                funcInt = cdf[n];
                if (funcInt == 0)
                    for (size_t i = 1; i < n + 1; ++i)
                        cdf[i] = float(i) / float(n);
                else
                    for (size_t i = 1; i < n + 1; ++i)
                        cdf[i] /= funcInt;
            }

            XPU float Integral() const { return funcInt; }

            XPU size_t size() const { return func.size(); }

            XPU float Sample(float u, float *pdf = nullptr, int *offset = nullptr) const {
                // Find surrounding CDF segments and _offset_
                int o = find_interval((int) cdf.size(), [&](int index) { return cdf[index] <= u; });
                if (offset)
                    *offset = o;

                // Compute offset along CDF segment
                float du = u - cdf[o];
                if (cdf[o + 1] - cdf[o] > 0)
                    du /= cdf[o + 1] - cdf[o];
                DCHECK(!is_nan(du));

                // Compute PDF for sampled offset
                if (pdf != nullptr)
                    *pdf = (funcInt > 0) ? func[o] / funcInt : 0;

                // Return $x$ corresponding to sample
                return lerp((o + du) / size(), min, max);
            }

            XPU lstd::optional<float> Invert(float x) const {
                // Compute offset to CDF values that bracket $x$
                if (x < min || x > max)
                    return {};
                float c = (x - min) / (max - min) * func.size();
                int offset = clamp(int(c), 0, func.size() - 1);
                DCHECK(offset >= 0 && offset + 1 < cdf.size());

                // Linearly interpolate between adjacent CDF values to find sample value
                float delta = c - offset;
                return lerp(delta, cdf[offset], cdf[offset + 1]);
            }
        };

    } // luminous::sampling
} // luminous