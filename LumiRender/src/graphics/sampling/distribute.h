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
        using ::lstd::optional;
#else
        using std::vector;
        using std::span;
        using std::Allocator;
#endif

        class PiecewiseConstant1D {
        public:
            vector<float> func, cdf;
            float min, max;
            float funcInt = 0;
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
                for (float &f : func) {
                    f = std::abs(f);
                }
                // Compute integral of step function at $x_i$
                cdf[0] = 0;
                size_t n = f.size();
                for (size_t i = 1; i < n + 1; ++i) {
                    DCHECK_GE(func[i - 1], 0);
                    cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
                }

                // Transform step function integral into CDF
                funcInt = cdf[n];
                if (funcInt == 0) {
                    for (size_t i = 1; i < n + 1; ++i) {
                        cdf[i] = float(i) / float(n);
                    }
                } else {
                    for (size_t i = 1; i < n + 1; ++i) {
                        cdf[i] /= funcInt;
                    }
                }

            }

            XPU float Integral() const { return funcInt; }

            XPU size_t size() const { return func.size(); }

            XPU float Sample(float u, float *pdf = nullptr, int *offset = nullptr) const {
                // Find surrounding CDF segments and _offset_
                int o = find_interval((int) cdf.size(), [&](int index) { return cdf[index] <= u; });
                if (offset) {
                    *offset = o;
                }

                // Compute offset along CDF segment
                float du = u - cdf[o];
                if (cdf[o + 1] - cdf[o] > 0) {
                    du /= cdf[o + 1] - cdf[o];
                }
                DCHECK(!is_nan(du));

                // Compute PDF for sampled offset
                if (pdf != nullptr) {
                    *pdf = (funcInt > 0) ? func[o] / funcInt : 0;
                }
                
                // Return $x$ corresponding to sample
                return lerp((o + du) / size(), min, max);
            }

            XPU lstd::optional<float> Invert(float x) const {
                // Compute offset to CDF values that bracket $x$
                if (x < min || x > max) {
                    return {};
                }
                float c = (x - min) / (max - min) * func.size();
                int offset = clamp(int(c), 0, func.size() - 1);
                DCHECK(offset >= 0 && offset + 1 < cdf.size());

                // Linearly interpolate between adjacent CDF values to find sample value
                float delta = c - offset;
                return lerp(delta, cdf[offset], cdf[offset + 1]);
            }
        };

        class PiecewiseConstant2D {
        private:
            vector<PiecewiseConstant1D> pConditionalV;
            PiecewiseConstant1D pMarginal;
            Box2f domain;
        public:
            PiecewiseConstant2D() = default;

            PiecewiseConstant2D(Allocator alloc)
                    : pConditionalV(alloc), pMarginal(alloc) {}

            PiecewiseConstant2D(span<const float> func, int nu, int nv,
                                Box2f domain = Box2f(make_float2(0), make_float2(1)),
                                Allocator alloc = {}) {
                DCHECK_EQ(func.size(), (size_t) nu * (size_t) nv);
                pConditionalV.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    pConditionalV.emplace_back(func.subspan(v * nu, nu),
                                               domain.lower.x,
                                               domain.upper.y, alloc);
                }

                std::vector<float> marginalFunc;
                marginalFunc.reserve(nv);
                for (int v = 0; v < nv; ++v) {
                    marginalFunc.push_back(pConditionalV[v].Integral());
                }

                pMarginal = PiecewiseConstant1D(marginalFunc, domain.lower[1], domain.upper[1], alloc);

            }

            XPU float Integral() const { return pMarginal.Integral(); }

            float2 Sample(const float2 &u, float *pdf = nullptr) const {
                float pdfs[2];
                int v;
                float d1 = pMarginal.Sample(u[1], &pdfs[1], &v);
                float d0 = pConditionalV[v].Sample(u[0], &pdfs[0]);
                if (pdf != nullptr) {
                    *pdf = pdfs[0] * pdfs[1];
                }
                return make_float2(d0, d1);
            }

            float PDF(float2 pr) const {
                float2 p = domain.offset(pr);
                int iu = clamp(int(p[0] * pConditionalV[0].size()), 0, pConditionalV[0].size() - 1);
                int iv = clamp(int(p[1] * pMarginal.size()), 0, pMarginal.size() - 1);
                return pConditionalV[iv].func[iu] / pMarginal.Integral();
            }

            XPU optional<float2> Invert(const float2 &p) const {
                optional<float> mInv = pMarginal.Invert(p[1]);
                if (!mInv) {
                    return {};
                }

                float p1o = (p[1] - domain.lower[1]) / (domain.upper[1] - domain.lower[1]);
                if (p1o < 0 || p1o > 1) {
                    return {};
                }
                int offset = clamp(p1o * pConditionalV.size(), 0, pConditionalV.size() - 1);
                optional<float> cInv = pConditionalV[offset].Invert(p[0]);
                if (!cInv) {
                    return {};
                }
                return make_float2(*cInv, *mInv);
            }
        };

    } // luminous::sampling
} // luminous