//
// Created by Zero on 2021/2/6.
//


#pragma once

namespace luminous {

    namespace sampling {

        XPU [[nodiscard]] float linear_PDF(float x, float a, float b) {
            assert(a > 0 && b > 0);
            if (x < 0 || x > 1)
                return 0;
            return 2 * lerp(x, a, b) / (a + b);
        }


        inline float sample_linear(float u, float a, float b) {
            assert(a > 0 && b > 0);
            if (u == 0 && a == 0)
                return 0;
            float x = u * (a + b) / (a + std::sqrt(lerp(u, sqr(a), sqr(b))));
            return std::min(x, one_minus_epsilon);
        }

        XPU [[nodiscard]] inline float fast_exp(float x) {
#ifdef IS_GPU_CODE
            return __expf(x);
#else
            return std::exp(x);
#endif
        }

        XPU inline int sample_discrete(lstd::span<const float> weights, float u,
                                       float *pmf, float *uRemapped) {
            // Handle empty _weights_ for discrete sampling
            if (weights.empty()) {
                if (pmf != nullptr)
                    *pmf = 0;
                return -1;
            }

            // Compute sum of _weights_
            float sumWeights = 0;
            for (float w : weights)
                sumWeights += w;

            // Find offset in _weights_ corresponding to _u_
            int offset = 0;
            while (offset < weights.size() && u >= weights[offset] / sumWeights) {
                u -= weights[offset] / sumWeights;
                ++offset;
            }
            // CHECK_RARE(1e-6, offset == weights.size());
            if (offset == weights.size())
                offset = weights.size() - 1;

            // Compute PMF and remapped _u_ value, if necessary
            float p = weights[offset] / sumWeights;
            if (pmf != nullptr)
                *pmf = p;
            if (uRemapped != nullptr)
                *uRemapped = std::min(float(u / p), one_minus_epsilon);

            return offset;
        }

        float gaussian(float x, float mu = 0, float sigma = 1) {
            return 1 / std::sqrt(2 * Pi * sigma * sigma) *
                   fast_exp(-sqr(x - mu) / (2 * sigma * sigma));
        }

        inline float gaussian_integral(float x0, float x1, float mu = 0,
                                       float sigma = 1) {
            assert(sigma > 0);
            float sigmaRoot2 = sigma * float(1.414213562373095);
            return 0.5f * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
        }

        XPU inline float2 sample_bilinear(float2 u, lstd::span<const float> w) {
            assert(4 == w.size());
            float2 p;
            // Sample $v$ for bilinear marginal distribution
            p[1] = sample_linear(u[1], w[0] + w[1], w[2] + w[3]);

            // Sample $u$ for bilinear conditional distribution
            p[0] = sample_linear(u[0], lerp(p[1], w[0], w[2]), lerp(p[1], w[1], w[3]));

            return p;
        }

        template <typename Float = float>
        class VarianceEstimator {
        private:
            // VarianceEstimator Private Members
            Float _mean = 0, _S = 0;
            int64_t _n = 0;
        public:
            // VarianceEstimator Public Methods

            XPU void add(Float x) {
                ++_n;
                Float delta = x - _mean;
                _mean += delta / _n;
                Float delta2 = x - _mean;
                _S += delta * delta2;
            }

            XPU Float mean() const { return _mean; }

            XPU Float variance() const { return (_n > 1) ? _S / (_n - 1) : 0; }

            XPU int64_t count() const { return _n; }

            XPU Float relative_variance() const {
                return (_n < 1 || _mean == 0) ? 0 : variance() / mean();
            }

            XPU void merge(const VarianceEstimator &ve) {
                if (ve.n == 0)
                    return;
                S = S + ve.S + Sqr(ve._mean - _mean) * n * ve.n / (n + ve.n);
                _mean = (_n * _mean + ve._n * ve._mean) / (_n + ve._n);
                _n += ve._n;
            }

        };
    }
}