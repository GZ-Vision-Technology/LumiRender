//
// Created by Zero on 2021/2/4.
//


#pragma once

namespace luminous {
    inline namespace math {

        template<unsigned int N = 4>
        class LCG {
        private:
            uint32_t state;
        public:
            [[nodiscard]] inline XPU LCG(unsigned int val0, unsigned int val1) {
                init(val0, val1);
            }

            inline XPU void init(unsigned int val0, unsigned int val1) {
                unsigned int v0 = val0;
                unsigned int v1 = val1;
                unsigned int s0 = 0;

                for (unsigned int n = 0; n < N; n++) {
                    s0 += 0x9e3779b9;
                    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
                }
                state = v0;
            }

            // Generate random unsigned int in [0, 2^24)
            [[nodiscard]] inline XPU float next() {
                const uint32_t LCG_A = 1664525u;
                const uint32_t LCG_C = 1013904223u;
                state = (LCG_A * state + LCG_C);
                return ldexpf(float(state), -32);
            }
        };


        class DRand48 {
        private:
            uint64_t state;
        public:
            /*! initialize the random number generator with a new seed (usually
              per pixel) */
            inline XPU void init(int seed = 0) {
                state = seed;
                for (int warmUp = 0; warmUp < 10; warmUp++) {
                    next();
                }
            }

            /*! get the next 'random' number in the sequence */
            [[nodiscard]] inline XPU float next() {
                const uint64_t a = 0x5DEECE66DULL;
                const uint64_t c = 0xBULL;
                const uint64_t mask = 0xFFFFFFFFFFFFULL;
                state = a * state + c;
                return float(state & mask) / float(mask + 1ULL);
            }
        };

    } // luminous::math
} // luminous