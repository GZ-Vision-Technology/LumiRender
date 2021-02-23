//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "../header.h"
#include "vector_types.h"

namespace luminous {
    inline namespace math {

        template<unsigned int N = 4>
        class LCG {
        private:
            uint32_t state{};
        public:
            XPU LCG() { init(0,0); }

            NDSC XPU LCG(uint2 v) {
                init(v);
            }

            XPU void init(uint2 v) {
                init(v.x, v.y);
            }

            [[nodiscard]] XPU LCG(unsigned int val0, unsigned int val1) {
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

        struct PCG {
            XPU PCG(uint64_t sequence = 0) { pcg32_init(sequence); }

            XPU uint32_t uniform_u32() { return pcg32(); }

            XPU double uniform_float() { return pcg32() / double(0xffffffff); }

        private:
            uint64_t state = 0x4d595df4d0f33173; // Or something seed-dependent
            static uint64_t const multiplier = 6364136223846793005u;
            static uint64_t const increment = 1442695040888963407u; // Or an arbitrary odd constant
            XPU static uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }

            XPU uint32_t pcg32(void) {
                uint64_t x = state;
                unsigned count = (unsigned) (x >> 59); // 59 = 64 - 5

                state = x * multiplier + increment;
                x ^= x >> 18;                              // 18 = (64 - 27)/2
                return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
            }

            XPU void pcg32_init(uint64_t seed) {
                state = seed + increment;
                (void) pcg32();
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
            inline XPU float next() {
                const uint64_t a = 0x5DEECE66DULL;
                const uint64_t c = 0xBULL;
                const uint64_t mask = 0xFFFFFFFFFFFFULL;
                state = a * state + c;
                return float(state & mask) / float(mask + 1ULL);
            }
        };

    } // luminous::math
} // luminous