//
// Created by Zero on 2021/2/4.
//


#pragma once

#include "../header.h"
#include "vector_types.h"
#include "inttypes.h"
#include "float.h"

namespace luminous {
    inline namespace math {

        template<unsigned int N = 4>
        class LCG {
        private:
            uint32_t state{};
        public:
            XPU LCG() { init(0, 0); }

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


// Random Number Declarations
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
        XPU inline uint64_t MixBits(uint64_t v);

        inline uint64_t MixBits(uint64_t v) {
            v ^= (v >> 31);
            v *= 0x7fb5d329728ea185;
            v ^= (v >> 27);
            v *= 0x81dadef4bc2dd44d;
            v ^= (v >> 33);
            return v;
        }

        class RNG {
        public:
            // RNG Public Methods
            XPU RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

            XPU RNG(uint64_t seqIndex, uint64_t start) { SetSequence(seqIndex, start); }

            XPU RNG(uint64_t seqIndex) { SetSequence(seqIndex); }

            XPU void SetSequence(uint64_t sequenceIndex, uint64_t seed);

            XPU void SetSequence(uint64_t sequenceIndex) {
                SetSequence(sequenceIndex, MixBits(sequenceIndex));
            }

            template<typename T>
            XPU T Uniform();

            template<typename T>
            XPU typename std::enable_if_t<std::is_integral_v<T>, T> Uniform(T b) {
                T threshold = (~b + 1u) % b;
                while (true) {
                    T r = Uniform < T > ();
                    if (r >= threshold)
                        return r % b;
                }
            }

            XPU void Advance(int64_t idelta);

            XPU int64_t operator-(const RNG &other) const;

            std::string ToString() const {
                return string_printf("[ RNG state: %" PRIu64 " inc: %" PRIu64 " ]", state, inc);
            }

        private:
            // RNG Private Members
            uint64_t state, inc;
        };

        // RNG Inline Method Definitions
        template <typename T>
        inline T RNG::Uniform() {
            return T::unimplemented;
        }

        template <>
        inline uint32_t RNG::Uniform<uint32_t>();

        template <>
        inline uint32_t RNG::Uniform<uint32_t>() {
            uint64_t oldstate = state;
            state = oldstate * PCG32_MULT + inc;
            uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = (uint32_t)(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
        }

        template <>
        inline uint64_t RNG::Uniform<uint64_t>() {
            uint64_t v0 = Uniform<uint32_t>(), v1 = Uniform<uint32_t>();
            return (v0 << 32) | v1;
        }

        template <>
        inline int32_t RNG::Uniform<int32_t>() {
            // https://stackoverflow.com/a/13208789
            uint32_t v = Uniform<uint32_t>();
            if (v <= (uint32_t)std::numeric_limits<int32_t>::max())
                return int32_t(v);
            DCHECK_GE(v, (uint32_t)std::numeric_limits<int32_t>::min());
            return int32_t(v - std::numeric_limits<int32_t>::min()) +
                   std::numeric_limits<int32_t>::min();
        }

        template <>
        inline int64_t RNG::Uniform<int64_t>() {
            // https://stackoverflow.com/a/13208789
            uint64_t v = Uniform<uint64_t>();
            if (v <= (uint64_t)std::numeric_limits<int64_t>::max())
                // Safe to type convert directly.
                return int64_t(v);
            DCHECK_GE(v, (uint64_t)std::numeric_limits<int64_t>::min());
            return int64_t(v - std::numeric_limits<int64_t>::min()) +
                   std::numeric_limits<int64_t>::min();
        }

        inline void RNG::SetSequence(uint64_t sequenceIndex, uint64_t seed) {
            state = 0u;
            inc = (sequenceIndex << 1u) | 1u;
            Uniform<uint32_t>();
            state += seed;
            Uniform<uint32_t>();
        }

        template <>
        inline float RNG::Uniform<float>() {
            return std::min<float>(OneMinusEpsilon, Uniform<uint32_t>() * 0x1p-32f);
        }

        template <>
        inline double RNG::Uniform<double>() {
            return std::min<double>(OneMinusEpsilon, Uniform<uint64_t>() * 0x1p-64);
        }

        inline void RNG::Advance(int64_t idelta) {
            uint64_t curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
            uint64_t accPlus = 0u, delta = (uint64_t)idelta;
            while (delta > 0) {
                if (delta & 1) {
                    accMult *= curMult;
                    accPlus = accPlus * curMult + curPlus;
                }
                curPlus = (curMult + 1) * curPlus;
                curMult *= curMult;
                delta /= 2;
            }
            state = accMult * state + accPlus;
        }

        XPU inline int64_t RNG::operator-(const RNG &other) const {
            DCHECK_EQ(inc, other.inc);
            uint64_t curMult = PCG32_MULT, curPlus = inc, curState = other.state;
            uint64_t theBit = 1u, distance = 0u;
            while (state != curState) {
                if ((state & theBit) != (curState & theBit)) {
                    curState = curState * curMult + curPlus;
                    distance |= theBit;
                }
                DCHECK_EQ(state & theBit, curState & theBit);
                theBit <<= 1;
                curPlus = (curMult + 1ULL) * curPlus;
                curMult *= curMult;
            }
            return (int64_t)distance;
        }

        struct PCG {
            XPU explicit PCG(uint64_t sequence = 0) { pcg32_init(sequence); }

            XPU void pcg32_init(uint64_t seed) {
                state = seed + increment;
                pcg32();
            }

            XPU uint32_t uniform_u32() { return pcg32(); }

            XPU double uniform_float() { return pcg32() / double(0xffffffff); }

        private:
            uint64_t state = 0x4d595df4d0f33173; // Or something seed-dependent
            static uint64_t const multiplier = 6364136223846793005u;
            static uint64_t const increment = 1442695040888963407u; // Or an arbitrary odd constant
            XPU static uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }

            XPU uint32_t pcg32() {
                uint64_t x = state;
                auto count = (unsigned) (x >> 59); // 59 = 64 - 5

                state = x * multiplier + increment;
                x ^= x >> 18;                              // 18 = (64 - 27)/2
                return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
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