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

            LM_NODISCARD XPU LCG(uint2 v) {
                init(v);
            }

            XPU void init(uint2 v) {
                init(v.x, v.y);
            }

            LM_NODISCARD XPU LCG(unsigned int val0, unsigned int val1) {
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
            LM_NODISCARD inline XPU float next() {
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
            XPU RNG() : _state(PCG32_DEFAULT_STATE), _inc(PCG32_DEFAULT_STREAM) {}

            XPU RNG(uint64_t seqIndex, uint64_t start) { set_sequence(seqIndex, start); }

            XPU RNG(uint64_t seqIndex) { set_sequence(seqIndex); }

            XPU void set_sequence(uint64_t sequenceIndex, uint64_t seed);

            XPU void set_sequence(uint64_t sequenceIndex) {
                set_sequence(sequenceIndex, MixBits(sequenceIndex));
            }

            template<typename T>
            XPU T uniform();

            template<typename T>
            XPU typename std::enable_if_t<std::is_integral_v<T>, T> Uniform(T b) {
                T threshold = (~b + 1u) % b;
                while (true) {
                    T r = uniform<T>();
                    if (r >= threshold)
                        return r % b;
                }
            }

            XPU void advance(int64_t idelta);

            XPU int64_t operator-(const RNG &other) const;

            GEN_STRING_FUNC({
                return string_printf("[ RNG state: %" PRIu64 " inc: %" PRIu64 " ]", _state, _inc);
            })

        private:
            // RNG Private Members
            uint64_t _state, _inc;
        };

        // RNG Inline Method Definitions
        template <typename T>
        inline T RNG::uniform() {
            return T::unimplemented;
        }

        template <>
        inline uint32_t RNG::uniform<uint32_t>();

        template <>
        inline uint32_t RNG::uniform<uint32_t>() {
            uint64_t oldstate = _state;
            _state = oldstate * PCG32_MULT + _inc;
            uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = (uint32_t)(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
        }

        template <>
        inline uint64_t RNG::uniform<uint64_t>() {
            uint64_t v0 = uniform<uint32_t > (), v1 = uniform<uint32_t > ();
            return (v0 << 32) | v1;
        }

        template <>
        inline int32_t RNG::uniform<int32_t>() {
            // https://stackoverflow.com/a/13208789
            uint32_t v = uniform<uint32_t > ();
            if (v <= (uint32_t)std::numeric_limits<int32_t>::max())
                return int32_t(v);
            DCHECK_GE(v, (uint32_t)std::numeric_limits<int32_t>::min());
            return int32_t(v - std::numeric_limits<int32_t>::min()) +
                   std::numeric_limits<int32_t>::min();
        }

        template <>
        inline int64_t RNG::uniform<int64_t>() {
            // https://stackoverflow.com/a/13208789
            uint64_t v = uniform<uint64_t > ();
            if (v <= (uint64_t)std::numeric_limits<int64_t>::max())
                // Safe to type convert directly.
                return int64_t(v);
            DCHECK_GE(v, (uint64_t)std::numeric_limits<int64_t>::min());
            return int64_t(v - std::numeric_limits<int64_t>::min()) +
                   std::numeric_limits<int64_t>::min();
        }

        inline void RNG::set_sequence(uint64_t sequenceIndex, uint64_t seed) {
            _state = 0u;
            _inc = (sequenceIndex << 1u) | 1u;
            uniform<uint32_t>();
            _state += seed;
            uniform<uint32_t>();
        }

        template <>
        inline float RNG::uniform<float>() {
            return std::min<float>(OneMinusEpsilon, uniform<uint32_t > () * 0x1p-32f);
        }

        template <>
        inline double RNG::uniform<double>() {
            return std::min<double>(OneMinusEpsilon, uniform<uint64_t > () * 0x1p-64);
        }

        inline void RNG::advance(int64_t idelta) {
            uint64_t curMult = PCG32_MULT, curPlus = _inc, accMult = 1u;
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
            _state = accMult * _state + accPlus;
        }

        XPU inline int64_t RNG::operator-(const RNG &other) const {
            DCHECK_EQ(_inc, other._inc);
            uint64_t curMult = PCG32_MULT, curPlus = _inc, curState = other._state;
            uint64_t theBit = 1u, distance = 0u;
            while (_state != curState) {
                if ((_state & theBit) != (curState & theBit)) {
                    curState = curState * curMult + curPlus;
                    distance |= theBit;
                }
                DCHECK_EQ(_state & theBit, curState & theBit);
                theBit <<= 1;
                curPlus = (curMult + 1ULL) * curPlus;
                curMult *= curMult;
            }
            return (int64_t)distance;
        }

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