//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/lstd/common.h"
#include <vector>

namespace luminous {
    inline namespace sampling {
        using std::vector;
        using std::move;

        struct DichotomyBuilder {
        public:
            std::vector<float> func;
            std::vector<float> CDF;
            float func_integral{};

            DichotomyBuilder() = default;

            DichotomyBuilder(std::vector<float> func, std::vector<float> CDF, float integral)
                    : func(move(func)), CDF(move(CDF)), func_integral(integral) {}
        };

        struct DichotomyData {
        public:
            using value_type = float;
            using const_value_type = const float;
            using Builder = DichotomyBuilder;
        public:
            // todo change to indice mode, reduce memory usage
            BufferView<const_value_type> func{};
            BufferView<const_value_type> CDF{};
            float func_integral{};

            DichotomyData() = default;

            DichotomyData(BufferView<const_value_type> func,
                          BufferView<const_value_type> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}
        };

        template<int Size>
        struct StaticDichotomyData {
        public:
            using Builder = DichotomyBuilder;
        private:
            static constexpr int size_in_bytes = Size * sizeof(float);
        public:
            Array<float, Size> func;
            Array<float, Size + 1> CDF;
            float func_integral{};

            StaticDichotomyData() = default;

            StaticDichotomyData(Array<float, Size> func,
                                Array<float, Size + 1> CDF, float integral)
                    : func(func), CDF(CDF), func_integral(integral) {}

            StaticDichotomyData(const float *f, const float *C, float integral) {
                init(f, C, integral);
            }

            explicit StaticDichotomyData(const Builder &builder) {
                init(builder.func.data(), builder.CDF.data(), builder.func_integral);
            }

            void init(const float *f, const float *C, float integral) {
                std::memcpy(func.begin(), f, size_in_bytes);
                std::memcpy(CDF.begin(), C, size_in_bytes + sizeof(float));
                func_integral = integral;
            }
        };

        template<typename T = DichotomyData>
        class TDichotomySampler {
        public:
            using Builder = DichotomyBuilder;
            using data_type = T;
        private:
            data_type _data;
        public:
            TDichotomySampler() = default;

            explicit TDichotomySampler(const data_type &data) : _data(data) {}

            template<typename ...Args>
            explicit TDichotomySampler(Args ...args) : TDichotomySampler(T(std::forward<Args>(args)...)) {}

            LM_ND_XPU size_t size() const { return _data.func.size(); }

            LM_ND_XPU float sample_continuous(float u, float *pdf, int *ofs) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                size_t offset = find_interval((int) _data.CDF.size(), predicate);
                *ofs = offset;
                float du = u - _data.CDF[offset];
                if ((_data.CDF[offset + 1] - _data.CDF[offset]) > 0) {
                    DCHECK_GT(_data.CDF[offset + 1], _data.CDF[offset]);
                    du /= (_data.CDF[offset + 1] - _data.CDF[offset]);
                }
                DCHECK(!is_nan(du));

                *pdf = PDF(offset);
                return (offset + du) / size();
            }

            LM_ND_XPU int sample_discrete(float u, float *p, float *u_remapped) const {
                auto predicate = [&](int index) {
                    return _data.CDF[index] <= u;
                };
                int offset = find_interval(_data.CDF.size(), predicate);
                *p = PMF(offset);
                *u_remapped = (u - _data.CDF[offset]) / (_data.CDF[offset + 1] - _data.CDF[offset]);
                DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                return offset;
            }

            LM_ND_XPU float integral() const { return _data.func_integral; }

            template<typename Index>
            LM_ND_XPU float func_at(Index i) const { return _data.func[i]; }

            template<typename Index>
            LM_ND_XPU float PMF(Index i) const {
                DCHECK(i >= 0 && i < size());
                return integral() > 0 ? (func_at(i) / (integral() * size())) : 0;
            }

            LM_ND_XPU float PDF(uint32_t i) const {
                DCHECK(i < size());
                float f = func_at(i);
                return integral() > 0 ? (func_at(i) / integral()) : 0;
            }

            static DichotomyBuilder create_builder(std::vector<float> func) {
                size_t num = func.size();
                std::vector<float> CDF(num + 1);
                CDF[0] = 0;
                for (int i = 1; i < num + 1; ++i) {
                    CDF[i] = CDF[i - 1] + func[i - 1] / num;
                }
                float integral = CDF[num];
                if (integral == 0) {
                    for (int i = 1; i < num + 1; ++i) {
                        CDF[i] = float(i) / float(num);
                    }
                } else {
                    for (int i = 1; i < num + 1; ++i) {
                        CDF[i] = CDF[i] / integral;
                    }
                }
                return {move(func), move(CDF), integral};
            }
        };
    } // luminous::sampling
} // luminous