//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../math/common.h"
#include "core/backend/buffer_view.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace sampling {

        class WrapDistribute1D;

        class Distribute1D {
        public:
            using value_type = float;
            using const_value_type = const float;
        private:
            BufferView<const_value_type> _func;
            BufferView<value_type> _CDF;
            float _func_integral;
        public:
            XPU Distribute1D(BufferView<const_value_type> func,
                             BufferView<value_type> CDF, float integral)
                    : _func(func), _CDF(CDF), _func_integral(integral) {}

            NDSC_XPU size_t size() const { return _func.size(); }

            NDSC_XPU float sample_continuous(float u, float *pdf = nullptr, int *off = nullptr) const {
                auto predicate = [&](int index) {
                    return _CDF[index] <= u;
                };
                int offset = find_interval((int) _CDF.size(), predicate);
                if (off) {
                    *off = offset;
                }
                float du = u - _CDF[offset];
                if ((_CDF[offset + 1] - _CDF[offset]) > 0) {
                    DCHECK_GT(_CDF[offset + 1], _CDF[offset]);
                    du /= (_CDF[offset + 1] - _CDF[offset]);
                }
                DCHECK(!is_nan(du));

                if (pdf) {
                    *pdf = (_func_integral > 0) ? _func[offset] / _func_integral : 0;
                }
                return (offset + du) / size();
            }

            NDSC_XPU int sample_discrete(float u, float *PMF = nullptr, float *u_remapped = nullptr) const {
                auto predicate = [&](int index) {
                    return _CDF[index] <= u;
                };
                int offset = find_interval((int) _CDF.size(), predicate);
                if (PMF) {
                    *PMF = (_func_integral > 0) ? _func[offset] / (_func_integral * size()) : 0;
                }
                if (u_remapped) {
                    *u_remapped = (u - _CDF[offset]) / (_CDF[offset + 1] - _CDF[offset]);
                    DCHECK(*u_remapped >= 0.f && *u_remapped <= 1.f);
                }
                return offset;
            }

            NDSC_XPU float integral() const { return _func_integral; }

            template<typename Index>
            NDSC_XPU float func_at(Index i) const { return _func[i]; }

            template<typename Index>
            NDSC_XPU float PMF(Index i) const {
                DCHECK(i >= 0 && i < size());
                return func_at(i) / (integral() * size());
            }

            static Distribute1D create(const std::vector<float> &func) {
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

                
//                return Distribute1D
            }
        };

        struct WrapDistribute1D {
            Distribute1D distribute_1d;
            Managed<Distribute1D::value_type> func;
            Managed<Distribute1D::value_type> CDF;
        };

    } // luminous::sampling
} // luminous