//
// Created by Zero on 2021/2/8.
//


#pragma once

#include "../math/common.h"
#include "core/backend/buffer_view.h"
#include "core/backend/managed.h"


namespace luminous {
    inline namespace sampling {

        struct Distribute1DBuilder {
        public:
            std::vector<float> func;
            std::vector<float> CDF;
            float func_integral;

            Distribute1DBuilder() = default;

            Distribute1DBuilder(std::vector<float> func, std::vector<float> CDF, float integral)
                    : func(move(func)), CDF(move(CDF)), func_integral(integral) {}
        };

        struct Distribute1DData {
            Managed<float> func;
            Managed<float> CDF;
            float func_integral;

            Distribute1DData() = default;

            Distribute1DData(vector<float> func, vector<float> CDF, float integral)
                    : func_integral(integral) {
                this->func.reset(move(func));
                this->CDF.reset(move(CDF));
            }

            void allocate_device(const SP<Device> device) {
                func.allocate_device(device);
                CDF.allocate_device(device);
            }

            void synchronize_to_gpu() {
                func.synchronize_to_gpu();
                CDF.synchronize_to_gpu();
            }

            void allocate_synchronize(const SP<Device> &device) {
                allocate_device(device);
                synchronize_to_gpu();
            }
        };

        class Distribute1D {
        public:
            using value_type = float;
            using const_value_type = const float;
        private:
            BufferView <value_type> _func;
            BufferView <value_type> _CDF;
            float _func_integral;
        public:
            XPU Distribute1D(BufferView <value_type> func,
                             BufferView <value_type> CDF, float integral)
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

            static Distribute1DBuilder create_builder(std::vector<float> func) {
                size_t num = func.size();
                vector<float> CDF(num + 1);
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
                return Distribute1DBuilder(move(func), move(CDF), integral);
            }

            static auto create_data(Distribute1DBuilder builder) {
                return Distribute1DData(move(builder.func), move(builder.CDF), builder.func_integral);
            }

//            static Distribute1D create_on_host(const Distribute1DData &data) {
//                return Distribute1D(data.func.host_buffer_view(), data.CDF.host_buffer_view(), data.func_integral);
//            }
//
//            static Distribute1D create_on_device(const Distribute1DData &data) {
//                return Distribute1D(data.func.device_buffer_view(), data.CDF.device_buffer_view(), data.func_integral);
//            }
        };

    } // luminous::sampling
} // luminous