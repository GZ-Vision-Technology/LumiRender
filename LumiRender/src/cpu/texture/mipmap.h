//
// Created by Zero on 06/01/2022.
//


#pragma once

#include "base_libs/math/common.h"
#include "util/image.h"
#include "pyramid_mgr.h"

namespace luminous {
    inline namespace cpu {

        struct ResampleWeight {
            int firstTexel;
            float weight[4];
        };

        class MIPMap : public ImageBase {
        private:
            const bool _tri_linear{true};
            const float _max_anisotropy{8.f};
            index_t _index{invalid_uint32};
            ImageWrap _image_wrap{};

        public:
            explicit MIPMap(const Image &image, ImageWrap image_wrap,
                            float max_anisotropy = 8.f, bool tri_linear = true)
                    : ImageBase(image.pixel_format(), image.resolution()),
                      _image_wrap(image_wrap),
                      _max_anisotropy(max_anisotropy),
                      _tri_linear(tri_linear) {
                init(image);
            }

            void init(const Image &image) {
                switch (pixel_format()) {
                    case PixelFormat::R8U: {
                        gen_pyramid<uchar, float>(reinterpret_cast<const uchar *>(image.pixel_ptr()));
                        break;
                    }
                    case PixelFormat::RG8U: {
                        gen_pyramid<uchar2, float2>(reinterpret_cast<const uchar2 *>(image.pixel_ptr()));
                        break;
                    }
                    case PixelFormat::RGBA8U: {
                        gen_pyramid<uchar4, float4>(reinterpret_cast<const uchar4 *>(image.pixel_ptr()));
                        break;
                    }
                    case PixelFormat::R32F: {
                        gen_pyramid<float>(reinterpret_cast<const float *>(image.pixel_ptr()));
                        break;
                    }
                    case PixelFormat::RG32F: {
                        gen_pyramid<float2>(reinterpret_cast<const float2 *>(image.pixel_ptr()));
                        break;
                    }
                    case PixelFormat::RGBA32F: {
                        gen_pyramid<float4>(reinterpret_cast<const float4 *>(image.pixel_ptr()));
                        break;
                    }
                    default:
                        DCHECK(0);
                }
            }

            template<typename T>
            LM_NODISCARD T Clamp(T val) {
                return val;
            }


            template<typename T>
            void gen_pyramid_index() {
                _index = PyramidMgr::instance()->template generate_empty_pyramid<T>();
            }

            template<typename T, typename U = T>
            void gen_pyramid(const T *img) {
                std::unique_ptr<T[]> resampled_image = nullptr;
                int2 res = make_int2(resolution());

                gen_pyramid_index<T>();

                if (!is_power_of_two(res.x) || !is_power_of_two(res.y)) {
                    int2 res_pot = make_int2(round_up_POT(int32_t(res.x)),
                                             round_up_POT(int32_t(res.y)));

                    std::unique_ptr<ResampleWeight[]> sWeights = resample_weights(res.x, res_pot[0]);
                    resampled_image.reset(new_array<T>(res_pot.x * res_pot.y));

                    parallel_for(res[1], [&](uint32_t t, uint32_t _) {
                        for (int s = 0; s < res_pot[0]; ++s) {
                            resampled_image[t * res_pot[0] + s] = T();
                            for (int j = 0; j < 4; ++j) {
                                int origS = sWeights[s].firstTexel + j;
                                if (_image_wrap == ImageWrap::Repeat) {
                                    origS = Mod(origS, res[0]);
                                } else if (_image_wrap == ImageWrap::Clamp) {
                                    origS = clamp(origS, 0, res[0] - 1);
                                }
                                if (origS >= 0 && origS < res[0]) {
                                    resampled_image[t * res_pot[0] + s] +=
                                            T(sWeights[s].weight[j] *
                                              U(img[t * res[0] + origS]));
                                }
                            }
                        }
                    }, 16);

                    std::unique_ptr<ResampleWeight[]> tWeights = resample_weights(res[1], res_pot[1]);

                    std::vector<T *> resample_buf;
                    int nThreads = num_work_threads();
                    for (int i = 0; i < nThreads; ++i) {
                        resample_buf.push_back(new_array<T>(res_pot[1]));
                    }

                    parallel_for(res_pot[0], [&](uint32_t s, uint tid) {
                        T *workData = resample_buf[tid];
                        for (int t = 0; t < res_pot[1]; ++t) {
                            workData[t] = T();
                            for (int j = 0; j < 4; ++j) {
                                int offset = tWeights[t].firstTexel + j;
                                if (_image_wrap == ImageWrap::Repeat) {
                                    offset = Mod(offset, res[1]);
                                } else if (_image_wrap == ImageWrap::Clamp) {
                                    offset = clamp(offset, 0, (int) _resolution[1] - 1);
                                }
                                if (offset >= 0 && offset < (int) _resolution[1]) {
                                    workData[t] += T(tWeights[t].weight[j] *
                                                     U(resampled_image[offset * res_pot[0] + s]));
                                }
                            }
                        }

                        for (int t = 0; t < res_pot[1]; ++t) {
                            resampled_image[t * res_pot[0] + s] = Clamp(workData[t]);
                        }
                    }, 32);
                    for (auto ptr : resample_buf) {
                        delete[] ptr;
                    }
                    _resolution = make_uint2(res_pot);
                }

                int nLevels = 1 + log2_int(std::max(_resolution[0], _resolution[1]));

            }

            template<typename T>
            const T &texel(int level, int s, int t) const {

            }

            static std::unique_ptr<ResampleWeight[]> resample_weights(int old_res, int new_res) {
                std::unique_ptr<ResampleWeight[]> ret(new ResampleWeight[new_res]);
                float filter_width = 2.0f;
                for (int i = 0; i < new_res; ++i) {
                    float center = (i + 0.5f) * old_res / new_res;
                    ret[i].firstTexel = std::floor((center - filter_width) + 0.5f);
                    float weight_sum = 0;
                    for (int j = 0; j < 4; ++j) {
                        float pos = ret[i].firstTexel + j + 0.5f;
                        ret[i].weight[j] = lanczos((pos - center) / filter_width, 2);
                        weight_sum += ret[i].weight[j];
                    }
                    float invSumWts = 1 / weight_sum;
                    for (int j = 0; j < 4; ++j) {
                        ret[i].weight[j] *= invSumWts;
                    }
                }
                return ret;
            }
        };
    }
}