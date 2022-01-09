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
            index_t _pyramid_index{invalid_uint32};
            ImageWrap _image_wrap{};

        public:
            explicit MIPMap(const Image &image, ImageWrap image_wrap = ImageWrap::Repeat,
                            float max_anisotropy = 8.f, bool tri_linear = true)
                    : ImageBase(image.pixel_format(), image.resolution()),
                      _image_wrap(image_wrap),
                      _max_anisotropy(max_anisotropy),
                      _tri_linear(tri_linear) {
                init(image);
            }

            void init(const Image &image) {
                switch (pixel_format()) {
//                    case PixelFormat::R8U: {
//                        gen_pyramid<uchar, float>(reinterpret_cast<const uchar *>(image.pixel_ptr()));
//                        break;
//                    }
//                    case PixelFormat::RG8U: {
//                        gen_pyramid<uchar2, float2>(reinterpret_cast<const uchar2 *>(image.pixel_ptr()));
//                        break;
//                    }
//                    case PixelFormat::RGBA8U: {
//                        gen_pyramid<uchar4, float4>(reinterpret_cast<const uchar4 *>(image.pixel_ptr()));
//                        break;
//                    }
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
            LM_NODISCARD T clamp_pixel(T val) {
                return val;
            }

            template<typename T>
            void gen_pyramid_index() {
                _pyramid_index = PyramidMgr::instance()->template generate_empty_pyramid<T>();
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
                                    float weight = sWeights[s].weight[j];
                                    auto pix = img[t * res[0] + origS];
                                    auto val = T(sWeights[s].weight[j] *
                                                 U(pix));
                                    resampled_image[t * res_pot[0] + s] += val;
                                }
                            }
                        }
                    }, 16);

                    std::unique_ptr<ResampleWeight[]> tWeights = resample_weights(res[1], res_pot[1]);

                    std::vector<T *> resample_buf;
                    int nThreads = num_work_threads();
                    resample_buf.reserve(nThreads);
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
                            resampled_image[t * res_pot[0] + s] = clamp_pixel(workData[t]);
                        }
                    }, 32);
                    for (auto ptr : resample_buf) {
                        delete[] ptr;
                    }
                    _resolution = make_uint2(res_pot);
                }

                int nLevels = 1 + log2_int(std::max(_resolution[0], _resolution[1]));

                Pyramid<T> &pyramid = get_pyramid<T>(_pyramid_index);
                pyramid.reserve(nLevels);

                pyramid.emplace_back(_resolution, resampled_image.get());

                for (int i = 1; i < nLevels; ++i) {
                    uint sRes = std::max(1, pyramid.at(i - 1).u_size() / 2);
                    uint tRes = std::max(1, pyramid.at(i - 1).v_size() / 2);

                    pyramid.emplace_back(make_uint2(sRes, tRes));
                    BlockedArray<T> &layer = pyramid.at(i);
                    parallel_for(tRes, [&](uint32_t t, uint32_t _) {
                        for (int s = 0; s < sRes; ++s) {
                            layer(s, t) = .25f *
                                    (texel<T>(i - 1, 2 * s, 2 * t) +
                                    texel<T>(i - 1, 2 * s + 1, 2 * t) +
                                    texel<T>(i - 1, 2 * s, 2 * t + 1) +
                                    texel<T>(i - 1, 2 * s + 1, 2 * t + 1));
                        }
                    }, 16);
                }
            }

            template<typename T>
            LM_NODISCARD Pyramid<T> &get_pyramid(index_t index) {
                return PyramidMgr::instance()->template get_pyramid<T>(index);
            }

            template<typename T>
            LM_NODISCARD const Pyramid<T> &get_pyramid(index_t index) const {
                return PyramidMgr::instance()->template get_pyramid<T>(index);
            }

            template<typename T>
            LM_NODISCARD const T &texel(int level, int s, int t) const {
                const Pyramid<T> &pyramid = get_pyramid<T>(_pyramid_index);
                DCHECK(level < pyramid.levels());
                const BlockedArray<T> &layer = pyramid.at(level);
                switch (_image_wrap) {
                    case ImageWrap::Repeat: {
                        s = Mod(s, layer.u_size());
                        t = Mod(t, layer.v_size());
                        break;
                    }
                    case ImageWrap::Clamp:{
                        s = clamp(s, 0, layer.u_size() - 1);
                        t = clamp(t, 0, layer.v_size() - 1);
                        break;
                    }
                    case ImageWrap::Black:{
                        static constexpr T black(0.f);
                        if (s < 0 || s >= layer.u_size() || t < 0 || t > layer.v_size()) {
                            return black;
                        }
                        break;
                    }
                }
                return layer(s, t);
            }

            static std::unique_ptr<ResampleWeight[]> resample_weights(int old_res, int new_res) {
                std::unique_ptr<ResampleWeight[]> ret(new ResampleWeight[new_res]);
                float filter_width = 2.0f;
                for (int i = 0; i < new_res; ++i) {
                    float center = (i + 0.5f) * float(old_res) / float(new_res);
                    ret[i].firstTexel = std::floor((center - filter_width) + 0.5f);
                    float weight_sum = 0;
                    for (int j = 0; j < 4; ++j) {
                        float pos = float(ret[i].firstTexel + j) + 0.5f;
                        ret[i].weight[j] = lanczos((pos - center) / filter_width, 2);
                        weight_sum += ret[i].weight[j];
                    }
                    float invSumWts = 1 / weight_sum;
                    for (float &j : ret[i].weight) {
                        j *= invSumWts;
                    }
                }
                return ret;
            }
        };
    }
}