//
// Created by Zero on 2021/4/16.
//

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define TINYEXR_IMPLEMENTATION

#include "ext/tinyexr/tinyexr.h"
#include "core/logging.h"

namespace luminous {

    inline namespace utility {

        Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const luminous_fs::path &path)
                : ImageBase(pixel_format, res),
                  _path(path) {
            _pixel.reset(pixel);
        }

        Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res)
                : ImageBase(pixel_format, res) {
            _pixel.reset(pixel);
        }

        Image::Image(Image &&other) noexcept
                : ImageBase(other._pixel_format, other._resolution) {
            _pixel = move(other._pixel);
        }


        Image Image::pure_color(float4 color, ColorSpace color_space) {
            constexpr auto pixel_size = PixelFormatImpl<float4>::pixel_size;
            auto pixel = new std::byte[pixel_size];
            auto dest = (float4 *) pixel;
            if (color_space == ColorSpace::LINEAR) {
                *dest = color;
            } else {
                *dest = Spectrum::srgb_to_linear(color);
            }
            return {PixelFormat::RGBA32F, pixel, make_uint2(1u)};
        }

        Image Image::load(const luminous_fs::path &path, ColorSpace color_space, float3 scale) {
            auto extension = to_lower(path.extension().string());
            LUMINOUS_INFO("load picture ", path.string());
            if (extension == ".exr") {
                return load_exr(path, color_space, scale);
            } else if (extension == ".hdr") {
                return load_hdr(path, color_space, scale);
            } else {
                return load_other(path, color_space, scale);
            }
        }

        Image Image::load_hdr(const luminous_fs::path &path, ColorSpace color_space, float3 scale) {
            int w, h;
            int comp;
            auto path_str = luminous_fs::absolute(path).string();
            float *rgb = stbi_loadf(path_str.c_str(), &w, &h, &comp, 3);
            int pixel_num = w * h;
            PixelFormat pixel_format = detail::PixelFormatImpl<float4>::format;
            int pixel_size = detail::PixelFormatImpl<float4>::pixel_size;
            size_t size_in_bytes = pixel_num * pixel_size;
            auto pixel = new std::byte[size_in_bytes];
            float *src = rgb;
            auto dest = (float *) pixel;
            if (color_space == SRGB) {
                for (int i = 0; i < pixel_num; ++i, src += 3, dest += 4) {
                    dest[0] = Spectrum::srgb_to_linear(src[0]) * scale.x;
                    dest[1] = Spectrum::srgb_to_linear(src[1]) * scale.y;
                    dest[2] = Spectrum::srgb_to_linear(src[2]) * scale.z;
                    dest[3] = 1.f;
                }
            } else {
                for (int i = 0; i < pixel_num; ++i, src += 3, dest += 4) {
                    dest[0] = src[0] * scale.x;
                    dest[1] = src[1] * scale.y;
                    dest[2] = src[2] * scale.z;
                    dest[3] = 1.f;
                }
            }
            free(rgb);
            return Image(pixel_format, pixel, make_uint2(w, h), path);
        }

        Image Image::load_exr(const luminous_fs::path &fn, ColorSpace color_space, float3 scale) {
            // Parse OpenEXR
            EXRVersion exr_version;
            auto path_str = luminous_fs::absolute(fn).string();
            if (auto ret = ParseEXRVersionFromFile(&exr_version, path_str.c_str()); ret != 0) {
                LUMINOUS_EXCEPTION("Failed to parse OpenEXR version for file: ", fn.string());
            }

            if (exr_version.multipart) {
                LUMINOUS_EXCEPTION("Multipart OpenEXR format is not supported in file: ", fn.string());
            }
            // 2. Read EXR header
            EXRHeader exr_header;
            InitEXRHeader(&exr_header);
            const char *err = nullptr; // or `nullptr` in C++11 or later.
            FreeEXRErrorMessage(err);
            if (auto ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, path_str.c_str(), &err); ret != 0) {
                LUMINOUS_EXCEPTION("Failed to parse ", fn.string(), ": ", err);
            }
            auto predict = [](const EXRChannelInfo &channel) noexcept {
                return channel.pixel_type != TINYEXR_PIXELTYPE_FLOAT &&
                       channel.pixel_type != TINYEXR_PIXELTYPE_HALF;
            };
            if (exr_header.num_channels > 4 || exr_header.tiled ||
                std::any_of(exr_header.channels, exr_header.channels + exr_header.num_channels, predict)) {
                LUMINOUS_EXCEPTION("Unsupported pixel format in file: ", fn.string());
            }

            // Read HALF channel as FLOAT.
            for (int i = 0; i < exr_header.num_channels; i++) {
                if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
                    exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
                }
            }

            EXRImage exr_image;
            InitEXRImage(&exr_image);
            if (auto ret = LoadEXRImageFromFile(&exr_image, &exr_header, path_str.c_str(), &err); ret != 0) {
                LUMINOUS_EXCEPTION("Failed to load ", fn.string(), ": ", err);
            }
            size_t pixel_num = exr_image.width * exr_image.height;
            uint2 resolution = make_uint2(exr_image.width, exr_image.height);
            switch (exr_image.num_channels) {
                case 1: {
                    using PixelType = float;
                    PixelFormat pixel_format = detail::PixelFormatImpl<PixelType>::format;
                    PixelType *pixel = new PixelType[pixel_num];
                    size_t size_in_bytes = pixel_num * detail::PixelFormatImpl<PixelType>::pixel_size;
                    if (color_space == SRGB) {
                        for (int i = 0; i < pixel_num; ++i) {
                            PixelType val = reinterpret_cast<PixelType *>(exr_image.images[0])[i];
                            pixel[i] = Spectrum::srgb_to_linear(val) * scale.x;
                        }
                    } else {
                        for (int i = 0; i < pixel_num; ++i) {
                            PixelType val = reinterpret_cast<PixelType *>(exr_image.images[0])[i];
                            pixel[i] = val * scale.x;
                        }
                    }
                    return Image(pixel_format, (std::byte *) pixel, resolution, fn);
                }
                case 2: {
                    using PixelType = float2;
                    PixelFormat pixel_format = detail::PixelFormatImpl<PixelType>::format;
                    PixelType *pixel = new PixelType[pixel_num];
                    size_t size_in_bytes = pixel_num * detail::PixelFormatImpl<PixelType>::pixel_size;
                    if (color_space == SRGB) {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float2(
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[1])[i]) * scale.x,
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[0])[i]) * scale.y);
                        }
                    } else {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float2(
                                    reinterpret_cast<float *>(exr_image.images[1])[i] * scale.x,
                                    reinterpret_cast<float *>(exr_image.images[0])[i] * scale.y);
                        }
                    }
                    return Image(pixel_format, (std::byte *) pixel, resolution, fn);
                }
                case 3:
                case 4: {
                    PixelFormat pixel_format = detail::PixelFormatImpl<float4>::format;
                    auto pixel = new float4[pixel_num];
                    if (color_space == SRGB) {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float4(
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[3])[i]),
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[2])[i]),
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[1])[i]),
                                    1.f) * make_float4(scale, 1.f);
                        }
                    } else {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float4(
                                    (reinterpret_cast<float *>(exr_image.images[3])[i]),
                                    (reinterpret_cast<float *>(exr_image.images[2])[i]),
                                    (reinterpret_cast<float *>(exr_image.images[1])[i]),
                                    1.f) * make_float4(scale, 1.f);
                        }
                    }
                    return Image(pixel_format, (std::byte *) pixel, resolution, fn);
                }
                default:
                    LUMINOUS_ERROR("unknown")
            }
        }

        Image Image::load_other(const luminous_fs::path &path, ColorSpace color_space, float3 scale) {
            uint8_t *rgba;
            int w, h;
            int channel;
            auto fn = path.string();
            rgba = stbi_load(fn.c_str(), &w, &h, &channel, 4);
            if (!rgba) {
                throw std::runtime_error(fn + " load fail");
            }
            PixelFormat pixel_format = detail::PixelFormatImpl<uchar4>::format;
            int pixel_size = detail::PixelFormatImpl<uchar4>::pixel_size;
            size_t pixel_num = w * h;
            size_t size_in_bytes = pixel_size * pixel_num;
            uint2 resolution = make_uint2(w, h);
            auto pixel = new std::byte[size_in_bytes];
            uint8_t *src = rgba;
            auto dest = (uint32_t *) pixel;
            if (color_space == SRGB) {
                for (int i = 0; i < pixel_num; ++i, src += 4, dest += 1) {
                    float r = (float) src[0] / 255;
                    float g = (float) src[1] / 255;
                    float b = (float) src[2] / 255;
                    float a = (float) src[3] / 255;
                    float4 color = make_float4(r, g, b, a) * make_float4(scale, 1.f);
                    color = Spectrum::srgb_to_linear(color);
                    *dest = make_rgba(color);
                }
            } else {
                for (int i = 0; i < pixel_num; ++i, src += 4, dest += 1) {
                    float r = (float) src[0] / 255;
                    float g = (float) src[1] / 255;
                    float b = (float) src[2] / 255;
                    float a = (float) src[3] / 255;
                    float4 color = make_float4(r, g, b, a) * make_float4(scale, 1.f);
                    *dest = make_rgba(color);
                }
            }
            free(rgba);
            return Image(pixel_format, pixel, resolution, path);
        }

        void Image::save_image(const luminous_fs::path &path) {
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                save_exr(path);
            } else if (extension == ".hdr") {
                save_hdr(path);
            } else {
                save_other(path);
            }
            LUMINOUS_INFO("save picture ", path);
        }

        void Image::save_hdr(const luminous_fs::path &path) {
            convert_to_32bit();
            auto path_str = luminous_fs::absolute(path).string();
            stbi_write_hdr(path_str.c_str(), _resolution.x, _resolution.y, 4,
                           reinterpret_cast<const float *>(_pixel.get()));
        }

        void Image::save_exr(const luminous_fs::path &fn) {
            convert_to_32bit();
            EXRHeader header;
            InitEXRHeader(&header);

            EXRImage image;
            int c = 4;
            InitEXRImage(&image);
            int count = _resolution.x * _resolution.y;
            std::array<float *, 4> image_ptr{nullptr, nullptr, nullptr, nullptr};
            image.num_channels = 4;
            image.width = _resolution.x;
            image.height = _resolution.y;
            image.images = reinterpret_cast<uint8_t **>(image_ptr.data());

            std::array<int, 4> pixel_types{TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT,
                                           TINYEXR_PIXELTYPE_FLOAT};
            std::array<EXRChannelInfo, 4> channels{};
            header.num_channels = c;
            header.channels = channels.data();
            header.pixel_types = pixel_types.data();
            header.requested_pixel_types = pixel_types.data();

            std::vector<float> images;
            images.resize(c * count);
            image_ptr[0] = images.data();
            image_ptr[1] = image_ptr[0] + count;
            image_ptr[2] = image_ptr[1] + count;
            image_ptr[3] = image_ptr[2] + count;
            auto rgba = reinterpret_cast<const float4 *>(_pixel.get());
            for (int i = 0u; i < count; i++) {
                image_ptr[3][i] = rgba[i].x;
                image_ptr[2][i] = rgba[i].y;
                image_ptr[1][i] = rgba[i].z;
                image_ptr[0][i] = rgba[i].w;
            }
            strcpy_s(header.channels[0].name, "A");
            strcpy_s(header.channels[1].name, "B");
            strcpy_s(header.channels[2].name, "G");
            strcpy_s(header.channels[3].name, "R");
            const char *err = nullptr;
            if (auto ret = SaveEXRImageToFile(&image, &header, fn.string().c_str(), &err); ret != TINYEXR_SUCCESS) {
                LUMINOUS_EXCEPTION_IF("Failed to save texture as OpenEXR image: ", fn.string());
            }
        }

        void Image::save_other(const luminous_fs::path &path) {
            auto path_str = luminous_fs::absolute(path).string();
            auto extension = to_lower(path.extension().string());
            convert_to_8bit();
            if (extension == ".png") {
                stbi_write_png(path_str.c_str(), _resolution.x, _resolution.y, 4, _pixel.get(), 0);
            } else if (extension == ".bmp") {
                stbi_write_bmp(path_str.c_str(), _resolution.x, _resolution.y, 4, _pixel.get());
            } else if (extension == ".tga") {
                stbi_write_tga(path_str.c_str(), _resolution.x, _resolution.y, 4, _pixel.get());
            } else {
                // jpg or jpeg
                stbi_write_jpg(path_str.c_str(), _resolution.x, _resolution.y, 4, _pixel.get(), 100);
            }
        }

        void Image::convert_to_32bit() {
            if (is_32bit()) {
                return;
            }
            switch (_pixel_format) {
                case PixelFormat::R8U: {
                    using TargetType = float;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType)];
                    auto dest = (TargetType *) pixel;
                    for (int i = 0; i < pixel_num(); ++i, ++dest) {
                        *dest = float(_pixel[i]) / 255.f;
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::R8U;
                    break;
                }
                case PixelFormat::RG8U: {
                    using TargetType = float2;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType)];
                    auto src = (uint8_t *) _pixel.get();
                    auto dest = (TargetType *) pixel;
                    for (int i = 0; i < pixel_num(); ++i, ++dest, src += 2) {
                        *dest = make_float2(float(src[0]) / 255.f, float(src[1]) / 255.f);
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::RG32F;
                    break;
                }
                case PixelFormat::RGBA8U: {
                    using TargetType = float4;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType)];
                    auto src = (uint8_t *) _pixel.get();
                    auto dest = (TargetType *) pixel;
                    for (int i = 0; i < pixel_num(); ++i, ++dest, src += 4) {
                        *dest = make_float4(float(src[0]) / 255.f,
                                            float(src[1]) / 255.f,
                                            float(src[2]) / 255.f,
                                            float(src[3]) / 255.f);
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::RGBA32F;
                    break;
                }
                default:
                    LUMINOUS_EXCEPTION("unknown pixel type");
            }
        }

        void Image::convert_to_8bit() {
            if (is_8bit()) {
                return;
            }
            switch (_pixel_format) {
                case PixelFormat::R32F: {
                    using TargetType = uint8_t;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType)];
                    auto dest = (TargetType *) pixel;
                    auto src = (float *) _pixel.get();
                    for (int i = 0; i < pixel_num(); ++i, ++dest, ++src) {
                        *dest = make_8bit(src[0]);
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::R8U;
                    break;
                }
                case PixelFormat::RG32F: {
                    using TargetType = uint8_t;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType) * channel_num()];
                    auto dest = (uint8_t *) pixel;
                    auto src = (float *) _pixel.get();
                    for (int i = 0; i < pixel_num(); ++i, dest += 2, src += 2) {
                        dest[0] = make_8bit(src[0]);
                        dest[1] = make_8bit(src[1]);
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::RG8U;
                    break;
                }
                case PixelFormat::RGBA32F: {
                    using TargetType = uint32_t;
                    auto pixel = new std::byte[pixel_num() * sizeof(TargetType)];
                    auto dest = (TargetType *) pixel;
                    auto src = (float4 *) _pixel.get();
                    for (int i = 0; i < pixel_num(); ++i, ++dest, ++src) {
                        *dest = make_rgba(*src);
                    }
                    _pixel.reset(pixel);
                    _pixel_format = PixelFormat::RGBA8U;
                    break;
                }
                default:
                    break;
            }
        }

        bool Image::is_8bit() const {
            return _pixel_format == PixelFormat::R8U
                   || _pixel_format == PixelFormat::RG8U
                   || _pixel_format == PixelFormat::RGBA8U;
        }

        bool Image::is_32bit() const {
            return _pixel_format == PixelFormat::R32F
                   || _pixel_format == PixelFormat::RG32F
                   || _pixel_format == PixelFormat::RGBA32F;
        }


    }

} // luminous