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

        Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const std::filesystem::path &path)
                : _pixel_format(pixel_format),
                _resolution(res),
                _path(path) {
            _pixel.reset(pixel);
        }

        Image::Image(Image &&other) noexcept {
            _pixel_format = other._pixel_format;
            _resolution = other._resolution;
            _pixel = move(other._pixel);
        }

        uint2 Image::resolution() const {
            return _resolution;
        }

        PixelFormat Image::pixel_format() const {
            return _pixel_format;
        }

        size_t Image::pixel_num() const {
            return _resolution.x * _resolution.y;
        }

        int Image::channels() const {
            if (_pixel_format == PixelFormat::R8U || _pixel_format == PixelFormat::R32F) { return 1u; }
            if (_pixel_format == PixelFormat::RG8U || _pixel_format == PixelFormat::RG32F) { return 2u; }
            return 4u;
        }

        Image Image::load(const filesystem::path &path, ColorSpace color_space) {
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                return load_exr(path, color_space);
            } else if (extension == ".hdr") {
                return load_hdr(path, color_space);
            } else {
                return load_other(path, color_space);
            }
        }

        Image Image::load_hdr(const filesystem::path &path, ColorSpace color_space) {
            int w, h;
            int comp;
            auto path_str = std::filesystem::absolute(path).string();
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
                    dest[0] = Spectrum::srgb_to_linear(src[0]);
                    dest[1] = Spectrum::srgb_to_linear(src[1]);
                    dest[2] = Spectrum::srgb_to_linear(src[2]);
                }
            } else {
                for (int i = 0; i < pixel_num; ++i, src += 3, dest += 4) {
                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest[2] = src[2];
                }
            }
            free(rgb);
            return Image(pixel_format, pixel, make_uint2(w, h), path);
        }

        Image Image::load_exr(const filesystem::path &fn, ColorSpace color_space) {
            // Parse OpenEXR
            EXRVersion exr_version;
            auto path_str = std::filesystem::absolute(fn).string();
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
                            pixel[i] = Spectrum::srgb_to_linear(val);
                        }
                    } else {
                        std::memcpy(pixel, exr_image.images[0], size_in_bytes);
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
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[1])[i]),
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[0])[i]));
                        }
                    } else {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float2(
                                    reinterpret_cast<float *>(exr_image.images[1])[i],
                                    reinterpret_cast<float *>(exr_image.images[0])[i]);
                        }
                    }
                    return Image(pixel_format, (std::byte *) pixel, resolution, fn);
                }
                case 3:
                case 4: {
                    PixelFormat pixel_format = detail::PixelFormatImpl<float4>::format;
                    auto pixel = new float4[pixel_num];
                    size_t size_in_bytes = pixel_num * detail::PixelFormatImpl<float4>::pixel_size;
                    if (color_space == SRGB) {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float4(
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[3])[i]),
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[2])[i]),
                                    Spectrum::srgb_to_linear(reinterpret_cast<float *>(exr_image.images[1])[i]),
                                    1.f);
                        }
                    } else {
                        for (int i = 0; i < pixel_num; ++i) {
                            pixel[i] = make_float4(
                                    (reinterpret_cast<float *>(exr_image.images[3])[i]),
                                    (reinterpret_cast<float *>(exr_image.images[2])[i]),
                                    (reinterpret_cast<float *>(exr_image.images[1])[i]),
                                    1.f);
                        }
                    }
                    return Image(pixel_format, (std::byte *)pixel, resolution, fn);
                }
                default:
                    LUMINOUS_ERROR("unknown")
            }
        }

        Image Image::load_other(const filesystem::path &path, ColorSpace color_space) {
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
            auto dest = (uint8_t *) pixel;
            if (color_space == SRGB) {
                for (int i = 0; i < pixel_num; ++i, src += 4, dest += 4) {
                    float r = (float) src[0] / 255;
                    float g = (float) src[1] / 255;
                    float b = (float) src[2] / 255;
                    float a = (float) src[3] / 255;
                    float4 color = make_float4(r, g, b, a);
                    color = Spectrum::srgb_to_linear(color);
                    *dest = make_rgba(color);
                }
            } else {
                std::memcpy(pixel, rgba, size_in_bytes);
            }
            free(rgba);
            return Image(pixel_format, pixel, resolution, path);
        }

        void Image::save_image(const filesystem::path &path) {
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                save_exr(path);
            } else if (extension == ".hdr") {
                save_hdr(path);
            } else {
                save_other(path);
            }
        }

        void Image::save_hdr(const filesystem::path &path) {

        }

        void Image::save_exr(const filesystem::path &fn) {

        }

        void Image::save_other(const filesystem::path &path) {

        }

        void Image::convert_to(PixelFormat pixel_format) {
            
        }

    }

} // luminous