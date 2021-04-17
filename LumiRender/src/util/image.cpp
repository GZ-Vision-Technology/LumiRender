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

        Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res)
                : _pixel_format(pixel_format), _resolution(res) {
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
                for (int i = 0; i < pixel_num; ++i, src += 3, dest += 3) {
                    dest[0] = Spectrum::srgb_to_linear(src[0]);
                    dest[1] = Spectrum::srgb_to_linear(src[1]);
                    dest[2] = Spectrum::srgb_to_linear(src[2]);
                }
            } else {
                std::memcpy(pixel, rgb, size_in_bytes);
            }
            free(rgb);
            return Image(pixel_format, pixel, make_uint2(w, h));
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
            switch (exr_image.num_channels) {
                case 1: {
                    PixelFormat pixel_format = detail::PixelFormatImpl<float>::format;
                    float *pixel = new float[pixel_num];
                    size_t size_in_bytes = pixel_num * detail::PixelFormatImpl<float>::pixel_size;
                    if (color_space == SRGB) {
                        for (int i = 0; i < pixel_num; ++i) {
                            float val = reinterpret_cast<float *>(exr_image.images[0])[i];
                            pixel[i] = Spectrum::srgb_to_linear(val);
                        }
                    } else {
                        std::memcpy(pixel, exr_image.images[0], size_in_bytes);
                    }
                    return Image(pixel_format, (std::byte *) pixel, make_uint2(exr_image.width, exr_image.height));
                }

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
            return Image(pixel_format, pixel, resolution);
        }

        void Image::save_image(const filesystem::path &fn) {

        }

        void Image::save_hdr(const filesystem::path &fn) {

        }

        void Image::save_exr(const filesystem::path &fn) {

        }

        void Image::save_other(const filesystem::path &fn) {

        }
    }

} // luminous