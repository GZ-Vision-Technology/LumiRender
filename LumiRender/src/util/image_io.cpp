//
// Created by Zero on 2021/2/20.
//

#include "image_io.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define TINYEXR_IMPLEMENTATION

#include "ext/tinyexr/tinyexr.h"
#include "core/logging.h"

namespace luminous {
    inline namespace utility {
        pair<RGBSpectrum *, int2> load_image(const filesystem::path &path) {
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                return load_exr(path);
            } else if (extension == ".hdr") {
                return load_hdr(path);
            } else {
                return load_other(path);
            }
        }

        pair<RGBSpectrum *, int2> load_hdr(const filesystem::path &path) {
            int w, h;
            int comp;
            auto path_str = std::filesystem::absolute(path).string();
            float *c_rgb = stbi_loadf(path_str.c_str(), &w, &h, &comp, 3);
            int count = w * h;
            RGBSpectrum *rgb = new RGBSpectrum[w * h];
            float *src = c_rgb;
            for (int i = 0; i < w * h; ++i, src += 3) {
                rgb[i] = RGBSpectrum(src[0], src[1], src[2]);
            }
            return make_pair(rgb, make_int2(w, h));
        }

        pair<RGBSpectrum *, int2> load_exr(const filesystem::path &fn) {
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
                std::any_of(exr_header.channels,exr_header.channels + exr_header.num_channels, predict)) {
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
            RGBSpectrum *rgb = new RGBSpectrum[exr_image.width * exr_image.height];
            switch (exr_image.num_channels) {
                case 1:
                    for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                        rgb[i] = RGBSpectrum(
                                reinterpret_cast<float *>(exr_image.images[0])[i],
                                reinterpret_cast<float *>(exr_image.images[0])[i],
                                reinterpret_cast<float *>(exr_image.images[0])[i]);
                    }
                    break;
                case 2:
                    LUMINOUS_EXCEPTION("unknow channel num in file ", fn.string());
                case 3:
                    for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                        rgb[i] = RGBSpectrum(
                                reinterpret_cast<float *>(exr_image.images[3])[i],
                                reinterpret_cast<float *>(exr_image.images[2])[i],
                                reinterpret_cast<float *>(exr_image.images[1])[i]);
                    }
                    break;
                case 4:
                    for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                        rgb[i] = RGBSpectrum(
                                reinterpret_cast<float *>(exr_image.images[3])[i],
                                reinterpret_cast<float *>(exr_image.images[2])[i],
                                reinterpret_cast<float *>(exr_image.images[1])[i]);
                    }
                    break;
                default:
                    break;
            }
            FreeEXRImage(&exr_image);
            return make_pair(rgb, make_int2(exr_image.width,exr_image.height));
        }

        pair<RGBSpectrum *, int2> load_other(const filesystem::path &path) {
            unsigned char *c_rgb;
            int w, h;
            int channel;
            auto fn = path.string();
            c_rgb = stbi_load(fn.c_str(), &w, &h, &channel, 4);
            if (!c_rgb) {
                throw std::runtime_error(fn + " load fail");
            }
            RGBSpectrum *rgb = new RGBSpectrum[w * h];
            unsigned char *src = c_rgb;
            for (int i = 0; i < w * h; ++i, src += 4) {
                float r = src[0] / 255.f;
                float g = src[1] / 255.f;
                float b = src[2] / 255.f;
                rgb[i] = RGBSpectrum(r, g, b);
            }
            free(c_rgb);
            return make_pair(rgb, make_int2(w, h));
        }

        void save_image(const filesystem::path &path, RGBSpectrum *rgb, int2 resolution) {
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                save_exr(path, rgb, resolution);
            } else if (extension == ".hdr") {
                save_hdr(path, rgb, resolution);
            } else {
                save_other(path, rgb, resolution);
            }
        }

        void save_hdr(const filesystem::path &path, RGBSpectrum *rgb, int2 resolution) {
            auto path_str = std::filesystem::absolute(path).string();
            stbi_write_hdr(path_str.c_str(), resolution.x, resolution.y, 4, reinterpret_cast<const float *>(rgb));
        }

        void save_exr(const filesystem::path &fn, RGBSpectrum *rgb, int2 resolution) {
            EXRHeader header;
            InitEXRHeader(&header);

            EXRImage image;
            int c = 4;
            InitEXRImage(&image);
            int count = resolution.x * resolution.y;
            std::array<float *, 4> image_ptr{nullptr, nullptr, nullptr, nullptr};
            image.num_channels = 4;
            image.width = resolution.x;
            image.height = resolution.y;
            image.images = reinterpret_cast<unsigned char **>(image_ptr.data());

            std::array<int, 4> pixel_types{TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT};
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
            auto rgba = reinterpret_cast<float4 *>(rgb);
            for (int i = 0u; i < count; i++) {
                image_ptr[3][i] = rgba[i].x;
                image_ptr[2][i] = rgba[i].y;
                image_ptr[1][i] = rgba[i].z;
                image_ptr[0][i] = rgba[i].w;
            }
            strcpy(header.channels[0].name, "A");
            strcpy(header.channels[1].name, "B");
            strcpy(header.channels[2].name, "G");
            strcpy(header.channels[3].name, "R");
            const char *err = nullptr;
            if (auto ret = SaveEXRImageToFile(&image, &header, fn.string().c_str(), &err); ret != TINYEXR_SUCCESS) {
                LUMINOUS_EXCEPTION_IF("Failed to save texture as OpenEXR image: ", fn.string());
            }
        }

        void save_other(const filesystem::path &path, RGBSpectrum *rgb, int2 resolution) {
            auto path_str = std::filesystem::absolute(path).string();
            auto extension = to_lower(path.extension().string());
            auto pixel_count = resolution.x * resolution.y;
            uint32_t *p = new uint32_t[pixel_count];
            for (int i = 0; i < resolution.x * resolution.y; ++i) {
                p[i] = make_rgba(rgb[i].vec());
            }
            if (extension == ".png") {
                stbi_write_png(path_str.c_str(), resolution.x, resolution.y, 4, p, 0);
            } else if (extension == ".bmp") {
                stbi_write_bmp(path_str.c_str(), resolution.x, resolution.y, 4, p);
            } else if (extension == ".tga") {
                stbi_write_tga(path_str.c_str(), resolution.x, resolution.y, 4, p);
            } else {
                // jpg or jpeg
                stbi_write_jpg(path_str.c_str(), resolution.x, resolution.y, 4, p, 100);
            }
            free(p);
        }
    }
}