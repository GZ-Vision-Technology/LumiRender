//
// Created by Zero on 2021/2/20.
//

#include "image_io.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace luminous {
    inline namespace utility {
        pair<RGBSpectrum *, int2> load_image(const filesystem::path &path) {
            auto path_str = std::filesystem::absolute(path).string();
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                return load_exr(path);
            } else if (extension == ".hdr") {
                return load_hdr(path);
            } else {
                return load_other(path);
            }
        }

        pair<RGBSpectrum *, int2> load_hdr(const filesystem::path &fn) {
            return pair(nullptr, make_int2());
        }

        pair<RGBSpectrum *, int2> load_exr(const filesystem::path &fn) {
            return pair(nullptr, make_int2());
        }

        pair<RGBSpectrum *, int2> load_other(const filesystem::path &path) {
            unsigned char *c_rgb;
            int w, h;
            int channel;
            auto fn = path.string();
            c_rgb = stbi_load(fn.c_str(), &w, &h , &channel , 4);
            if (!c_rgb) {
                throw std::runtime_error(fn + " load fail");
            }
            RGBSpectrum *rgb = new RGBSpectrum[w * h];
            unsigned char *src = c_rgb;
            for (int i = 0; i < w * h; ++i, src += 4) {
                float r = src[0] / 255.f;
                float g = src[1] / 255.f;
                float b = src[2] / 255.f;
                rgb[i] = RGBSpectrum(r,g,b);
            }
            return pair(rgb, make_int2(w, h));
        }

        void save_image(const filesystem::path &path, RGBSpectrum * rgb, int2 resolution) {
            auto path_str = std::filesystem::absolute(path).string();
            auto extension = to_lower(path.extension().string());
            if (extension == ".exr") {
                save_exr(path, rgb, resolution);
            } else if (extension == ".hdr") {
                save_hdr(path, rgb, resolution);
            } else {
                save_other(path, rgb, resolution);
            }
        }

        void save_hdr(const filesystem::path &fn, RGBSpectrum * rgb, int2 resolution) {

        }

        void save_exr(const filesystem::path &fn, RGBSpectrum * rgb, int2 resolution) {

        }

        void save_other(const filesystem::path &path, RGBSpectrum * rgb, int2 resolution) {
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
                stbi_write_jpg(path_str.c_str(), resolution.x, resolution.y, 4, p, 100);
            }
        }
    }
}