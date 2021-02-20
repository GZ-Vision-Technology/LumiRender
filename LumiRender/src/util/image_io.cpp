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

            return pair(nullptr, make_int2(w, h));
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

        void save_other(const filesystem::path &path, RGBSpectrum * p, int2 resolution) {


        }
    }
}