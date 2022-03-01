//
// Created by Zero on 16/02/2022.
//


#pragma once

#include "base_libs/common.h"

#ifdef LUMINOUS_DENOISER_BUILD
#ifdef _WIN32
#define LUMINOUS_DENOISER_LIB_VISIBILITY  __declspec(dllexport)
#else
#define LUMINOUS_DENOISER_LIB_VISIBILITY __attribute__((visibility("default")))
#endif
#else
#ifdef _WIN32
#define LUMINOUS_DENOISER_LIB_VISIBILITY __declspec(dllimport)
#else
#define LUMINOUS_DENOISER_LIB_VISIBILITY
#endif
#endif

namespace oidn {
class DeviceRef;
};

namespace luminous {
    inline namespace denoise {
        class LUMINOUS_DENOISER_LIB_VISIBILITY Denoiser {
        private:
            Denoiser(const Denoiser &) = delete;
            Denoiser &operator=(const Denoiser &) = delete;

            oidn::DeviceRef *_device{};
        public:
            Denoiser();
            ~Denoiser();

            void execute(uint2 res, float4 *output, float4 *color,
                         float4 *normal = nullptr, float4 *albedo = nullptr);
        };
    }
}