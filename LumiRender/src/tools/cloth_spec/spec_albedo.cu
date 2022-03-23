#include "base_libs/common.h"
#include "base_libs/math/constants.h"
#include "render/scattering/microfacet.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace luminous {
namespace tools {

LM_GPU_INLINE float fresnel_schlick_weight(float cos_theta_h) {
    float one_minus_theta = clamp(cos_theta_h, .0f, 1.0f);
    return Pow<5>(one_minus_theta);
}

LM_GPU_INLINE float radicalInverse_VdC(uint32_t bits) {
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

LM_GPU_INLINE float2 hammersley2d(uint32_t i, float inv_nsample) {
     return float2(float(i) * inv_nsample, radicalInverse_VdC(i));
}

LM_GPU_INLINE float3 hemisphereSample_uniform(float u, float v) {
     float phi = v * 2.0 * Pi;
     float cosTheta = u;
     float sinTheta = sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
     return float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

LM_GPU_INLINE float hemispehreSample_uniform_pdf(float3 w) {
    return inv2Pi;
}

LM_GPU_INLINE float3 sampling_vector(float cos_theta_o) {

  // Uniformly sample point on a hemisphere.
  float sin_theta_o = sqrt(std::max(1.0f - cos_theta_o * cos_theta_o, .0f));
  return float3(sin_theta_o, 0.f, cos_theta_o);
}

extern "C" __global__ void cloth_spec_albedo_kernel(
  uint nsample,
  uint2 image_dim,
  cudaSurfaceObject_t albedo_image
) {
    float alpha;
    float2 u;
    float D, G, Fc;
    float pdf;
    float inv_nsample = 1.0f / nsample;
    float3 wo, wh, wi;
    float2 albedo(.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < image_dim.x && y < image_dim.y) {

        wo = sampling_vector(float(x + 1) / image_dim.x);
        alpha = float(y + 1) / image_dim.y;

        for (uint i = 0; i < nsample; ++i) {

            u = hammersley2d(i, inv_nsample);
            wh = hemisphereSample_uniform(u.x, u.y);
            wi = reflect(wo, wh);
            pdf = hemispehreSample_uniform_pdf(wh);

            if(same_hemisphere(wi, wh)) {
                D = microfacet::D_Charlie(wh, alpha);
                G = microfacet::G_Neubelt_soften(wo, wi, alpha);

                Fc = fresnel_schlick_weight(Frame::cos_theta(wh));

                albedo +=  float2(1.0f - Fc, Fc) * (D * G / (4.0f * pdf));
            }
        }

        albedo /= (Frame::cos_theta(wo) * nsample);

        surf2Dwrite(::make_float2(albedo.x, albedo.y), albedo_image, x*8, y);
    }
}

extern "C" __global__ void cloth_spec_albedo_avg_kernel(
  cudaSurfaceObject_t albedo_image,
  uint2 image_dim,
  cudaSurfaceObject_t albedo_avg_image
) {

    ::float2 albedo_intrin;
    float2 albedo;
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    float2 albedo_sum(.0f);
    float cos_theta;
    float inv_width;

    if(x < image_dim.y) {
        inv_width = 1.0f / image_dim.x;
        for (uint i = 0; i < image_dim.x; ++i) {
            albedo_intrin = surf2Dread<::float2>(albedo_image, i*8, x);
            albedo = make_float2(albedo_intrin.x, albedo_intrin.y);
            cos_theta = std::min((i + 1) * inv_width, 1.0f);

            albedo_sum += albedo * (2.0f  * cos_theta);
        }

        albedo_sum *= inv_width;

        surf2Dwrite(::make_float2(albedo_sum.x, albedo_sum.y), albedo_avg_image, x*8, 0);
    }
}

}
}