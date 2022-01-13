#include "disney.h"
#include "material.h"

// standarded by merl file format
#define BRDF_SAMPLING_RES_THETA_H 90
#define BRDF_SAMPLING_RES_THETA_D 90
#define BRDF_SAMPLING_RES_PHI_D_DIV2 180


namespace luminous {

  void set_cartesian_coords(float theta_h, float theta_d, float phi_d,
    float3 *wi, float3 *wo) {

      // for isotropic material, assume phi_h = 0
      float sin_theta_h = std::sin(theta_h);

      float3 normal{.0f, std::cos(theta_h), sin_theta_h};
      float3 tangent{0.0f, 0.0f, 1.0f};
      float3 bitangent = cross(normal, tangent);
      tangent = cross(bitangent, normal);

      float sin_theta_wo = std::sin(theta_d);
      float3 wo_l{sin_theta_wo * std::sin(theta_d), std::cos(theta_d), sin_theta_wo * std::cos(theta_d)};

      *wo = wo_l.x * tangent + wo_l.y * bitangent + wo_l.z * normal;
      *wi = 2.0f*dot(*wo, normal) * normal - *wo;
    }

  int export_merl_binary_file(const Material *mat, const char *fpath) {

      constexpr float M_PI = 3.1415926535897932384626433832795f;
      constexpr float M_PI_DIV_2 = 1.5707963267948966192313216916398f;
      constexpr int brdfChannelSize = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D_DIV2;

      float theta_h,
              theta_d, phi_d;
      float3 wo, wi;
      std::unique_ptr<float[]> brdfSamples{new float[3 * brdfChannelSize]};

      for (int i = 0; i < BRDF_SAMPLING_RES_THETA_H; ++i) {
          float theta_h = M_PI_DIV_2 * i / BRDF_SAMPLING_RES_THETA_H;
          for (int j = 0; j < BRDF_SAMPLING_RES_THETA_D; ++j) {
              float theta_d = M_PI_DIV_2 * j / BRDF_SAMPLING_RES_THETA_D;
              for (int k = 0; k < BRDF_SAMPLING_RES_PHI_D_DIV2; ++k) {
                  float phi_d = M_PI * k / BRDF_SAMPLING_RES_PHI_D_DIV2;
                  float3 wi, wo;
                  set_cartesian_coords(theta_h, theta_d, phi_d, &wi, &wo);

                  int ind = k + j * BRDF_SAMPLING_RES_PHI_D_DIV2 + i * BRDF_SAMPLING_RES_THETA_D;
                  Spectrum brdf;
                  // Compute brdf values here.

                  brdfSamples[ind] = brdf.R();
                  brdfSamples[ind + brdfChannelSize] = brdf.G();
                  brdfSamples[ind + brdfChannelSize*2] = brdf.B();
              }
          }
      }

      uint32_t dims[3] = {BRDF_SAMPLING_RES_PHI_D_DIV2, BRDF_SAMPLING_RES_THETA_D, BRDF_SAMPLING_RES_THETA_H };

      std::ofstream fout{fpath, std::ios::binary};

      if(!fout)
          return -1;

      fout.write((char *)dims, sizeof(dims));
      fout.write((char *)brdfSamples.get(), 3 * brdfChannelSize);
      fout.close();

      return 0;
  }

}