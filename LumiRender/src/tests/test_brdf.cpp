#include  "../render/materials/disney.h"
#include "cxxopts.hpp"
#include <iostream>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

luminous::float3 baseColor;
float metallic;
float specular;
float roughness;
float specularTint;
float anisotropic;
float sheen;
float sheenTint;
float clearcoat;
float clearcoatGloss;

std::string outputFilePath;

luminous::float3 string_tuple_to_float3(const char *str) {

    const char *p0 = str;
    float v[3] = { .0f, .0f, .0f };
    int i = 0;
    std::string tmp;

    for (; *str && i < 3; ++str) {
      if(*str == ',') {
          tmp = std::string(p0, str);
          v[i++] = strtod(tmp.c_str(), nullptr);
          p0 = ++str;
      }
    }

    if(p0 != str) {
        tmp = std::string(p0, str);
        v[i] = strtod(tmp.c_str(), nullptr);
    }

    return luminous::float3(v[0], v[1], v[2]);
}

void validate_args(int argc, char **argv) {

    cxxopts::Options options{argv[0]};

    options.add_options(
            "disney brdf arguments",
            {{"baseColor", "base color", cxxopts::value<std::string>()->default_value("1,1,1")},
             { "metallic", "metallic", cxxopts::value<double>()->default_value("0.0") },
             {"specular", "specular factor", cxxopts::value<double>()->default_value("0.0")},
             {"roughness", "roughness", cxxopts::value<double>()->default_value("0.0")},
             {"specularTint", "specular tint", cxxopts::value<double>()->default_value("0.0")},
             {"anisotropic", "anisotropic", cxxopts::value<double>()->default_value("0.0")},
             {"sheen", "sheen", cxxopts::value<double>()->default_value("0.0")},
             {"sheenTint", "sheen tint", cxxopts::value<double>()->default_value("0.0")},
             {"clearcoat", "clearcoat", cxxopts::value<double>()->default_value("0.0")},
             {"clearcoatGloss", "clearcoat tint", cxxopts::value<double>()->default_value("0.0")}});

    options.add_options(
            "output arguments",
            {{"o,output", "output file path", cxxopts::value<std::string>()}});

    auto result = options.parse(argc, argv);
    if(argc < 2 || result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    baseColor = string_tuple_to_float3(result["baseColor"].as<std::string>().data());
    metallic = (float) result["metallic"].as<double>();
    specular = (float) result["specular"].as<double>();
    roughness = (float) result["roughness"].as<double>();
    specularTint = (float) result["specularTint"].as<double>();
    anisotropic = (float) result["anisotropic"].as<double>();
    sheen = (float) result["sheen"].as<double>();
    sheenTint = (float) result["sheenTint"].as<double>();
    clearcoat = (float) result["clearcoat"].as<double>();
    clearcoatGloss = (float) result["clearcoatGloss"].as<double>();

    outputFilePath = result["output"].as<std::string>();

    if (outputFilePath.empty()) {
        std::cout << "Unknown output path" << std::endl;
        exit(-1);
    }
}

luminous::BSDFWrapper create_bsdf() {

    luminous::BSDFWrapper bsdf;

    using namespace luminous;

    // todo merge texture
    luminous::float4 color = make_float4(baseColor, 1.0f);
    float metallic = ::metallic;
    float eta = 1.0;
    float spec_trans = 0.0;
    float diffuse_weight = (1 - metallic) * (1 - spec_trans);
    float diff_trans = 1.0 / 2.f;
    float spec_tint = specularTint;
    float roughness = ::roughness;
    float lum = Spectrum{color}.Y();
    Spectrum color_tint = lum > 0 ? (color / lum) : Spectrum(1.f);
    float sheen_weight = sheen;
    float sheen_tint = sheenTint;
    Spectrum color_sheen_tint = sheen_weight > 0.f ? lerp(sheen_tint, Spectrum{1.f}, color_tint) : Spectrum(0.f);
    luminous::float4 scatter_distance = make_float4(.0f);
    Spectrum R0 = lerp(metallic,
                       schlick_R0_from_eta(eta) * lerp(spec_tint, Spectrum{1.f}, color_sheen_tint),
                       Spectrum(color));
    float clearcoat = ::clearcoat;
    float clearcoat_gloss = clearcoatGloss;
    DisneyBSDF disney_bsdf;
    BSDFHelper helper;
    helper.set_roughness(roughness);
    helper.set_metallic(metallic);
    helper.set_R0(R0);
    helper.set_eta(eta);
    helper.set_clearcoat_alpha(lerp(clearcoat_gloss, 1, 0.001f));

    disney_bsdf.set_data(helper);

    if (0) {
        float flatness = .0f;
        disney::Diffuse diffuse(diffuse_weight * (1 - flatness) * (1 - diff_trans) * color);
        disney_bsdf.add_BxDF(diffuse);
        disney::FakeSS fake_ss(diffuse_weight * flatness * (1 - diff_trans) * color);
        disney_bsdf.add_BxDF(fake_ss);
    } else {
        if (is_zero(scatter_distance)) {
            disney_bsdf.add_BxDF(disney::Diffuse(diffuse_weight * color));
        } else {
            disney_bsdf.add_BxDF(SpecularTransmission(Spectrum{1.f}));
            // todo process BSSRDF
        }
    }

    disney_bsdf.add_BxDF(disney::Retro(diffuse_weight * color));

    disney_bsdf.add_BxDF(disney::Sheen(diffuse_weight * sheen_weight * color_sheen_tint));

    float aspect = safe_sqrt(1 - anisotropic * 0.9f);
    float ax = std::max(0.001f, sqr(roughness) / aspect);
    float ay = std::max(0.001f, sqr(roughness) * aspect);
    Microfacet distrib{ax, ay, MicrofacetType::Disney};
    disney_bsdf.add_BxDF(MicrofacetReflection(Spectrum{1.f}, distrib));
    disney_bsdf.add_BxDF(disney::Clearcoat{clearcoat});

    Spectrum T = spec_trans * sqrt(color);
    if (0) {
        float rscaled = (0.65f * eta - 0.35f) * roughness;
        float ax = std::max(0.001f, sqr(rscaled) / aspect);
        float ay = std::max(0.001f, sqr(rscaled) * aspect);
        Microfacet distrib{ax, ay, GGX};
        disney_bsdf.add_BxDF(MicrofacetTransmission{T, distrib});
        disney_bsdf.add_BxDF(LambertTransmission(diff_trans * color));

    } else {
        disney_bsdf.add_BxDF(MicrofacetTransmission{T, distrib});
    }

    return {luminous::make_float3(.0f, 0.0f, 1.0f), luminous::make_float3(.0f, .0f, 1.0f), luminous::make_float3(1.0f, 0.0f, 0.0f), BSDF{disney_bsdf}};
}

// standarded by merl file format
#define BRDF_SAMPLING_RES_THETA_H 90
#define BRDF_SAMPLING_RES_THETA_D 90
#define BRDF_SAMPLING_RES_PHI_D_DIV2 180


namespace luminous {

    constexpr float M_PI = 3.1415926535897932384626433832795f;
    constexpr float M_PI_DIV_2 = 1.5707963267948966192313216916398f;

    void set_cartesian_coords(float theta_h, float theta_d, float phi_d,
                              float3 *wi, float3 *wo) {

        // for isotropic material, assume phi_h = 0
        constexpr float phi_h = M_PI / 4.0f;

        float sin_theta_h = std::sin(theta_h);

        float3 normal{sin_theta_h * std::cos(phi_d), sin_theta_h * sin(phi_d), std::cos(theta_h)};
        float3 tangent;
        float3 bitangent;
        if(normal.z < 1.0e-4) {
            tangent = float3(0.0f, 0.0f, -1.0f);
            bitangent = normalize(cross(normal, tangent));
        } else {
            tangent = float3(normal.x / sin_theta_h, normal.y / sin_theta_h, 0.0f);
            bitangent = normalize(cross(normal, tangent));
        }
        tangent = cross(bitangent, normal);

        float sin_theta_wo = std::sin(theta_d);
        float3 wo_l{sin_theta_wo * std::cos(phi_d), sin_theta_wo * std::sin(phi_d), std::cos(theta_d)};

        *wo = normalize(wo_l.x * tangent + wo_l.y * bitangent + wo_l.z * normal);
        *wi = normalize(2.0f * dot(*wo, normal) * normal - *wo);
    }

    int export_merl_binary_file(const BSDFWrapper &bsdf, const char *fpath) {

        constexpr int brdfChannelSize = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D_DIV2;

        float theta_h, theta_d, phi_d;
        float3 wo, wi;
        std::unique_ptr<double[]> brdfSamples{new double[3 * brdfChannelSize]};

        for (int i = 0; i < BRDF_SAMPLING_RES_THETA_H; ++i) {
            float theta_h = M_PI_DIV_2 * i / BRDF_SAMPLING_RES_THETA_H;
            for (int j = 0; j < BRDF_SAMPLING_RES_THETA_D; ++j) {
                float theta_d = M_PI_DIV_2 * j / BRDF_SAMPLING_RES_THETA_D;
                for (int k = 0; k < BRDF_SAMPLING_RES_PHI_D_DIV2; ++k) {
                    float phi_d = M_PI * k / BRDF_SAMPLING_RES_PHI_D_DIV2;
                    float3 wi, wo;
                    set_cartesian_coords(theta_h, theta_d, phi_d, &wi, &wo);

                    int ind = k + j * (BRDF_SAMPLING_RES_PHI_D_DIV2 + i * BRDF_SAMPLING_RES_THETA_D);
                    Spectrum brdf = bsdf.eval(wo, wi);

                    brdfSamples[ind] = static_cast<double>(brdf.R());
                    brdfSamples[ind + brdfChannelSize] = static_cast<double>(brdf.G());
                    brdfSamples[ind + brdfChannelSize * 2] = static_cast<double>(brdf.B());
                }
            }
        }

        uint32_t dims[3] = {BRDF_SAMPLING_RES_PHI_D_DIV2, BRDF_SAMPLING_RES_THETA_D, BRDF_SAMPLING_RES_THETA_H};

        std::ofstream fout{fpath, std::ios::binary};

        if (!fout)
            return -1;

        fout.write((char *) dims, sizeof(dims));
        fout.write((char *) brdfSamples.get(), 3 * brdfChannelSize * sizeof(double));
        fout.close();

        return 0;
    }

}// namespace luminous

int main(int argc, char **argv) {

    validate_args(argc, argv);

    auto bsdf = create_bsdf();

    luminous::export_merl_binary_file(bsdf, outputFilePath.c_str());

    return 0;
}