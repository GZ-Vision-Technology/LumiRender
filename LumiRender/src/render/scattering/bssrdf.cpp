#include "bssrdf.h"
#include "base_libs/geometry/frame.h"
#include <cmath>
#include "core/memory/allocator.h"
#include "microfacet.h"

namespace luminous {
namespace render {

inline float SampleExponential(float u, float a) {
    DCHECK_GT(a, 0);
    return -std::log(1.0f - u) / a;
}

inline float HenyeyGreenstein(float cosTheta, float g) {
    float denom = 1 + sqr(g) + 2 * g * cosTheta;
    return inv4Pi * (1 - sqr(g)) / (denom * safe_sqrt(denom));
}

float IntegrateCatmullRom(BufferView<const float> nodes, BufferView<const float> f,
                          BufferView<float> cdf) {
    DCHECK_EQ(nodes.size(), f.size());
    float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < nodes.size() - 1; ++i) {
        // Look up $x_i$ and function values of spline segment _i_
        float x0 = nodes[i], x1 = nodes[i + 1];
        float f0 = f[i], f1 = f[i + 1];
        float width = x1 - x0;

        // Approximate derivatives using finite differences
        float d0 = (i > 0) ? width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) : (f1 - f0);
        float d1 = (i + 2 < nodes.size()) ? width * (f[i + 2] - f0) / (nodes[i + 2] - x0)
                                          : (f1 - f0);

        // Keep a running sum and build a cumulative distribution function
        sum += width * ((f0 + f1) / 2 + (d0 - d1) / 12);
        cdf[i + 1] = sum;
    }
    return sum;
}

// BSSRDF Function Definitions
float BeamDiffusionMS(float sigma_s, float sigma_a, float g, float eta, float r) {
    const int nSamples = 100;
    float Ed = 0;
    // Precompute information for dipole integrand
    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo $\rhop$
    float sigmap_s = sigma_s * (1 - g);
    float sigmap_t = sigma_a + sigmap_s;
    float rhop = sigmap_s / sigmap_t;

    // Compute non-classical diffusion coefficient $D_\roman{G}$ using Equation
    // $(\ref{eq:diffusion-coefficient-grosjean})$
    float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);

    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    float sigma_tr = safe_sqrt(sigma_a / D_g);

    // Determine linear extrapolation distance $\depthextrapolation$ using Equation
    // $(\ref{eq:dipole-boundary-condition})$
    float fm1 = fresnel_moment1(eta), fm2 = fresnel_moment2(eta);
    float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);

    // Determine exitance scale factors using Equations $(\ref{eq:kp-exitance-phi})$ and
    // $(\ref{eq:kp-exitance-e})$
    float cPhi = 0.25f * (1 - 2 * fm1), cE = 0.5f * (1 - 3 * fm2);

    for (int i = 0; i < nSamples; ++i) {
        // Sample real point source depth $\depthreal$
        float zr = SampleExponential((i + 0.5f) / nSamples, sigmap_t);

        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to _Ed_
        float zv = -zr + 2 * ze;
        float dr = sqrt(sqr(r) + sqr(zr)), dv = sqrt(sqr(r) + sqr(zv));
        // Compute dipole fluence rate $\dipole(r)$ using Equation
        // $(\ref{eq:diffusion-dipole})$
        float phiD =
                inv4Pi / D_g * (fast_exp(-sigma_tr * dr) / dr - fast_exp(-sigma_tr * dv) / dv);

        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using Equation
        // $(\ref{eq:diffusion-dipole-vector-irradiance-normal})$
        float EDn =
                inv4Pi * (zr * (1 + sigma_tr * dr) * fast_exp(-sigma_tr * dr) / (Pow<3>(dr)) -
                          zv * (1 + sigma_tr * dv) * fast_exp(-sigma_tr * dv) / (Pow<3>(dv)));

        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        float E = phiD * cPhi + EDn * cE;
        float kappa = 1 - fast_exp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E;
    }
    return Ed / nSamples;
}

float BeamDiffusionSS(float sigma_s, float sigma_a, float g, float eta, float r) {
    // Compute material parameters and minimum $t$ below the critical angle
    float sigma_t = sigma_a + sigma_s, rho = sigma_s / sigma_t;
    float tCrit = r * safe_sqrt(sqr(eta) - 1);

    float Ess = 0;
    const int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        // Evaluate single-scattering integrand and add to _Ess_
        float ti = tCrit + SampleExponential((i + 0.5f) / nSamples, sigma_t);
        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        float d = sqrt(sqr(r) + sqr(ti));
        float cosTheta_o = ti / d;

        // Add contribution of single scattering at depth $t$
        Ess += rho * fast_exp(-sigma_t * (d + tCrit)) / sqr(d) *
               HenyeyGreenstein(cosTheta_o, g) * (1 - fresnel_dielectric(-cosTheta_o, eta)) *
               std::abs(cosTheta_o);
    }
    return Ess / nSamples;
}

void ComputeBeamDiffusionBSSRDF(float g, float eta, BSSRDFTable *t) {
    // Choose radius values of the diffusion profile discretization
    t->radius_samples[0] = 0;
    t->radius_samples[1] = 2.5e-3f;
    for (int i = 2; i < t->radius_samples.size(); ++i)
        t->radius_samples[i] = t->radius_samples[i - 1] * 1.2f;

    // Choose albedo values of the diffusion profile discretization
    for (int i = 0; i < t->rho_samples.size(); ++i)
        t->rho_samples[i] =
                (1 - fast_exp(-8 * i / (float) (t->rho_samples.size() - 1))) / (1 - fast_exp(-8));

        {
            for (int i = 0;  i < t->rho_samples.size(); ++i) {
                // Compute the diffusion profile for the _i_th albedo sample
                // Compute scattering profile for chosen albedo $\rho$
                size_t nSamples = t->radius_samples.size();
                for (int j = 0; j < nSamples; ++j) {
                    float rho = t->rho_samples[i], r = t->radius_samples[j];
                    t->profile[i * nSamples + j] = 2 * Pi * r *
                                                   (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
                                                    BeamDiffusionMS(rho, 1 - rho, g, eta, r));
                }

                // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance sampling
                t->rhoEff[i] = IntegrateCatmullRom(
                        BufferView<const float>(&t->radius_samples[0], nSamples),
                        BufferView<const float>(&t->profile[i * nSamples], nSamples),
                        BufferView<float>(&t->profileCDF[i * nSamples], nSamples));
            }
        }
}

BSSRDFTable::BSSRDFTable(int rho_sample_count, int radius_sample_count) {

    const int table_size = rho_sample_count * radius_sample_count;

    this->rho_sample_count = rho_sample_count;
    this->radius_sample_count = radius_sample_count;
    this->rho_samples = BufferView<float>(get_arena().allocate<float>(rho_sample_count), rho_sample_count);
    this->radius_samples = BufferView<float>(get_arena().allocate<float>(radius_sample_count), radius_sample_count);
    this->profile = BufferView<float>(get_arena().allocate<float>(table_size), table_size);
    this->rhoEff = BufferView<float>(get_arena().allocate<float>(rho_sample_count), rho_sample_count);
    this->profileCDF = BufferView<float>(get_arena().allocate<float>(table_size), table_size);
}

BSDFSample DielectricBxDF::sample_f(float3 wo, float uc, float2 u, BxDFFlags sampleFlags, TransportMode mode) const {

    if(_eta == 1 || _mfDistrib.effectively_smooth()) {
        // Sample perfectly specular dielectric BSDF
        float pr = fresnel_dielectric(Frame::cos_theta(wo), _eta), pt = 1.0 - pr;

        if(!(sampleFlags & BxDFFlags::Reflection))
            pr = 0;
        if(!(sampleFlags & BxDFFlags::Transmission))
            pt = 0;

        if(pr = 0 && pt == 0)
            return {};
        
        if(uc < pr / (pr + pt)) {
            // Sample perfect specular dielectric BRDF
            float3 wi{ -wo.x, -wo.y, wo.z };
            Spectrum fr = pr / Frame::cos_theta(wi);
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecRefl);
        } else {
            // Sample perfect specular dielectric BTDF
            // Compute ray direction for specular transmission
            float3 wi;

            bool valid = refract(wo, float3(0, 0, 1), _eta, &wi);
            if(!valid)
                return {};

            Spectrum ft = pt / Frame::cos_theta(wi);
            ft *= cal_factor(mode, _eta);

            return BSDFSample{ft, wi, pt / (pr + pt), BxDFFlags::SpecTrans, _eta};
        }
    } else {
        // Sample rough dielectric BSDF
        float3 wh = _mfDistrib.sample_wh(wo, u);
        float pr = fresnel_dielectric(dot(wh, wo), _eta);
        float pt = 1 - pr;

        if(!(sampleFlags & BxDFFlags::Reflection))
            pr = 0;
        if(!(sampleFlags & BxDFFlags::Transmission))
            pt = 0;

        if(pr == 0 && pt == 0)
            return {};

        float pdf;
        if(uc < pr / (pr + pt)) {
            // Sample reflection at rough dielectric interface
            float3 wi = reflect(wo, wh);
            if(!same_hemisphere(wo, wi))
                return {};

            pdf = _mfDistrib.PDF_wi_reflection(wo, wh) * pr / (pr + pt);

            DCHECK(!is_nan(pdf));
            Spectrum f = _mfDistrib.BRDF(wo, wh, wi, pr, Frame::cos_theta(wi), Frame::cos_theta(wo));
            return BSDFSample{f, wi, pdf, BxDFFlags::GlossyRefl};
        } else {
            // Sample transmission at rough dielectric interface
            float3 wi;
            bool tir = !refract(wo, wh, _eta, &wi);
            if(same_hemisphere(wo, wi) || wi.z == 0 || tir)
                return {};

            float pdf = _mfDistrib.PDF_wi_transmission(wo, wh, wi, _eta);

            DCHECK(!is_nan(pdf));

            Spectrum ft = _mfDistrib.BTDF(wo, wh, wi, pt, Frame::cos_theta(wi), Frame::cos_theta(wo), _eta, mode);

            return BSDFSample{ft, wi, pdf, BxDFFlags::GlossyTrans, _eta};
        }
    }
}

Spectrum DielectricBxDF::eval(float3 wo, float3 wi, TransportMode mode) const {

    if(_eta == 1 || _mfDistrib.effectively_smooth())
        return {};

    float cos_theta_o = Frame::cos_theta(wo), cos_theta_i = Frame::cos_theta(wi);
    bool reflect = cos_theta_o * cos_theta_i > 0;
    float etap = 1;
    if(!reflect)
        etap = cos_theta_o > 0 ? _eta : rcp(_eta);

    float3 wh = wi * etap + wo;
    if(cos_theta_i == 0 || cos_theta_o == 0 || length_squared(wh) == 0)
        return {};

    wh = normalize(face_forward(wh, float3(0, 0, 1)));

    // Discard backface microfacets
    if(dot(wh, wi) * cos_theta_i < 0 || dot(wh, wo) * cos_theta_o < 0)
        return {};

    float Fr = fresnel_dielectric(dot(wo, wh), _eta);
    if(reflect) {
        // Reflection pdf
        return _mfDistrib.BRDF(wo, wh, wi, Fr, cos_theta_i, cos_theta_o);
    } else {
        // Refraction pdf
        return _mfDistrib.BTDF(wo, wh, wi, 1.0 - Fr, cos_theta_i, cos_theta_o, etap, mode);
    }
}

float DielectricBxDF::PDF(float3 wo, float3 wi, BxDFFlags sampleFlags, TransportMode mode) const {

    if (_eta == 1 || _mfDistrib.effectively_smooth())
        return {};

    float cos_theta_o = Frame::cos_theta(wo), cos_theta_i = Frame::cos_theta(wi);
    bool reflect = cos_theta_o * cos_theta_i > 0;
    float etap = 1;
    if (!reflect)
        etap = cos_theta_o > 0 ? _eta : rcp(_eta);

    float3 wh = wi * etap + wo;
    if (cos_theta_i == 0 || cos_theta_o == 0 || length_squared(wh) == 0)
        return {};

    wh = normalize(face_forward(wh, float3(0, 0, 1)));

    // Discard backface microfacets
    if (dot(wh, wi) * cos_theta_i < 0 || dot(wh, wo) * cos_theta_o < 0)
        return {};

    float Fr = fresnel_dielectric(dot(wo, wh), _eta);
    float Ft = 1.0 - Fr;

    if(!(sampleFlags & BxDFFlags::Reflection))
        Fr = 0;
    if(!(sampleFlags & BxDFFlags::Transmission))
        Ft = 0;

    if (Fr == 0 && Ft == 0)
        return {};

    float pdf;
    if(reflect)
        pdf = _mfDistrib.PDF_wi_reflection(wo, wh) * Fr / (Fr + Ft);
    else
        pdf = _mfDistrib.PDF_wi_transmission(wo, wh, wi, _eta) * Ft / (Fr + Ft);

    return pdf;
}

}// namespace render
}// namespace luminous