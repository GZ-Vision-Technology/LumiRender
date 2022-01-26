#ifndef __BSSRDF_H__
#define __BSSRDF_H__

#include "bsdf_data.h"
#include "core/backend/buffer_view.h"
#include "base.h"
#include "bsdf_data.h"
#include "specular_scatter.h"
#include "diffuse_scatter.h"
#include "bsdfs.h"
#include <utility>
#include "render/include/interaction.h"
#include <functional>

namespace luminous {
inline namespace render {

template<class T, class C>
constexpr T evaluate_polynomial(T t, C c) {
    return c;
}


template<class T, class C, class... Remaing>
constexpr T evaluate_polynomial(T t, C c, Remaing... args) {
    return fma(t, evaluate_polynomial(t, args...), c);
}

template<class Func>
float newton_bisection(float x0, float x1, Func f, float xEps = 1.0e-6f, float fEps = 1.0e-6) {
    // Check function endpoints for roots
    DCHECK_LT(x0, x1);
    float fx0 = f(x0).first, fx1 = f(x1).first;
    if (std::abs(fx0) < fEps)
        return x0;
    if (std::abs(fx1) < fEps)
        return x1;
    bool startIsNegative = fx0 < 0;

    // Set initial midpoint using linear approximation of _f_
    float xMid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    while (true) {
        // Fall back to bisection if _xMid_ is out of bounds
        if (!(x0 < xMid && xMid < x1))
            xMid = (x0 + x1) / 2;

        // Evaluate function and narrow bracket range _[x0, x1]_
        std::pair<float, float> fxMid = f(xMid);
        DCHECK(!is_nan(fxMid.first));
        if (startIsNegative == (fxMid.first < 0))
            x0 = xMid;
        else
            x1 = xMid;

        // Stop the iteration if converged
        if ((x1 - x0) < xEps || std::abs(fxMid.first) < fEps)
            return xMid;

        // Perform a Newton step
        xMid -= fxMid.first / fxMid.second;
    }
}

/// BSSRDF utility functions

// Spline Interpolation Function Definitions
inline
bool catmull_rom_weights(BufferView<const float> nodes, float x, int *offset,
                         BufferView<float> weights) {
    DCHECK_GE(weights.size(), 4);
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes.front() && x <= nodes.back()))
        return false;

    // Search for the interval _idx_ containing _x_
    int idx = find_interval(nodes.size(), [&x, nodes](int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    float x0 = nodes[idx], x1 = nodes[idx + 1];

    // Compute the $t$ parameter and powers
    float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;

    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;

    // Compute first node weight $w_0$
    if (idx > 0) {
        float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if (idx + 2 < nodes.size()) {
        float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }

    return true;
}

inline
float sample_catmull_rom_2d(BufferView<const float> nodes1, BufferView<const float> nodes2,
                            BufferView<const float> values, BufferView<const float> cdf,
                            float alpha, float u, float *fval = nullptr, float *pdf = nullptr) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    float weights[4];
    if (!catmull_rom_weights(nodes1, alpha, &offset, {weights, 4}))
        return 0;

    // Define a lambda function to interpolate table entries
    auto interpolate = [&](const BufferView<const float> &array, int idx) {
        float v = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                v += array[(offset + i) * nodes2.size() + idx] * weights[i];
        return v;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    float maximum = interpolate(cdf, nodes2.size() - 1);
    u *= maximum;
    int idx =
            find_interval(nodes2.size(), [&](int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    float width = x1 - x0;
    float d0, d1;

    // Rescale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < nodes2.size())
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert definite integral over spline segment
    float Fhat, fhat;
    auto eval = [&](float t) -> std::pair<float,float> {
        Fhat = evaluate_polynomial(t, 0, f0, 0.5f * d0,
                                   (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                                   0.25f * (d0 + d1) + 0.5f * (f0 - f1));
        fhat = evaluate_polynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                   d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    float t = newton_bisection(0, 1, eval);

    if (fval)
        *fval = fhat;
    if (pdf)
        *pdf = fhat / maximum;
    return x0 + width * t;
}

extern LM_XPU float fresnel_moment1(float eta);
extern LM_XPU float fresnel_moment2(float eta);

class DielectricBxDF {
public:
    DielectricBxDF() = default;

    constexpr DielectricBxDF(float eta, Microfacet mfDistrib)
        : _eta(eta), _mfDistrib(mfDistrib) {
    }

    constexpr BxDFFlags flags() const {
        BxDFFlags flags = (_eta == 1) ? BxDFFlags::Transmission : (BxDFFlags::Reflection | BxDFFlags::Transmission);
        return flags || _mfDistrib.effectively_smooth() ? BxDFFlags::Specular : BxDFFlags::Glossy;
    }

    BSDFSample sample_f(float3 wo, float uc, float2 u, BxDFFlags sampleFlags, TransportMode mode) const;

    Spectrum eval(float3 wo, float3 wi, TransportMode mode) const;

    float PDF(float3 wo, float3 wi, BxDFFlags sampleFlags, TransportMode mode) const;

private:
    float _eta;
    Microfacet _mfDistrib;
};

struct BSSRDFSample {
    Spectrum Sp, pdf;
    BSDF Sw;
    float3 wo;
};

struct SubsurfaceInteraction {
    SubsurfaceInteraction(const SurfaceInteraction &si)
    : pi(si.pos), frame(si.s_uvn) {}

    const float3 &p() const {
        return pi;
    }

    float3 pi;
    Frame frame;
};

// BSSRDF Function Declarations
float BeamDiffusionSS(float sigma_s, float sigma_a, float g, float eta, float r);
float BeamDiffusionMS(float sigma_s, float sigma_a, float g, float eta, float r);

struct BSSRDFTable {
    int rho_sample_count;
    int radius_sample_count;
    BufferView<float> rho_samples;
    BufferView<float> radius_samples;
    BufferView<float> profile;
    BufferView<float> rhoEff;
    BufferView<float> profileCDF;

    BSSRDFTable(int rho_sample_count, int radius_sample_count);

    LM_XPU float eval_profile(int rho_index, int rad_index) const {
        return profile[rho_index * radius_sample_count + rad_index];
    }
};

void ComputeBeamDiffusionBSSRDF(float g, float eta, BSSRDFTable *t);

struct BSSRDFProbeSegment {
    float3 po, pi;
};

class TabulatedBSSRDF {
public:
    TabulatedBSSRDF() = default;

    TabulatedBSSRDF(float3 po, float3 ns, float3 wo, float eta,
        Spectrum sigma_a, Spectrum sigma_s, const BSSRDFTable *table)
     : _po(po), _ns(ns), _wo(wo), _eta(eta), _table(table) {
         for (int ch = 0; ch < Spectrum::nSamples; ++ch) {
             _sigma_t[ch] = sigma_a[ch] + sigma_s[ch];
             _rho[ch] = _sigma_t[ch] != 0 ? sigma_s[ch] / _sigma_t[ch] : 0;
         }
    }

    Spectrum Sp(float3 pi) const {
        return Sr(distance(pi, _po));
    }

    Spectrum Sr(float r) const {
        Spectrum Sr{.0f};

        for (int ch = 0; ch < Spectrum::nSamples; ++ch) {
            float r_optical = _sigma_t[ch] * r;

            int rho_offset, radius_offset;
            float rho_weights[4], radius_weights[4];

            if(!catmull_rom_weights(_table->rho_samples, _rho[ch], &rho_offset, BufferView<float>{rho_weights})
                || !catmull_rom_weights(_table->radius_samples, r_optical, &radius_offset, BufferView<float>{radius_weights}))
                continue;

            // Set BSSRDF value _Sr[i]_ using tensor spline interpolation
            float sr = 0;
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k) {
                    // Accumulate contribution of $(j,k)$ table sample
                    if (float weight = rho_weights[j] * radius_weights[k]; weight != 0)
                        sr +=
                                weight * _table->eval_profile(rho_offset + j, radius_offset + k);
                }
            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if (r_optical != 0)
                sr /= 2 * Pi * r_optical;

            Sr[ch] = sr;
        }

        // Transform BSSRDF value into rendering space units
        Sr *= sqr(_sigma_t);
        return Spectrum{vclamp(Sr, Spectrum{.0f}, Spectrum{1.0f})};
    }

    float sample_Sr(float u) const {

        if (_sigma_t[0] == 0)
            return -1;
        return sample_catmull_rom_2d(_table->rho_samples, _table->radius_samples, _table->profile,
                                  _table->profileCDF, _rho[0], u) /
               _sigma_t[0];
    }

    Spectrum pdf_Sr(float r) const {
        Spectrum pdf(0.f);
        for (int i = 0; i < Spectrum::nSamples; ++i) {
            // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
            float r_optical = r * _sigma_t[i];

            // Compute spline weights to interpolate BSSRDF at _i_th wavelength
            int rho_offset, radius_offset;
            float rho_weights[4], radius_weights[4];
            if (!catmull_rom_weights(_table->rho_samples, _rho[i], &rho_offset, BufferView<float>(rho_weights)) ||
                !catmull_rom_weights(_table->radius_samples, r_optical, &radius_offset,
                                   BufferView<float>(radius_weights)))
                continue;

            // Set BSSRDF profile probability density for wavelength
            float sr = 0, rho_eff = 0;
            for (int j = 0; j < 4; ++j)
                if (rho_weights[j] != 0) {
                    // Update _rhoEff_ and _sr_ for wavelength
                    rho_eff += _table->rhoEff[rho_offset + j] * rho_weights[j];
                    for (int k = 0; k < 4; ++k)
                        if (radius_weights[k] != 0)
                            sr += _table->eval_profile(rho_offset + j, radius_offset + k) *
                                  rho_weights[j] * radius_weights[k];
                }
            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if (r_optical != 0)
                sr /= 2 * Pi * r_optical;

            pdf[i] = sr * sqr(_sigma_t[i]) / rho_eff;
        }
        return vclamp(pdf, Spectrum{.0f}, Spectrum{1.0f});
    }

    optional<BSSRDFProbeSegment> sample_Sp(const Frame &fpo, float u1, float2 u2) const {

        // Choose projection axis for BSSRDF sampling
        float3 vx, vy, vz;
        if (u1 < 0.25f) {
            vx = fpo.z;
            vy = fpo.x;
            vz = fpo.y;
        } else if (u1 < 0.5f) {
            vx = fpo.y;
            vy = fpo.z;
            vz = fpo.x;
        } else {
            vx = fpo.x;
            vy = fpo.y;
            vz = fpo.z;
        }

        // Sample BSSRDF profile in polar coordinates
        float r = sample_Sr(u2.x);
        if (r <= 0)
            return {};
        float phi = 2 * Pi * u2.y;

        // Compute BSSRDF profile bounds and intersection height
        float r_max = sample_Sr(0.999f);
        if (r_max < 0 || r >= r_max)
            return {};
        float l = 2 * std::sqrt(sqr(r_max) - sqr(r));

        // Return BSSRDF sampling ray segment
        float3 pStart =
                _po + r * (vx * std::cos(phi) + vy * std::sin(phi)) - l * vz / 2.f;
        float3 pTarget = pStart + l * vz;
        return BSSRDFProbeSegment{pStart, pTarget};
    }

    Spectrum pdf_Sp(const Frame &fpo, float3 pi, float3 ni) const {
        // Express $\pti-\pto$ and $\N{}_\roman{i}$ with respect to local coordinates at
        // $\pto$
        float3 d = pi - _po;
        float3 dLocal = fpo.to_local(d);
        float3 nLocal = fpo.to_local(ni);

        // Compute BSSRDF profile radius under projection along each axis
        float rProj[3] = {std::sqrt(sqr(dLocal.y) + sqr(dLocal.z)),
                          std::sqrt(sqr(dLocal.z) + sqr(dLocal.x)),
                          std::sqrt(sqr(dLocal.x) + sqr(dLocal.y))};

        // Return combined probability from all BSSRDF sampling strategies
        Spectrum pdf(0.f);
        float axisProb[3] = {.25f, .25f, .5f};
        for (int axis = 0; axis < 3; ++axis)
            pdf += pdf_Sr(rProj[axis]) * std::abs(nLocal[axis]) * axisProb[axis];
        return pdf;
    }

    BSSRDFSample probe_intersection_to_sample(const SurfaceInteraction &si) {

        SubsurfaceInteraction ssi{si};
        float3 wo = si.s_uvn.z;

        BSDFHelper data{};
        data.set_eta(_eta);

        // BSDF bsdf{ NormalizedFresnelBSDF{ data, NormalizedFresnelBxDF{_eta} } };
        BSDF bsdf;

        return BSSRDFSample{Sp(ssi.p()), {}, bsdf, wo};
    }

protected:
    float3 _po;
    float3 _ns;
    float3 _wo;
    float _eta;
    Spectrum _sigma_t;// extinction coeffient: $\sigma_t = \sigma_a + \sigma_s $
    Spectrum _rho;    // extinction albedo: $\rho = \sigma_s / \sigma_t$
    const BSSRDFTable *_table;
};

class BSSRDF: lstd::Variant<TabulatedBSSRDF> {
    using Variant::Variant;
};

}// namespace render
}// namespace luminous


#endif// __BSSRDF_H__