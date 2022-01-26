//
// Created by Zero on 24/11/2021.
//

#include "gaussian.h"
#include "filter.h"
#include <memory>

namespace luminous {
    inline namespace render {

        GaussianFilter::GaussianFilter(float2 r, float sigma)
                : FittedFilter(r),
                  _exp_x(gaussian(r.x, 0, sigma)),
                  _exp_y(gaussian(r.y, 0, sigma)),
                  _sigma(sigma) {
            _sampler.init(std::make_shared<Filter>(*this).get());
        }

        float GaussianFilter::evaluate(const float2 &p) const {
            return (std::max<float>(0, gaussian(p.x, 0, _sigma) - _exp_x) *
                    std::max<float>(0, gaussian(p.y, 0, _sigma) - _exp_y));
        }
    }
}