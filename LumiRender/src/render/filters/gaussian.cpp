//
// Created by Zero on 24/11/2021.
//

#include "gaussian.h"
#include "filter.h"

namespace luminous {
    inline namespace render {

        GaussianFilter::GaussianFilter(float2 r, float sigma)
                : FilterBase(r),
                  _exp_x(gaussian(r.x, 0, sigma)),
                  _exp_y(gaussian(r.y, 0, sigma)),
                  _sigma(sigma) {
            _sampler.init(std::make_shared<Filter>(*this).get());
        }
    }
}