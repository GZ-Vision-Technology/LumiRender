//
// Created by Zero on 26/11/2021.
//

#include "mitchell.h"
#include "filter.h"

namespace luminous {
    inline namespace render {

        MitchellFilter::MitchellFilter(float2 r, float b, float c)
                : FittedFilter(r), b(b), c(c) {
            _sampler.init(std::make_shared<Filter>(*this).get());
        }

        float MitchellFilter::evaluate(const float2 &p) const {
            return mitchell_1d(2 * p.x / _radius.x) *
                   mitchell_1d(2 * p.y / _radius.y);
        }
    }
}
