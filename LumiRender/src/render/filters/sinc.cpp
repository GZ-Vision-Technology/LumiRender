//
// Created by Zero on 26/11/2021.
//

#include "sinc.h"
#include "filter.h"

namespace luminous {
    inline namespace render {

        LanczosSincFilter::LanczosSincFilter(float2 r, float tau)
                : FittedFilter(r),
                _tau(tau) {
            _sampler.init(std::make_shared<Filter>(*this).get());
        }
    }
}