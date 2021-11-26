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

        float LanczosSincFilter::evaluate(const float2 &p) const {
//            auto r1 = [&](float v, float r) {
//                if (abs(v) < 0.81649 * r) {
//                    return true;
//                }
//                return false;
//            };
//
//            if (r1(p.x, _radius.x) && r1(p.y, _radius.y)) {
//                return 1;
//            }
//            else if (abs(p.x) > _radius.x && abs(p.y) > _radius.y) {
//                return 0;
//            } else {
//                return -1;
//            }
            return windowed_sinc(p.x, _radius.x, _tau) * windowed_sinc(p.y, _radius.y, _tau)*4;
        }
    }
}