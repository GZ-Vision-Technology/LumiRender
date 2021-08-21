//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class WavefrontAggregate : public Noncopyable {
        public:
            virtual void intersect_closest(/*todo params*/) = 0;

            virtual void intersect_any(/*todo params*/) = 0;

            virtual void intersect_any_tr(/*todo params*/) = 0;
        };
    }
}