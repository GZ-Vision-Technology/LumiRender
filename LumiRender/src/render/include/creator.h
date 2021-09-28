//
// Created by Zero on 2021/4/10.
//


#pragma once

#ifndef __CUDACC__

#include "base_libs/lstd/lstd.h"
#include "core/logging.h"
namespace luminous {
    inline namespace render {
        namespace detail {
            template<typename Handle, typename Config, uint8_t current_index = 0>
            LM_NODISCARD Handle create(const Config &config) {
                using Class = std::tuple_element_t<current_index, typename Handle::TypeTuple>;
                if (type_name<Class>() == config.type()) {
                    return Handle(Creator<Class>::create(config));
                }
                if constexpr (current_index + 1 == std::tuple_size_v<typename Handle::TypeTuple>) {
                    LUMINOUS_ERROR(string_printf("unknown %s type %s", Handle::base_name(), config.type().c_str()));
                } else {
                    return create<Handle, Config, current_index + 1>(config);
                }
            }
        }
    } // luminous::render
} // luminous

#endif