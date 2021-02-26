//
// Created by Zero on 2021/2/25.
//


#pragma once

#include "graphics/header.h"
#include "scene_graph.h"

namespace luminous {
    inline namespace render {

        class HandlerBase {

        };

        class ClassFactory {
            using Creator = HandlerBase (const Config &config);
        };
    }
}