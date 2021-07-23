//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/concepts.h"
#include "scene_graph.h"
#include "ext/nlohmann/json.hpp"
using DataWrap = nlohmann::json ;

namespace luminous {

    inline namespace render {
        class Parser : public Noncopyable {
        private:
            DataWrap _data;
            Context *_context;
        public:
            explicit Parser(Context *context) : _context(context) {}

            void load(const std::filesystem::path &fn);

            SP<SceneGraph> parse() const;
        };
    }
}