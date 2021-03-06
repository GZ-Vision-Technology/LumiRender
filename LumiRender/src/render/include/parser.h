//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/concepts.h"
#include "scene_graph.h"

namespace luminous {

    inline namespace render {
        class Parser : public Noncopyable {
        private:
            DataWrap _data;
            Context *_context;
        public:
            explicit Parser(Context *context) : _context(context) {}

            void load_from_json(const std::filesystem::path &fn);

            UP<SceneGraph> parse() const;
        };
    }
}