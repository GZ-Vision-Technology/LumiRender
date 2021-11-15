//
// Created by Zero on 15/11/2021.
//


#pragma once

#include "core/concepts.h"
#include "scene_graph.h"

namespace luminous {

    inline namespace utility {
        class Parser : public Noncopyable {
        protected:
            Context *_context;
        public:
            explicit Parser(Context *context) : _context(context) {}

            virtual void load(const luminous_fs::path &fn) = 0;

            LM_NODISCARD virtual SP<SceneGraph> parse() const = 0;
        };
    }
}