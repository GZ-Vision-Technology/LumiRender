//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/concepts.h"
#include "scene_graph.h"

namespace luminous {

    class Parser : public Noncopyable {
    private:
        DataWrap _data;
        Context *_context;
    public:
        explicit Parser(Context *context) : _context(context) {}

        UP<SceneGraph> load_from_json(const std::filesystem::path &fn);
    };
}