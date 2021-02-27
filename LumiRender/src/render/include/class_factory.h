//
// Created by Zero on 2021/2/25.
//


#pragma once

#include "graphics/header.h"
#include "scene_graph.h"
#include <map>

namespace luminous {
    inline namespace render {

        class ClassFactory {
        public:
            using Creator = IObject *(const Config &config);
        private:
            std::map<std::string, Creator *> _creator_map;

            void register_creator(const std::string &name, Creator *func);

            static ClassFactory *instance();

            ClassFactory::Creator *get_creator(const std::string &name);

        private:

            ClassFactory(const ClassFactory &) {}

            ClassFactory &operator=(const ClassFactory &) { return *this; }

            ClassFactory() = default;

            static ClassFactory *_instance;
        };
    }
}