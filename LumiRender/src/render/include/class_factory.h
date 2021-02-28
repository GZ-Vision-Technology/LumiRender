//
// Created by Zero on 2021/2/25.
//


#pragma once

#include "graphics/header.h"
#include "scene_graph.h"
#include <map>

namespace luminous {
    inline namespace render {
        using Creator = IObject *(const Config &config);

        class ClassFactory {
        private:
            std::map<std::string, Creator *> _creator_map;
        public:
            static ClassFactory *instance();

            void register_creator(const std::string &name, Creator *func);

            Creator *get_creator(const std::string &name);

        private:

            ClassFactory(const ClassFactory &) {}

            ClassFactory &operator=(const ClassFactory &) { return *this; }

            ClassFactory() = default;

            static ClassFactory *_instance;
        };

        class RegisterAction {

        public:

            RegisterAction(const std::string &name, Creator creator) {
                ClassFactory::instance()->register_creator(name, creator);
            }
        };

#define REGISTER(name, creator)                            \
RegisterAction g_Register##name(#name,(Creator*)creator);

#define GET_CREATOR(name) ClassFactory::instance()->get_creator(name);
    }
}