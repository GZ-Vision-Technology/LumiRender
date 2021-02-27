//
// Created by Zero on 2021/2/25.
//

#include "class_factory.h"

namespace luminous {
    inline namespace render {
        ClassFactory * ClassFactory::_instance = nullptr;

        void ClassFactory::register_creator(const std::string &name, ClassFactory::Creator * func) {
            _creator_map.insert(make_pair(name, func));
        }

        ClassFactory *ClassFactory::instance() {
            if (_instance == nullptr) {
                _instance = new ClassFactory();
            }
            return _instance;
        }

        ClassFactory::Creator * ClassFactory::get_creator(const std::string &name) {
            return nullptr;
        }
    }
}