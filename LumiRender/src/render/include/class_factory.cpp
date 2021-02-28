//
// Created by Zero on 2021/2/25.
//

#include "class_factory.h"

namespace luminous {
    inline namespace render {
        ClassFactory *ClassFactory::_instance = nullptr;

        void ClassFactory::register_creator(const std::string &name, Creator *func) {
            _creator_map.insert(make_pair(name, func));
        }

        ClassFactory *ClassFactory::instance() {
            if (_instance == nullptr) {
                _instance = new ClassFactory();
            }
            return _instance;
        }

        Creator *ClassFactory::get_creator(const std::string &name) {
            auto iter = _creator_map.find(name);
            if (iter == _creator_map.end()) {
                LUMINOUS_ERROR(name, "'s creator is not exist!");
            }
            return iter->second;
        }
    }
}