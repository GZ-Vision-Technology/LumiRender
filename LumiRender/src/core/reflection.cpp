//
// Created by Zero on 13/09/2021.
//

#include "reflection.h"
#include "logging.h"

namespace luminous {

    ClassFactory *ClassFactory::_instance = nullptr;

    ClassFactory *ClassFactory::instance() {
        if (_instance == nullptr) {
            _instance = new ClassFactory();
        }
        return _instance;
    }

    size_t ClassFactory::size_of(const std::string &class_name) const {
        return _type_data.at(class_name).size;
    }

    size_t ClassFactory::align_of(const std::string &class_name) const {
        return _type_data.at(class_name).align;
    }

    size_t ClassFactory::size_of(const Object *object) const {
        return size_of(type_name(object));
    }

    size_t ClassFactory::align_of(const Object *object) const {
        return align_of(type_name(object));
    }

    void ClassFactory::register_class(const std::string &class_name, const TypeData &type_data) {
        LUMINOUS_INFO("register : ", class_name);
        _type_data.insert(std::make_pair(class_name, type_data));
    }
}