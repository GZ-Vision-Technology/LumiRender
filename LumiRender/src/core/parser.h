//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "header.h"
#include "concepts.h"
#include "graphics/string_util.h"
#include "parameter_set.h"

namespace luminous {

    inline DataWrap create_json_from_file(const std::filesystem::path &fn) {
        std::ifstream fst;
        fst.open(fn.c_str());
        std::stringstream buffer;
        buffer << fst.rdbuf();
        std::string str = buffer.str();
        str = jsonc_to_json(str);
//        LUMINOUS_INFO(str);
        if (fn.extension() == "bson") {
            return DataWrap::from_bson(str);
        } else {
            return DataWrap::parse(str);
        }
    }

    class Parser : public Noncopyable {
    private:
        DataWrap _data;
    public:
        void load_from_json(const std::filesystem::path &fn) {
            _data = create_json_from_file(fn);
        }
    };
}