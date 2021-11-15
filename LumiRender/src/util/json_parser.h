//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "parser.h"
#include "ext/nlohmann/json.hpp"
using DataWrap = nlohmann::json ;

namespace luminous {

    inline namespace utility {
        class JsonParser : public Parser {
        private:
            DataWrap _data;
        public:
            explicit JsonParser(Context *context) : Parser(context) {}

            void load(const luminous_fs::path &fn) override;

            LM_NODISCARD SP<SceneGraph> parse() const override;
        };
    }
}