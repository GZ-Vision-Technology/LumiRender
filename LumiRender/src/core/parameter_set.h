//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "header.h"

namespace luminous {
    inline namespace utility {

        class ParameterSet {
        private:
            DataWrap _data;
            std::string _key;
        public:
            ParameterSet(const DataWrap &data, const std::string &key = "")
                    : _data(data),
                      _key(key) {}

        };

    } //luminous::utility
} // luminous