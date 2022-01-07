//
// Created by Zero on 07/01/2022.
//

#include "pyramid_mgr.h"

namespace luminous {
    inline namespace cpu {
        PyramidMgr * PyramidMgr::_instance = nullptr;

        PyramidMgr *PyramidMgr::instance() {
            if (_instance == nullptr) {
                _instance = new PyramidMgr();
            }
            return _instance;
        }
    }
}