//
// Created by Zero on 17/09/2021.
//

#include "ptr_mapper.h"

namespace luminous {
    inline namespace refl {

        PtrMapper *PtrMapper::_instance = nullptr;

        PtrMapper *PtrMapper::instance() {
            if (_instance == nullptr) {
                _instance = new PtrMapper();
            }
            return _instance;
        }

        void PtrMapper::add_pair(PtrInterval host, PtrInterval device) {
            ptr_t key = host.begin;
            _map[key] = std::make_pair(host, device);
        }

        uint64_t PtrMapper::get_device_ptr(ptr_t host_ptr) const {
            for (auto elm : _map) {
                auto val = elm.second;
                PtrInterval host_interval = val.first;
                PtrInterval device_interval = val.second;
                if (host_interval.contains(host_ptr)) {
                    uint64_t offset = host_ptr - device_interval.begin;
                    return device_interval.begin + offset;
                }
            }
            DCHECK(0)
            return 0;
        }

        void PtrMapper::add_reverse_mapping(ptr_t device_ptr, ptr_t host_ptr) {
            _device_to_host[device_ptr] = host_ptr;
        }

        ptr_t PtrMapper::get_host_ptr(ptr_t device_ptr) const {
            return _device_to_host.at(device_ptr);
        }
    }
}