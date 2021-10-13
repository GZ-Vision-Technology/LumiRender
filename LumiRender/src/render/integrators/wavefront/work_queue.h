//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "soa.h"
#include "core/backend/device.h"
#include "core/atomic.h"

namespace luminous {

    inline namespace render {
        template<typename WorkItem>
        class WorkQueue : public SOA<WorkItem> {
        private:
            AtomicInt _size;
        public:
            WorkQueue() = default;

            WorkQueue(int n, Device *device)
                    : SOA<WorkItem>(n, device) {}

            WorkQueue(const WorkQueue &other)
                    : SOA<WorkItem>(other) {
                _size.store(other._size.value());
            }

            WorkQueue &operator=(const WorkQueue &other) {
                SOA<WorkItem>::operator=(other);
                _size.store(other._size.value());
                return *this;
            }

            LM_XPU int size() const {
                return _size.value();
            }

            LM_XPU void reset() {
                _size.store(0);
            }

            LM_XPU int allocate_entry() {
                return _size.fetch_add(1);
            }

            LM_XPU int push(WorkItem w) {
                int index = allocate_entry();
                (*this)[index] = w;
                return index;
            }
        };
    }
}