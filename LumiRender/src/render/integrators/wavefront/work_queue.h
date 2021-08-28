//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "work_items.h"

namespace luminous {
    inline namespace render {
        template<typename WorkItem>
        class WorkQueue : public SOA<WorkItem> {
        private:
            std::atomic<int> _size{0};
        public:
            WorkQueue() = default;

            WorkQueue(int n, std::shared_ptr<Device> &device)
                    : SOA<WorkItem>(n, device) {}

            WorkQueue &operator=(const WorkQueue &other) {
                SOA<WorkItem>::operator=(other);
                _size.store(other._size.load());
            }

            XPU int size() const {
                return _size.load(std::memory_order_relaxed);
            }

            XPU void reset() {
                _size.store(0, std::memory_order_relaxed);
            }

            XPU int allocate_entry() {
                return _size.fetch_add(1, std::memory_order_relaxed);
            }

            XPU int push(WorkItem w) {
                int index = allocate_entry();
                (*this)[index] = w;
                return index;
            }
        };
    }
}