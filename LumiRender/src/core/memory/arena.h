//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/concepts.h"
#include "base_libs/string_util.h"
#include "core/logging.h"


namespace luminous {
    inline namespace core {
        static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

        template<typename T = void>
        T *aligned_alloc(size_t alignment, size_t size) noexcept {
            return reinterpret_cast<T *>(_aligned_malloc(size, alignment));
        }

        template<typename T = void>
        void aligned_free(T *p) noexcept {
            _aligned_free(p);
        }

        template<typename T, typename... Args>
        constexpr T *construct_at(T *p, Args &&...args) {
            return ::new(const_cast<void *>(static_cast<const volatile void *>(p)))
                    T(std::forward<Args>(args)...);
        }

        struct MemoryBlock {
        private:
            std::byte *address{};
            uint64_t next_allocate_ptr{};
            size_t capacity{};
        public:
            template<typename T, size_t alignment = alignof(T)>
            LM_NODISCARD T *alloc(size_t byte_size) {
                static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
                static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
                auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment *
                                  alloc_alignment;
                capacity = alloc_size;
                address = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size));
                next_allocate_ptr = reinterpret_cast<uint64_t>(address + byte_size);
                return reinterpret_cast<T *>(address);
            }

            template<typename T, size_t alignment = alignof(T)>
            LM_NODISCARD T *use(size_t byte_size) {
                auto aligned_p = reinterpret_cast<std::byte *>((next_allocate_ptr + alignment - 1u) /
                                                               alignment * alignment);
                next_allocate_ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
                return reinterpret_cast<T *>(aligned_p);
            }

            LM_NODISCARD std::byte *aligned_ptr(size_t alignment) const {
                return reinterpret_cast<std::byte *>((next_allocate_ptr + alignment - 1u) /
                                                     alignment * alignment);
            }

            LM_NODISCARD std::byte *end_ptr() const { return address + capacity; }
        };

        class MemoryArena : public Noncopyable {
        public:

            using BlockList = std::list<MemoryBlock>;
            using BlockIterator = BlockList::iterator;
        private:
            std::list<std::byte *> _blocks;
            uint64_t _ptr{0ul};
            size_t _total{0ul};

            BlockList _memory_blocks;

        public:
            MemoryArena() noexcept = default;

            MemoryArena(MemoryArena &&) noexcept = default;

            MemoryArena &operator=(MemoryArena &&) noexcept = default;

            ~MemoryArena() noexcept {
                for (auto p : _blocks) { aligned_free(p); }
            }

            LM_NODISCARD auto total_size() const noexcept { return _total; }

            BlockIterator find_suitable_blocks(size_t byte_size, size_t alignment) {
                auto best_iter = _memory_blocks.end();
                auto min_remain = block_size;
                for (auto iter = _memory_blocks.begin(); iter != _memory_blocks.end(); ++iter) {
                    auto aligned_p = iter->aligned_ptr(alignment);
                    std::byte *next_allocate_ptr = aligned_p + byte_size;
                    int64_t remain = iter->end_ptr() - next_allocate_ptr;
                    if (remain > 0 && remain < min_remain) {
                        best_iter = iter;
                        min_remain = remain;
                    }
                }
                return best_iter;
            }

            template<typename T = std::byte, size_t alignment = alignof(T)>
            LM_NODISCARD T *allocate(size_t n = 1u) {
                static constexpr auto size = sizeof(T);
                auto byte_size = n * size;
                auto iter = find_suitable_blocks(byte_size, alignment);
                if (iter == _memory_blocks.end()) {
                    _memory_blocks.emplace_back();
                    MemoryBlock &back = _memory_blocks.back();
                    return back.template alloc<T>(byte_size);
                } else {
                    return iter->template use<T>(byte_size);
                }
            }

            template<typename T = std::byte, size_t alignment = alignof(T)>
            LM_NODISCARD auto allocate_old(size_t n = 1u) {

                static constexpr auto size = sizeof(T);

                auto byte_size = n * size;
                auto aligned_p = reinterpret_cast<std::byte *>((_ptr + alignment - 1u) / alignment * alignment);
                if (_blocks.empty() || aligned_p + byte_size > _blocks.back() + block_size) {
                    static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
                    static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
                    auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment *
                                      alloc_alignment;
                    aligned_p = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size));
                    if (aligned_p == nullptr) {
                        LUMINOUS_ERROR(
                                string_printf("Failed to allocate memory: size = %d, alignment = %d, count = %d",
                                              size, alignment, n))
                    }
                    _blocks.emplace_back(aligned_p);
                    _total += alloc_size;
                }
                _ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
                return reinterpret_cast<T *>(aligned_p);
            }

            template<typename T, typename... Args>
            LM_NODISCARD T *create(Args &&...args) {
                return construct_at(allocate<T>(1u), std::forward<Args>(args)...);
            }
        };

    }
}

