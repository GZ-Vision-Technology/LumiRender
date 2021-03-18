//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/header.h"
#include "core/concepts.h"


namespace luminous {
    inline namespace utility {

        void *aligned_alloc(size_t alignment, size_t size) noexcept {
            return _aligned_malloc(size, alignment);
        }

        void aligned_free(void *p) noexcept {
            _aligned_free(p);
        }

        template<typename T, typename... Args>
        constexpr T *construct_at(T *p, Args &&...args) {
            return ::new(const_cast<void *>(static_cast<const volatile void *>(p)))
                    T(std::forward<Args>(args)...);
        }

        class MemoryArena : public Noncopyable {
        public:
            static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

        private:
            std::list<std::byte *> _blocks;
            uint64_t _ptr{0ul};
            size_t _total{0ul};

        public:
            MemoryArena() noexcept = default;

            MemoryArena(MemoryArena &&) noexcept = default;

            MemoryArena &operator=(MemoryArena &&) noexcept = default;

            ~MemoryArena() noexcept {
                for (auto p : _blocks) { aligned_free(p); }
            }

            [[nodiscard]] auto total_size() const noexcept { return _total; }

            template<typename T = std::byte, size_t alignment = alignof(T)>
            [[nodiscard]] auto allocate(size_t n = 1u) {

                static_assert(std::is_trivially_destructible_v<T>);
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
                        LUMINOUS_ERROR(string_printf("Failed to allocate memory: size = %d, alignment = %d, count = %d"),
                                       size, alignment, n)
                    }
                    _blocks.emplace_back(aligned_p);
                    _total += alloc_size;
                }
                _ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
                return reinterpret_cast<T *>(aligned_p);
            }

            template<typename T, typename... Args>
            [[nodiscard]] T *create(Args &&...args) {
                return construct_at(allocate<T>(1u), std::forward<Args>(args)...);
            }
        };

    }
}

