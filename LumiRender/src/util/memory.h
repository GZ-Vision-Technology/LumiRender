//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/header.h"
#include "core/concepts.h"

#define ARENA_ALLOC(arena, Type) new ((arena).alloc(sizeof(Type))) Type

namespace luminous {
    inline namespace utility {
//        void *alloc_aligned(size_t size);
//
//        template<typename T>
//        T *alloc_aligned(size_t count) {
//            return (T *) alloc_aligned(count * sizeof(T));
//        }


//        void free_aligned(void *);


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

        /*
         * 内存管理是一个很复杂的问题，但在离线渲染器中，内存管理的情况相对简单，大部分的内存申请
         * 主要集中在解析场景的阶段，这些内存在渲染结束之前一直被使用
         * 为何要使用内存池？
         *
         * 1.频繁的new跟delete性能消耗很高，new运算符执行的时候相当于会使当前线程block，
         * 直到操作系统返回可用内存时，线程才继续执行，如果使用了内存池，预先申请一大块连续内存
         * 之后每次申请内存时不是向操作系统申请，而是直接将指向当前地址的指针自增就可以了，分配效率高
         *
         * 2.用内存池可以自定义内存对齐的方式，从而写出对缓存友好的程序
         *     好的对齐方式可以提高缓存命中率，比如CPU从内存中将数据加载到缓存中时
         *     会从特定的地址(必须是cache line长度的整数倍)中加载特定的长度(必须是cache line的长度)
         *     通常cache line的长度为64字节，如果一个int所占的位置横跨了两个cache line，cache miss最多为两次
         *     如果该数据的完全在一个cache line以内，那么cache miss的次数最多为一次
         *
         */
        class Arena : public Noncopyable {
        public:
            static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

        private:
            std::vector<std::byte *> _blocks;
            uint64_t _ptr{0ul};
            size_t _total{0ul};

        public:
            Arena() noexcept = default;

            Arena(Arena &&) noexcept = default;

            Arena &operator=(Arena &&) noexcept = default;

            ~Arena() noexcept {
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
                return luisa::construct_at(allocate<T>(1u), std::forward<Args>(args)...);
            }
        };

    }
}

