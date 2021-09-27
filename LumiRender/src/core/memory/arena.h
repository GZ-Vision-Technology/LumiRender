//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/concepts.h"
#include "base_libs/string_util.h"
#include "core/logging.h"
#include "base_libs/math/common.h"


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
            std::byte *_address{};
            ptr_t _next_allocate_ptr{};
            size_t _capacity{};
        public:
            MemoryBlock() noexcept = default;

            explicit MemoryBlock(size_t byte_size) noexcept {
                //todo strange bug
                _address = alloc<std::byte>(byte_size);
            }

            virtual ~MemoryBlock() noexcept {
                aligned_free(_address);
            }

            LM_NODISCARD PtrInterval interval_used() noexcept {
                return build_interval(reinterpret_cast<ptr_t>(_address), _next_allocate_ptr);
            }

            template<typename T, size_t alignment = alignof(T)>
            LM_NODISCARD T *alloc(size_t byte_size) noexcept {
                static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
                static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
                auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment *
                                  alloc_alignment;
                _capacity = alloc_size;
                _address = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size));
                if (_address == nullptr) {
                    LUMINOUS_ERROR(
                            string_printf("Failed to allocate memory: size = %d, alignment = %d, count = %d",
                                          byte_size, alignment, byte_size / sizeof(T)));
                }
                _next_allocate_ptr = reinterpret_cast<ptr_t>(_address + byte_size);
                return reinterpret_cast<T *>(_address);
            }

            template<typename T, size_t alignment = alignof(T)>
            LM_NODISCARD T *use(size_t byte_size) {
                auto aligned_p = reinterpret_cast<std::byte *>((_next_allocate_ptr + alignment - 1u) /
                                                               alignment * alignment);
                _next_allocate_ptr = reinterpret_cast<ptr_t>(aligned_p + byte_size);
                return reinterpret_cast<T *>(aligned_p);
            }

            LM_NODISCARD std::byte *aligned_ptr(size_t alignment) const {
                return reinterpret_cast<std::byte *>((_next_allocate_ptr + alignment - 1u) /
                                                     alignment * alignment);
            }

            LM_NODISCARD size_t usage() const {
                return _next_allocate_ptr - reinterpret_cast<size_t>(_address);
            }

            LM_NODISCARD double usage_rate() const {
                return double(usage()) / double(_capacity);
            }

            LM_NODISCARD size_t capacity() const { return _capacity; }

            template<typename T = std::byte>
            LM_NODISCARD T *address() { return reinterpret_cast<T *>(_address); }

            LM_NODISCARD std::byte *end_ptr() const { return _address + _capacity; }
        };

        class MemoryArena : public Noncopyable {
        public:

            using BlockList = std::list<MemoryBlock>;
            using BlockIterator = BlockList::iterator;
            using ConstBlockIterator = BlockList::const_iterator;
        private:
            BlockList _memory_blocks;
            size_t _total{0ul};

        public:
            MemoryArena() noexcept = default;

            MemoryArena(MemoryArena &&) noexcept = default;

            MemoryArena &operator=(MemoryArena &&) noexcept = default;

            void clear() { _memory_blocks.clear(); }

            ~MemoryArena() noexcept = default;

            template<typename F>
            void for_each_block(const F &f) const {
                for (auto iter = _memory_blocks.cbegin();
                     iter != _memory_blocks.cend(); ++iter) {
                    f(iter);
                }
            }

            template<typename F>
            void for_each_block(const F &f) {
                for (auto iter = _memory_blocks.begin();
                     iter != _memory_blocks.end(); ++iter) {
                    f(iter);
                }
            }

            LM_NODISCARD size_t capacity() const {
                size_t ret{0u};
                for_each_block([&](ConstBlockIterator iter) {
                    ret += iter->capacity();
                });
                return ret;
            }

            LM_NODISCARD size_t usage() const {
                size_t ret{0u};
                for_each_block([&](ConstBlockIterator iter) {
                    ret += iter->usage();
                });
                return ret;
            }

            LM_NODISCARD double usage_rate() const {
                size_t usage{0u};
                size_t capacity{0u};
                for_each_block([&](ConstBlockIterator iter) {
                    usage += iter->usage();
                    capacity += iter->capacity();
                });
                return double(usage) / double(capacity);
            }

            LM_NODISCARD size_t block_num() const {
                return _memory_blocks.size();
            }

            LM_NODISCARD std::string description() const {
                return string_printf("Memory arena \n "
                                     "the count of block : %u,\n"
                                     "total capacity is : %.5f MB,\n"
                                     "total usage is %.5f MB,\n"
                                     "usage rate is %3.f%%\n",
                                     block_num(),
                                     float(capacity()) / sqr(1024),
                                     float(usage()) / sqr(1024),
                                     usage_rate() * 100);
            }

            BlockIterator find_suitable_blocks(size_t byte_size, size_t alignment) {
                auto best_iter = _memory_blocks.end();
                auto min_remain = block_size;
                for_each_block([&](BlockIterator iter) {
                    auto aligned_p = iter->aligned_ptr(alignment);
                    std::byte *next_allocate_ptr = aligned_p + byte_size;
                    int64_t remain = iter->end_ptr() - next_allocate_ptr;
                    if (remain >= 0 && remain < min_remain) {
                        best_iter = iter;
                        min_remain = remain;
                    }
                });
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

            template<typename T, typename... Args>
            LM_NODISCARD T *create(Args &&...args) {
                return construct_at(allocate<T>(1u), std::forward<Args>(args)...);
            }
        };

    }
}

