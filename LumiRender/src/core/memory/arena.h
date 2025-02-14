//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/concepts.h"
#include "base_libs/string_util.h"
#include "core/logging.h"
#include "util.h"
#include "base_libs/math/common.h"


namespace luminous {
    inline namespace core {
        static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

        struct MemoryBlock {
        private:
            std::byte *_address{0};
            ptr_t _next_allocate_ptr{0};
            size_t _capacity{0};
        public:
            explicit MemoryBlock() noexcept = default;

            explicit MemoryBlock(size_t byte_size) noexcept {
                //todo strange bug
                _address = allocate_and_use<std::byte>(byte_size);
            }

            virtual ~MemoryBlock() noexcept {
                clear();
            }

            void clear() noexcept {
                if (_address) {
                    aligned_free(_address);
                }
                _address = nullptr;
                _next_allocate_ptr = 0;
                _capacity = 0;
            }

            LM_NODISCARD bool valid() const noexcept { return bool(_address); }

            LM_NODISCARD PtrInterval interval_used() const noexcept {
                return build_interval(reinterpret_cast<ptr_t>(_address), _next_allocate_ptr);
            }

            LM_NODISCARD PtrInterval interval_allocated() const noexcept {
                ptr_t begin = reinterpret_cast<ptr_t>(_address);
                ptr_t end = begin + capacity();
                return build_interval(begin, end);
            }

            void *allocate(size_t byte_size) noexcept {
                _capacity = byte_size;
                _address = aligned_alloc<std::byte>(byte_size);
                if (_address == nullptr) {
                    LUMINOUS_ERROR(
                            string_printf("Failed to allocate memory: size = %u",byte_size));
                }
                _next_allocate_ptr = reinterpret_cast<ptr_t>(_address);
                return _address;
            }

            template<typename T = std::byte, size_t alignment = alignof(T)>
            T *allocate_and_use(size_t byte_size) noexcept {
                static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
                static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
                auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment *
                                  alloc_alignment;
                _capacity = alloc_size;
                _address = aligned_alloc<std::byte>(alloc_alignment, alloc_size);
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

            template<typename T = std::byte>
            LM_NODISCARD const T *address() const { return reinterpret_cast<const T *>(_address); }

            LM_NODISCARD std::byte *end_ptr() const { return _address + _capacity; }
        };

        class MemoryArena : public Noncopyable {
        public:
            using BlockList = std::list<MemoryBlock>;
            using BlockIterator = BlockList::iterator;
            using ConstBlockIterator = BlockList::const_iterator;
        private:
            BlockList _memory_blocks{};
            MemoryBlock *_external_block{nullptr};
        public:
            MemoryArena() noexcept = default;

            MemoryArena(MemoryArena &&) noexcept = default;

            MemoryArena &operator=(MemoryArena &&) noexcept = default;

            void clear() {
                _memory_blocks.clear();
            }

            void reset_external_block(MemoryBlock *memory_block = nullptr) noexcept {
                DCHECK(memory_block == nullptr || _external_block == nullptr);
                _external_block = memory_block;
            }

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

            LM_NODISCARD MemoryBlock *find_suitable_blocks(size_t byte_size, size_t alignment) {
                if (_external_block) {
                    return _external_block;
                }
                MemoryBlock *best_block = nullptr;
                auto min_remain = block_size;
                for_each_block([&](BlockIterator iter) {
                    auto aligned_p = iter->aligned_ptr(alignment);
                    std::byte *next_allocate_ptr = aligned_p + byte_size;
                    int64_t remain = iter->end_ptr() - next_allocate_ptr;
                    if (remain >= 0 && remain < min_remain) {
                        best_block = &(*iter);
                        min_remain = remain;
                    }
                });
                return best_block;
            }

            template<typename T = std::byte, size_t alignment = alignof(T)>
            LM_NODISCARD T *allocate(size_t n = 1u) {
                static constexpr auto size = sizeof(T);
                size_t byte_size = n * size;
                MemoryBlock *block = find_suitable_blocks(byte_size, alignment);

                if (block == nullptr) {
                    _memory_blocks.emplace_back();
                    MemoryBlock &back = _memory_blocks.back();
                    return back.template allocate_and_use<T>(byte_size);
                } else {
                    return block->template use<T>(byte_size);
                }
            }

            template<typename T, typename... Args>
            LM_NODISCARD T *create(Args &&...args) {
                return construct_at(allocate<T>(1u), std::forward<Args>(args)...);
            }
        };

    }
}

