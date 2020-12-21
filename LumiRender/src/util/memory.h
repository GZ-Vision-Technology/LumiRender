//
// Created by Zero on 2020/11/6.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/header.h"

#define ARENA_ALLOC(arena, Type) new ((arena).alloc(sizeof(Type))) Type

namespace luminous {
    inline namespace utility {
        void *alloc_aligned(size_t size);
        template <typename T>
        T *alloc_aligned(size_t count) {
            return (T *)alloc_aligned(count * sizeof(T));
        }

        void free_aligned(void *);


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
        class alignas(L1_CACHE_LINE_SIZE) MemoryArena {

        private:
            MemoryArena(const MemoryArena &) = delete;
            MemoryArena &operator=(const MemoryArena &) = delete;

            // 默认内存块大小
            const size_t _block_size;

            // 当前块已经分配的位置
            size_t _current_block_pos = 0;

            // 当前块的尺寸
            size_t _current_alloc_size = 0;

            // 当前块指针
            uint8_t *_current_block = nullptr;

            // 已经使用的内存块列表
            std::list<std::pair<size_t, uint8_t *>> _used_blocks;

            // 可使用的内存块列表
            std::list<std::pair<size_t, uint8_t *>> _available_blocks;

        public:

            /**
             * MemoryArena是内存池的一种，基于arena方式分配内存
             * 整个对象大体分为三个部分
             *   1.可用列表，用于储存可分配的内存块
             *   2.已用列表，储存已经使用的内存块
             *   3.当前内存块
             */
            MemoryArena(size_t block_size = 262144) : _block_size(block_size) {

            }

            ~MemoryArena() {
                free_aligned(_current_block);

                for (auto &block : _used_blocks) {
                    free_aligned(block.second);
                }

                for (auto &block : _available_blocks) {
                    free_aligned(block.second);
                }
            }

            /**
             * 1.对齐内存块
             * 2.
             */
            void * alloc(size_t nBytes) {

                // 16位对齐，对齐之后nBytes为16的整数倍
                nBytes = (nBytes + 15) & ~(15);
                if (_current_block_pos + nBytes > _current_alloc_size) {
                    // 如果已经分配的内存加上请求内存大于当前内存块大小

                    if (_current_block) {
                        // 如果当前块不为空，则把当前块放入已用列表中
                        _used_blocks.push_back(std::make_pair(_current_alloc_size, _current_block));
                        _current_block = nullptr;
                        _current_alloc_size = 0;
                    }

                    // 在可用列表中查找是否有尺寸大于请求内存的块
                    for (auto iter = _available_blocks.begin(); iter != _available_blocks.end(); ++iter) {
                        if (iter->first >= nBytes) {
                            // 如果找到将当前块指针指向该块，并将该块从可用列表中移除
                            _current_alloc_size = iter->first;
                            _current_block = iter->second;
                            _available_blocks.erase(iter);
                            break;
                        }
                    }

                    if (!_current_block) {
                        // 如果没有找到符合标准的内存块，则申请一块内存
                        _current_alloc_size = std::max(nBytes, _block_size);
                        _current_block = alloc_aligned<uint8_t>(_current_alloc_size);
                    }
                    _current_block_pos = 0;
                }
                void * ret = _current_block + _current_block_pos;
                _current_block_pos += nBytes;
                return ret;
            }

            /**
             * 外部向内存池申请一块连续内存
             */
            template <typename T>
            T * alloc(size_t n = 1, bool runConstructor = true) {
                T *ret = (T *)alloc(n * sizeof(T));
                if (runConstructor) {
                    for (size_t i = 0; i < n; ++i) {
                        new (&ret[i]) T();
                    }
                }
                return ret;
            }

            /**
             * 重置当前内存池，将可用列表与已用列表合并，已用列表在可用列表之前
             */
            void reset() {
                _current_block_pos = 0;
                _available_blocks.splice(_available_blocks.begin(), _used_blocks);
            }

            //获取已经分配的内存大小
            size_t total_allocated() const {
                size_t total = _current_alloc_size;
                for (const auto &alloc : _used_blocks) {
                    total += alloc.first;
                }
                for (const auto &alloc : _available_blocks) {
                    total += alloc.first;
                }
                return total;
            }

        };

    }
}

