//
// Created by Zero on 2020/11/9.
//

#pragma once

#include <list>
#include <cstddef>
#include "arena.h"
#include "base_libs/header.h"

namespace luminous {
    inline namespace core {
        /**
          * 2D array optimization, rearrangement of memory structure
          * The arrangement of a two-dimensional array in memory is actually 1-dimensional continuous
          * When indexing a[u][v], the compiler actually changes to indexing 
          * a one-dimensional array, 
          * assuming that the two-dimensional coordinate are uv
          * then idx = u * width + v
          *
          * What are the benefits of BlockedArray?
          * The default memory structure of an array, 
          * a[u][v] is usually accessible from the surrounding memory blocks
          * 
          * For example, a[u + 1][v] and a[U-1][v] are not on the same cache line with a[u][v],
          * resulting in a cache miss
          *
          * In order to optimize this shortcoming, the basic idea is to divide a series of memory addresses into blocks, 
          * the size of the blocks is an integer power of 2, and each block is arranged line by line
          *
          * The memory addresses are compared as follows
          *
          * |----------0 block----------| |----------1 block----------|
          *
          * |__0___|__1___|__2___|__3___| |__16__|__17__|__18__|__19__|
          * |_0,0__|_1,0__|_2,0__|_3,0__| |_4,0__|_5,0__|_6,0__|_7,0__|
          *
          * |__4___|__5___|__6___|__7___| |__20__|__21__|__22__|__23__|
          * |__0,1_|_1,1__|_2,1__|_3,1__| |_4,1__|_5,1__|_6,1__|_7,1__|
          *
          * |___8__|___9__|__10__|__11__| |__24__|__25__|__26__|__27__|
          * |__0,2_|__1,2_|_2,2__|_3,2__| |_4,2__|_5,2__|_6,2__|_7,2__|
          *
          * |__12__|__13__|__14__|__15__| |__28__|__29__|__30__|__31__|
          * |_0,3__|__1,3_|_2,3__|_3,3__| |_4,3__|_5,3__|_6,3__|_7,3__|
          *
          * |----------2 block----------| |----------3 block----------|
          *
          * |__32__|__33__|__34__|__35__| |__48__|__49__|__50__|__51__|
          * |__0,4_|_1,4__|_2,4__|_3,4__| |__4,4_|_5,4__|_6,4__|_7,4__|
          *
          * |__36__|__37__|__38__|__39__| |__52__|__53__|__54__|__55__|
          * |_0,5__|_1,5__|_2,5__|_3,5__| |_4,5__|_5,5__|_6,5__|_7,5__|
          *
          * |__40__|__41__|__42__|__43__| |__56__|__57__|__58__|__59__|
          * |_0,6__|_1,6__|_2,6__|_3,6__| |_4,6__|_5,6__|_6,6__|_7,6__|
          *
          * |__44__|__45__|__46__|__47__| |__60__|__61__|__62__|__63__|
          * |__0,7_|_1,7__|_2,7__|_3,7__| |_4,7__|_5,7__|_6,7__|_7,7__|
          *
          */
        template <typename T, int logBlockSize = 2>
        class BlockedArray {

        private:
            T *_data;
            const int _u_res{}, _v_res{}, _u_blocks{};

        public:
            BlockedArray() = default;
  
            BlockedArray(int u_res, int v_res, const T *d = nullptr)
                    : _u_res(u_res),
                      _v_res(v_res),
                      _u_blocks(round_up(u_res) >> logBlockSize) {
                int nAlloc = round_up(_u_res) * round_up(_v_res);
                _data = aligned_alloc<T>(nAlloc);
                for (int i = 0; i < nAlloc; ++i) {
                    new (&_data[i]) T();
                }
                if (d) {
                    for (int v = 0; v < _v_res; ++v){
                        for (int u = 0; u < _u_res; ++u) {
                            (*this)(u, v) = d[v * _u_res + u];
                        }
                    }
                }
            }

            LM_NODISCARD constexpr int block_size() const {
                return 1 << logBlockSize;
            }

            LM_NODISCARD int round_up(int x) const {
                return (x + block_size() - 1) & ~(block_size() - 1);
            }

            LM_NODISCARD int u_size() const {
                return _u_res;
            }

            LM_NODISCARD int v_size() const {
                return _v_res;
            }

            LM_NODISCARD size_t size_in_bytes() const {
                return _v_res * _u_res * sizeof(T);
            }

            ~BlockedArray() {
                for (int i = 0; i < _u_res * _v_res; ++i) {
                    _data[i].~T();
                }
                aligned_free(_data);
            }

            LM_NODISCARD int block(int a) const {
                return a >> logBlockSize;
            }

            LM_NODISCARD int offset(int a) const {
                return (a & (block_size() - 1));
            }

            LM_NODISCARD int get_total_offset(int u, int v) const {
                int bu = block(u);
                int bv = block(v);
                int ou = offset(u);
                int ov = offset(v);
                int offset = block_size() * block_size() * (_u_blocks * bv + bu);
                offset += block_size() * ov + ou;
                return offset;
            }

            T &operator()(int u, int v) {
                int offset = get_total_offset(u, v);
                return _data[offset];
            }

            const T &operator()(int u, int v) const {
                int offset = get_total_offset(u, v);
                return _data[offset];
            }

            void get_linear_array(T *a) const {
                for (int v = 0; v < _v_res; ++v) {
                    for (int u = 0; u < _u_res; ++u) {
                        *a++ = (*this)(u, v);
                    }
                }
            }
        };
    }
}