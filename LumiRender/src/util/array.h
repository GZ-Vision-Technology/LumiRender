//
// Created by Zero on 2020/11/9.
//

#pragma once

#include <list>
#include <cstddef>
#include "core/header.h"

namespace luminous {
    inline namespace utility {
        /**
          * 2D数组优化，重新排布内存结构
          * 二维数组在内存中的排布实际上是1维连续的
          * 假设二维数组两个维度分别为uv，当索引a[u][v]时，编译器实际上会转成一维数组的索引方式
          * idx = u * width + v
          *
          * BlockedArray有何好处？
          * 数组的默认内存结构，a[u][v]被访问之后，通常周围的内存块容易被访问到
          * 比如说a[u + 1][v]，a[u - 1][v]但这这两个元素通常跟a[u][v]不在一个cache line上，导致cache miss
          *
          * 为了优化目前这个缺点，基本思路，将一连串内存地址分隔为一个个块，块的尺寸为2的整数次幂，每个块逐行排布
          *
          * 假设
          * logBlockSize = 2 ，block_size = 4 每个块的长宽均为4
          * _u_res = 8
          * _v_res = 8
          * _u_blocks = round_up(_u_res) >> logBlockSize = 2
          * nAlloc = round_up(_u_res) * round_up(_v_res) = 64 需要申请64个单位的内存空间
          *
          * 如上参数，内存排布如下，上部分是实际内存中连续地址0-63，下部分是经过下标(u,v)重新映射之后索引的地址，
          * 可以看到，经过重新映射之后,(0,0)与(1,0)是相邻的，并且(0,0)与(0,1)距离不远，
          * 位于同一个cache line的概率也较高，所以可以提高缓存命中率
          *
          * 整个内存片段分为2x2个block，每两个块一行
          *
          * 基本思路是先找到(u,v)所在的块的索引的offset，然后找到块内的索引的offset
          *
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
        template <typename T, int logBlockSize>
        class BlockedArray {

        private:
            T *_data;
            const int _u_res, _v_res, _u_blocks;

        public:
            /**
             * 分配一段连续的内存块，用uv参数重排二维数组的索引
             */
            BlockedArray(int u_res, int v_res, const T *d = nullptr)
                    : _u_res(u_res),
                      _v_res(v_res),
                      _u_blocks(round_up(u_res) >> logBlockSize) {
                // 先向上取到2^logBlockSize
                int nAlloc = round_up(_u_res) * round_up(_v_res);
                _data = alloc_aligned<T>(nAlloc);
                for (int i = 0; i < nAlloc; ++i) {
                    // placement new，在指定地址上调用构造函数
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

            /**
             * 2^logBlockSize
             */
            CONSTEXPR int block_size() const {
                return 1 << logBlockSize;
            }

            /**
             * 向上取到最近的2的logBlockSize次幂
             */
            int round_up(int x) const {
                return (x + block_size() - 1) & ~(block_size() - 1);
            }

            int u_size() const {
                return _u_res;
            }

            int v_size() const {
                return _v_res;
            }

            ~BlockedArray() {
                for (int i = 0; i < _u_res * _v_res; ++i) {
                    _data[i].~T();
                }
                free_aligned(_data);
            }

            /**
             * 返回a * 2^logBlockSize
             */
            int block(int a) const {
                return a >> logBlockSize;
            }

            int offset(int a) const {
                return (a & (block_size() - 1));
            }

            /**
             * 通过uv参数找到指定内存的思路
             * 1.先找到指定内存在哪个块中(bu,bv)
             * 2.然后找到块中的偏移量 (ou, ov)
             */
            inline int get_total_offset(int u, int v) const {
                int bu = block(u);
                int bv = block(v);
                int ou = offset(u);
                int ov = offset(v);
                // 小block的偏移
                int offset = block_size() * block_size() * (_u_blocks * bv + bu);
                // 小block内的偏移
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