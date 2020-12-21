//
// Created by Zero on 2020/11/3.
//

#pragma once

#include "core/header.h"
#include <atomic>
#include <functional>
#include <future>
#include "core/math/data_types.h"

namespace luminous {
    inline namespace utility {

        class AtomicFloat {
        private:
            std::atomic<float> val;

        public:
            using Float = float;
            explicit AtomicFloat(Float v = 0) : val(v) {}

            AtomicFloat(const AtomicFloat &rhs) : val((float)rhs.val) {}

            void add(Float v) {
                auto current = val.load();
                while (!val.compare_exchange_weak(current, current + v)) {
                }
            }

            [[nodiscard]] float value() const { return val.load(); }

            explicit operator float() const { return value(); }

            void set(Float v) { val = v; }
        };

        void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize = 1);

        [[nodiscard]] size_t num_work_threads();

        inline void parallel_for_2d(const int2 &dim, const std::function<void(int2, uint32_t)> &func,
                                    size_t chunkSize = 1) {
            parallel_for(
                    dim.x * dim.y,
                    [&](uint32_t idx, int tid) {
                        auto x = idx % dim.x;
                        auto y = idx / dim.x;
                        func(make_int2(x, y), tid);
                    },
                    chunkSize);
        }


        template <class F>
        void tiled_for_2d(const int2 &dim, const int2 &tile_size, bool parallel, F &&func) {
            auto tiles = (dim + tile_size - make_int2(1)) / tile_size;
            auto body = [&](int2 t, uint32_t) {
                for (int ty = 0; ty < tile_size[1]; ty++) {
                    for (int tx = 0; tx < tile_size[0]; tx++) {
                        int x = tx + t[0] * tile_size.x;
                        int y = ty + t[1] * tile_size.y;
                        func(make_float2(x, y));
                    }
                }
            };
            if (parallel) {
                parallel_for_2d(tiles, body);
            } else {
                for (int j = 0; j < tiles[1]; j++) {
                    for (int i = 0; i < tiles[0]; i++) {
                        body(make_float2(i, j), 0);
                    }
                }
            }
        }

        namespace thread {
            void finalize();
        }

        template <typename T>
        class Future {
            std::future<T> inner;
            template <typename R>
            friend class Future;

        public:
            Future(std::future<T> ft) : inner(std::move(ft)) {}
            template <typename F, typename R = std::invoke_result_t<F, decltype(std::declval<std::future<T>>().get())>>
            auto then(F &&f, std::launch policy = std::launch::deferred) -> Future<R> {
                return Future<R>(std::async(std::launch::deferred, [=, ft = std::move(inner)]() mutable {
                    if constexpr (std::is_same_v<T, void>) {
                        ft.get();
                        return f();
                    } else {
                        decltype(auto) result = ft.get();
                        return f(result);
                    }
                }));
            }
        };

        template <class _Fty, class... _ArgTypes>
        Future<std::invoke_result_t<std::decay_t<_Fty>, std::decay_t<_ArgTypes>...>>
        async_do(std::launch policy, _Fty &&_Fn_arg, _ArgTypes &&... _Args) {
            return std::async(policy, std::forward<_Fty>(_Fn_arg), std::forward<_ArgTypes>(_Args)...);
        }
    }
}