//
// Created by Zero on 16/10/2021.
//


#pragma once

#include "cpu/cpu_impl.h"
#include "gpu/framework/cuda_impl.h"

namespace luminous {
    inline namespace render {

        template<typename T>
        class Kernel {
        public:
            using func_type = T;
            using function_trait = FunctionTrait<func_type>;
        protected:
            // for cpu backend
            func_type _func{};

            // for cuda backend
            CUfunction _cu_func{};
            uint3 _grid_size = make_uint3(1);
            uint3 _block_size = make_uint3(5);
            int _auto_block_size = 0;
            int _min_grid_size = 0;
            size_t _shared_mem = 1024;
        private:
            // n_item_idx is the index of n_item arg in Args
            static constexpr int n_item_idx = 0;

            template<typename Ret, typename...Args, size_t...Is>
            void call_impl(Ret(*f)(Args...), void *args[], std::index_sequence<Is...>) {
                f(*static_cast<std::tuple_element_t<Is, std::tuple<Args...>> *>(args[Is])...);
            }

            template<typename Ret, typename...Args>
            void call(Ret(*f)(Args...), void *args[]) {
                call_impl(f, args, std::make_index_sequence<sizeof...(Args)>());
            }

            template<typename Ret, typename...Args, size_t...Is>
            void check_signature(Ret(*f)(Args...), std::index_sequence<Is...>) {
                using OutArgs = std::tuple<Args...>;
                static_assert(std::is_invocable_v<func_type, std::tuple_element_t<Is, OutArgs>...>);
            }

            template<typename ...Args>
            void cpu_launch(Args &&...args) {
//                async(1, [&](uint idx, uint tid) {
                    _func(std::forward<Args>(args)...);
//                });
            }

            template<typename ...Args>
            void cuda_launch(Dispatcher &dispatcher, Args &&...args) {
                void *array[]{(&args)...};
                auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
//                CU_CHECK(cuLaunchKernel(_cu_func, _grid_size.x, _grid_size.y, _grid_size.z,
//                                        _block_size.x, _block_size.y, _block_size.z,
//                                        _shared_mem, stream, array, nullptr));
            }

            template<typename...Args>
            void launch_func_impl(Dispatcher &dispatcher, Args &&...args) {
                check_signature(_func, std::make_index_sequence<sizeof...(Args)>());

                if (on_cpu()) {
                    
                    cpu_launch(std::forward<Args>(args)...);
                } else {
                    cuda_launch(dispatcher, std::forward<Args>(args)...);
                }
            }

            template<typename Ret, typename...Args, typename ...OutArgs>
            void launch_func(Dispatcher &dispatcher, Ret(*f)(Args...), OutArgs &&...args) {
                launch_func_impl(dispatcher, (static_cast<Args>(std::forward<OutArgs>(args)))...);
            }

        public:

            explicit Kernel(func_type func) : _func(func) {}

            LM_NODISCARD bool on_cpu() const {
                return _cu_func == 0;
            }

            void compute_fit_size() {
                if (on_cpu()) {
                    return;
                }
                cuOccupancyMaxPotentialBlockSize(&_min_grid_size, &_auto_block_size,
                                                 _cu_func, 0, _shared_mem, 0);
            }

            void configure(uint3 grid_size,
                           uint3 local_size,
                           size_t sm) {
                if (on_cpu()) {
                    return;
                }
                _shared_mem = sm == 0 ? _shared_mem : sm;
                _grid_size = grid_size;
                _block_size = local_size;
            }

            void set_cu_function(uint64_t handle) {
                _cu_func = reinterpret_cast<CUfunction>(handle);
            }

            /**
             * @tparam Args : The first two parameters are num of item and index, respectively
             * @param dispatcher
             * @param args
             */
            template<typename...Args>
            void launch(Dispatcher &dispatcher, Args &&...args) {
                launch_func(dispatcher, _func, std::forward<Args>(args)...);
            }
        };
    }
}