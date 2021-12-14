//
// Created by Zero on 22/08/2021.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "gpu/framework/cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "parser/scene_graph.h"
#include "shader_wrapper.h"
#include "core/backend/managed.h"
#include "core/hash.h"
#include "render/include/accelerator.h"

namespace luminous {
    inline namespace gpu {

        class OptixAccel : public Accelerator {
        private:
            std::map<SHA1, OptixModule> _module_map;

            void clear_modules() {
                for (const auto &iter : _module_map) {
                    optixModuleDestroy(iter.second);
                }
            }

            OptixModule create_module(const std::string_view &ptx_code);

            static SHA1 generate_key(const std::string_view &ptx_code) {
                return SHA1(ptx_code);
            }

            LM_NODISCARD bool is_contain(const SHA1 &key) const {
                return _module_map.find(key) != _module_map.end();
            }

        protected:
            Context *_context{};
            OptixDeviceContext _optix_device_context{};
            OptixPipelineCompileOptions _pipeline_compile_options = {};
            Device *_device{};
            mutable Dispatcher _dispatcher;
            OptixTraversableHandle _root_as_handle{};
            std::list<Buffer<std::byte>> _as_buffer_list;
            Buffer<OptixInstance> _instances{nullptr};
            OptixPipeline _optix_pipeline{};

            OptixModule obtain_module(const std::string_view &ptx_code) {
                SHA1 key = generate_key(ptx_code);
                if (is_contain(key)) {
                    return _module_map[key];
                }
                OptixModule optix_module = create_module(ptx_code);
                _module_map[key] = optix_module;
                return optix_module;
            }

            LM_NODISCARD OptixBuildInput get_mesh_build_input(const Buffer<const float3> &positions,
                                                      const Buffer<const TriangleHandle> &triangles,
                                                      const MeshHandle &mesh,
                                                      std::list<CUdeviceptr> &vert_buffer_ptr);

            LM_NODISCARD OptixTraversableHandle build_mesh_bvh(const Buffer<const float3> &positions,
                                                       const Buffer<const TriangleHandle> &triangles,
                                                       const MeshHandle &mesh,
                                                       std::list<CUdeviceptr> &_vert_buffer_ptr);

        public:
            OptixAccel(Device *device, Context *context, const Scene *scene);

            OptixDeviceContext create_context();

            LM_NODISCARD uint64_t handle() const override { return _root_as_handle; }

            void build_pipeline(const std::vector<OptixProgramGroup> &program_groups);

            LM_NODISCARD ShaderWrapper create_shader_wrapper(const std::string_view &ptx_code, const ProgramName &program_name);

            void clear() override {
                optixPipelineDestroy(_optix_pipeline);
                clear_modules();
                _as_buffer_list.clear();
                optixDeviceContextDestroy(_optix_device_context);
            }

            LM_NODISCARD std::string description() const override {
                float size_in_M = (_bvh_size_in_bytes * 1.f) / (sqr(1024));
                return string_printf("bvh size is %f MB\n", size_in_M);
            }

            void build_bvh(const Managed<float3> &positions, const Managed<TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) override;

        };
    }
}