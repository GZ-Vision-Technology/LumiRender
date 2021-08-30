//
// Created by Zero on 22/08/2021.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "gpu/framework/cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "render/include/scene_graph.h"
#include "shader_wrapper.h"
#include "core/backend/managed.h"
#include "core/hash.h"
#include "render/include/accelerator.h"

namespace luminous {
    inline namespace gpu {

//        class GPUScene;

//        class Scene;

        class OptixAccel : public Accelerator {
        private:
            std::map<SHA1, OptixModule> _module_map;

            void clear_modules() {
                for (const auto& iter : _module_map) {
                    optixModuleDestroy(iter.second);
                }
            }

            OptixModule create_module(const std::string &ptx_code);

            static SHA1 generate_key(const std::string &ptx_code) {
                return SHA1(ptx_code);
            }

            NDSC bool is_contain(const SHA1 &key) const {
                return _module_map.find(key) != _module_map.end();
            }

        protected:
            Context *_context{};
            OptixDeviceContext _optix_device_context{};
            OptixPipelineCompileOptions _pipeline_compile_options = {};
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            OptixTraversableHandle _root_as_handle{};
            std::list<Buffer<std::byte>> _as_buffer_list;
            Buffer<OptixInstance> _instances{nullptr};
            OptixPipeline _optix_pipeline{};
//            const GPUScene *_gpu_scene{};

            OptixModule obtain_module(const std::string &ptx_code) {
                SHA1 key = generate_key(ptx_code);
                if (is_contain(key)) {
                    return _module_map[key];
                }
                OptixModule optix_module = create_module(ptx_code);
                _module_map[key] = optix_module;
                return optix_module;
            }

            NDSC OptixBuildInput get_mesh_build_input(const Buffer<const float3> &positions,
                                                      const Buffer<const TriangleHandle> &triangles,
                                                      const MeshHandle &mesh,
                                                      std::list<CUdeviceptr> &vert_buffer_ptr);

            NDSC OptixTraversableHandle build_mesh_bvh(const Buffer<const float3> &positions,
                                                       const Buffer<const TriangleHandle> &triangles,
                                                       const MeshHandle &mesh,
                                                       std::list<CUdeviceptr> &_vert_buffer_ptr);

        public:
            OptixAccel(const SP<Device> &device, Context *context, const Scene *scene);

            OptixDeviceContext create_context();

            NDSC uint64_t handle() const override { return _root_as_handle; }

            void build_pipeline(const std::vector<OptixProgramGroup> &program_groups);

            NDSC ShaderWrapper create_shader_wrapper(const std::string &ptx_code, const ProgramName &program_name);

            void clear() override {
                optixPipelineDestroy(_optix_pipeline);
                clear_modules();
                _as_buffer_list.clear();
                optixDeviceContextDestroy(_optix_device_context);
            }

            NDSC std::string description() const override {
                float size_in_M = (_bvh_size_in_bytes * 1.f) / (sqr(1024));
                return string_printf("bvh size is %f MB\n", size_in_M);
            }

            void build_bvh(const Managed<float3> &positions, const Managed<TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) override;

        };
    }
}