//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "parser/shape.h"
#include "parser/scene_graph.h"
#include "render/lights/light.h"
#include "base_libs/sampling/common.h"
#include "render/light_samplers/light_sampler.h"
#include "render/textures/texture.h"
#include "render/include/distribution_mgr.h"
#include "core/backend/synchronizer.h"
#include "render/materials/material.h"
#include "render/include/accelerator.h"
#include "scene_data.h"

using std::cout;
using std::endl;

namespace luminous {

    inline namespace render {

        /**
         * all memory data manage
         */
        class Scene : public Noncopyable {
        protected:
            Device *_device{};
            Context *_context{nullptr};
            size_t _inst_vertices_num{0};
            size_t _inst_triangle_num{0};
            size_t _texture_size_in_byte{0};
            size_t _texture_num{0};
            int _infinite_light_num{0};

            Managed<SceneData> _scene_data{_device};

            Box3f _scene_box;

            // instance data
            Managed<uint> _inst_to_mesh_idx{_device};
            Managed<uint> _inst_to_transform_idx{_device};
            Managed<Transform> _transforms{_device};

            // mesh data
            Managed<MeshHandle> _meshes{_device};
            Managed<float3> _positions{_device};
            Managed<float3> _normals{_device};
            Managed<float2> _tex_coords{_device};
            Managed<TriangleHandle> _triangles{_device};

            // light data
            Synchronizer<Light> _lights{_device};
            Synchronizer<LightSampler> _light_sampler{_device};
            DistributionMgr _distribution_mgr{_device};

            // texture data
            Managed<Texture> _textures{_device};

            // cloth sheen layer preload albedo textures
            Managed<ImageTexture> _cloth_spec_albedos;

            // material data, the last element is light material, black diffuse
            Synchronizer<Material> _materials{_device};

            // prepare for texture out of core render
            vector<MaterialAttrConfig> _tex_configs;

            vector<Image> _images;

            UP<Accelerator> _accelerator;

        public:
            Scene(Device *device, Context *context)
                    : _device(device), _context(context) {
                _scene_data.reset(1);
            }

            void shrink_to_fit();

            LM_NODISCARD virtual std::string description() const;

            LM_NODISCARD virtual size_t size_in_bytes() const;

            virtual void clear();

            virtual void reserve_geometry(const SP<SceneGraph> &scene_graph);

            virtual void fill_scene_data(const SP<SceneGraph> &scene_graph) {
#if DEBUG_RENDER
                _scene_data->debug_pixel = scene_graph->debug_config.pixel;
                _scene_data->mis_mode = scene_graph->debug_config.mis_mode;
#endif
            }

            LM_NODISCARD const SceneData *scene_data_host_ptr() const { return _scene_data.data(); }

            LM_NODISCARD const SceneData *
            scene_data_device_ptr() const { return _scene_data.device_ptr<const SceneData *>(); }

            LM_NODISCARD uint64_t as_handle() const { return _accelerator->handle(); }

            template<typename TAccel>
            LM_NODISCARD TAccel *accel() const {
                DCHECK(dynamic_cast<TAccel *>(_accelerator.get()));
                return dynamic_cast<TAccel *>(_accelerator.get());
            }

            void init_materials(const SP<SceneGraph> &scene_graph);

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            virtual void create_device_memory() = 0;

            template<typename TAccel>
            void init_accel() {
                _accelerator = std::make_unique<TAccel>(_device, _context, this);
                _accelerator->build_bvh(_positions, _triangles, _meshes, _inst_to_mesh_idx,
                                        _transforms, _inst_to_transform_idx);
                cout << _accelerator->description() << endl;
                cout << description() << endl;
            }

            void append_light_material(vector<MaterialConfig> &material_configs);

            void convert_geometry_data(const SP<SceneGraph> &scene_graph);

            void init_lights(const SP<SceneGraph> &scene_graph);

            void preload_textures(const SP<SceneGraph> &scene_graph);

            void relevance_material_and_texture(vector<MaterialConfig> &material_configs);

            void relevance_light_and_texture(vector<LightConfig> &light_configs);

            void load_lights(const vector<LightConfig> &lc, const LightSamplerConfig &lsc);

            void preprocess_meshes();
        };
    }
}