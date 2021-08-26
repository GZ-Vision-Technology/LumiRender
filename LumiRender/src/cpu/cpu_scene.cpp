//
// Created by Zero on 2021/5/16.
//

#include "cpu_scene.h"
#include "util/stats.h"
#include "embree_accel.h"

using std::cout;
using std::endl;
namespace luminous {
    inline namespace cpu {

        CPUScene::CPUScene(Context *context)
                : Scene(context) {}

        void CPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_geometry_data(scene_graph);
            preload_textures(scene_graph);
            init_lights(scene_graph);
            create_device_memory();
            init_accel();
            fill_scene_data();
        }

        void CPUScene::create_device_memory() {
            _light_sampler->set_lights(_lights.const_host_buffer_view());
            _distribution_mgr.init_on_host();
        }

        void CPUScene::init_accel() {
            EmbreeAccel::init_device();
            _embree_accel = std::make_unique<EmbreeAccel>();
            build_accel();
        }

        void CPUScene::preload_textures(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("preload_textures")
            for (auto &tc : _tex_configs) {
                _textures.push_back(Texture::create(tc));
            }
        }

        void CPUScene::build_accel() {
            _embree_accel->build_bvh(_positions, _triangles, _meshes, _inst_to_mesh_idx,
                                     _transforms, _inst_to_transform_idx);

        }

        void CPUScene::fill_scene_data() {
            _scene_data->positions = this->_positions.const_host_buffer_view();
            _scene_data->normals = this->_normals.const_host_buffer_view();
            _scene_data->tex_coords = this->_tex_coords.const_host_buffer_view();
            _scene_data->triangles = this->_triangles.const_host_buffer_view();
            _scene_data->meshes = this->_meshes.const_host_buffer_view();

            _scene_data->inst_to_mesh_idx = this->_inst_to_mesh_idx.const_host_buffer_view();
            _scene_data->inst_to_transform_idx = this->_inst_to_transform_idx.const_host_buffer_view();
            _scene_data->transforms = this->_transforms.const_host_buffer_view();

            _scene_data->light_sampler = this->_light_sampler.data();
            _scene_data->distributions = this->_distribution_mgr.distributions.const_host_buffer_view();
            _scene_data->distribution2ds = this->_distribution_mgr.distribution2ds.const_host_buffer_view();

            _scene_data->textures = this->_textures.const_host_buffer_view();
            _scene_data->materials = this->_materials.const_host_buffer_view();
        }


    } // luminous::cpu
} // luminous