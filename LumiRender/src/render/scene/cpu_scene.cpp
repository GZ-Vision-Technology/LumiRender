//
// Created by Zero on 2021/5/16.
//

#include "cpu_scene.h"
#include "util/stats.h"
#include "cpu/accel/embree_accel.h"

using std::cout;
using std::endl;
namespace luminous {
    inline namespace cpu {

        CPUScene::CPUScene(Device *device, Context *context)
                : Scene(device, context) {}

        void CPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_geometry_data(scene_graph);
            preload_textures(scene_graph);
            init_lights(scene_graph);
            create_device_memory();
            fill_scene_data(scene_graph);
        }

        void CPUScene::create_device_memory() {
            _light_sampler->set_lights(_lights.const_host_buffer_view());
            _light_sampler->set_infinite_lights(_lights.const_host_buffer_view(0, _infinite_light_num));
            _distribution_mgr.init_on_host();
        }

        void CPUScene::fill_scene_data(const SP<SceneGraph> &scene_graph) {
            Scene::fill_scene_data(scene_graph);
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
            _scene_data->cloth_spec_albedos = this->_cloth_spec_albedos.const_host_buffer_view();
            _scene_data->materials = this->_materials.const_host_buffer_view();
        }


    } // luminous::cpu
} // luminous