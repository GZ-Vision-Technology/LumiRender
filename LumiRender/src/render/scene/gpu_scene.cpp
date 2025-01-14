//
// Created by Zero on 2021/2/1.
//


#include "gpu_scene.h"
#include <cassert>
#include <iostream>

using std::cout;
using std::endl;

namespace luminous {
    inline namespace gpu {

        GPUScene::GPUScene(Device *device, Context *context)
                : Scene(device, context) {}

        void GPUScene::create_device_memory() {
            {
                // instance data
                _inst_to_mesh_idx.allocate_device();
                _inst_to_transform_idx.allocate_device();
                _transforms.allocate_device();
            }
            {
                // mesh data
                _meshes.allocate_device();
                _positions.allocate_device();
                _tex_coords.allocate_device();
                _triangles.allocate_device();
                _normals.allocate_device();
            }
            {
                // light data
                _distribution_mgr.init_on_device(_device);
            }
            {
                // texture data
                _textures.allocate_device();
                _cloth_spec_albedos.allocate_device();
            }
            _scene_data.allocate_device();
        }

        void GPUScene::fill_scene_data(const SP<SceneGraph> &scene_graph) {
            Scene::fill_scene_data(scene_graph);
            _scene_data->positions = this->_positions.const_device_buffer_view();
            _scene_data->normals = this->_normals.const_device_buffer_view();
            _scene_data->tex_coords = this->_tex_coords.const_device_buffer_view();
            _scene_data->triangles = this->_triangles.const_device_buffer_view();
            _scene_data->meshes = this->_meshes.const_device_buffer_view();

            _scene_data->inst_to_mesh_idx = this->_inst_to_mesh_idx.const_device_buffer_view();
            _scene_data->inst_to_transform_idx = this->_inst_to_transform_idx.const_device_buffer_view();
            _scene_data->transforms = this->_transforms.const_device_buffer_view();

            _scene_data->light_sampler = this->_light_sampler.device_ptr();
            _scene_data->distributions = this->_distribution_mgr.distributions.const_device_buffer_view();
            _scene_data->distribution2ds = this->_distribution_mgr.distribution2ds.const_device_buffer_view();

            _scene_data->materials = this->_materials.const_device_buffer_view();
            _scene_data->textures = this->_textures.const_device_buffer_view();
            _scene_data->cloth_spec_albedos = this->_cloth_spec_albedos.const_device_buffer_view();
            _scene_data.synchronize_to_device();
        }

        void GPUScene::synchronize_to_gpu() {
            {
                // instance data
                _inst_to_mesh_idx.synchronize_to_device();
                _inst_to_transform_idx.synchronize_to_device();
                _transforms.synchronize_to_device();
            }
            {
                // mesh data
                _meshes.synchronize_to_device();
                _positions.synchronize_to_device();
                _tex_coords.synchronize_to_device();
                _triangles.synchronize_to_device();
                _normals.synchronize_to_device();
            }
            {
                // light data
                _lights.synchronize_all_to_device();
                _distribution_mgr.synchronize_to_device();
                _light_sampler->set_lights(_lights.const_device_buffer_view());
                _light_sampler->set_infinite_lights(_lights.const_device_buffer_view(0,_infinite_light_num));
                _light_sampler.synchronize_all_to_device();
            }
            {
                // texture data
                _textures.synchronize_to_device();
                _cloth_spec_albedos.synchronize_to_device();
                _materials.synchronize_all_to_device();
            }
        }

        void GPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_geometry_data(scene_graph);
            preload_textures(scene_graph);
            init_lights(scene_graph);
            create_device_memory();
            synchronize_to_gpu();
            fill_scene_data(scene_graph);
            shrink_to_fit();
        }

        void GPUScene::clear() {
            Scene::clear();
        }

        size_t GPUScene::size_in_bytes() const {
            return Scene::size_in_bytes();
        }

        std::string GPUScene::description() const {
            return Scene::description();
        }

    } // luminous::gpu
} // luminous