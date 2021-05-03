//
// Created by Zero on 2021/2/1.
//


#include "gpu_scene.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include "framework/cuda_impl.h"
#include "util/image.h"
#include "util/stats.h"

namespace luminous {
    inline namespace gpu {

        GPUScene::GPUScene(const SP<Device> &device, Context *context)
                : Scene(context), _device(device) {}

        void GPUScene::create_device_memory() {
            {
                // instance data
                _inst_to_mesh_idx.allocate_device(_device);
                _inst_to_transform_idx.allocate_device(_device);
                _transforms.allocate_device(_device);
            }
            {
                // mesh data
                _meshes.allocate_device(_device);
                _positions.allocate_device(_device);
                _tex_coords.allocate_device(_device);
                _triangles.allocate_device(_device);
                _normals.allocate_device(_device);
            }
            {
                // light data
                _lights.allocate_device(_device);
                _emission_distrib.init_on_device(_device);
                _light_sampler.allocate_device(_device);
            }
            {
                // texture data
                _textures.allocate_device(_device);
                _materials.allocate_device(_device);
            }
        }

        void GPUScene::synchronize_to_gpu() {
            {
                // instance data
                _inst_to_mesh_idx.synchronize_to_gpu();
                _inst_to_transform_idx.synchronize_to_gpu();
                _transforms.synchronize_to_gpu();
            }
            {
                // mesh data
                _meshes.synchronize_to_gpu();
                _positions.synchronize_to_gpu();
                _tex_coords.synchronize_to_gpu();
                _triangles.synchronize_to_gpu();
                _normals.synchronize_to_gpu();
            }
            {
                // light data
                _lights.synchronize_to_gpu();
                _emission_distrib.synchronize_to_gpu();
                _light_sampler->set_lights(_lights.device_buffer_view());
                _light_sampler.synchronize_to_gpu();
            }
            {
                // texture data
                _textures.synchronize_to_gpu();
                _materials.synchronize_to_gpu();
            }
        }

        void GPUScene::init_accel() {
            _optix_accel = make_unique<OptixAccel>(_device, this);
            build_accel();
        }

        void GPUScene::preload_textures(const SP<SceneGraph> &scene_graph) {
            TASK_TAG("preload_textures")
            for (auto &tc : _tex_configs) {
                if (tc.type() == type_name<ImageTexture>()) {
                    if (tc.fn.empty()) {
                        continue;
                    }
                    if (filesystem::path(tc.fn).is_relative()) {
                        auto path = _context->scene_path() / tc.fn;
                        tc.fn = path.string();
                    }
                    auto image = Image::load(tc.fn, tc.color_space);
                    auto texture = _device->allocate_texture(image.pixel_format(), image.resolution());
                    texture.copy_from(image);
                    tc.handle = texture.tex_handle();
                    tc.pixel_format = texture.pixel_format();
                    _texture_mgr.push_back(move(texture));
                    _texture_num += 1;
                    _texture_size_in_byte += image.size_in_bytes();
                }
                _textures.push_back(Texture::create(tc));
            }
        }

        void GPUScene::init(const SP<SceneGraph> &scene_graph) {
            convert_data(scene_graph);
            preload_textures(scene_graph);
            create_device_memory();
            synchronize_to_gpu();
            init_accel();
        }

        void GPUScene::build_accel() {
            _optix_accel->build_bvh(_positions.device_buffer(),
                                    _triangles.device_buffer(),
                                    _meshes,
                                    _inst_to_mesh_idx,
                                    _transforms,
                                    _inst_to_transform_idx);
            cout << _optix_accel->description() << endl;
            cout << description() << endl;
        }

        size_t GPUScene::size_in_bytes() const {
            return Scene::size_in_bytes();
        }

        std::string GPUScene::description() const {
            return Scene::description();
        }

    } // luminous::gpu
} // luminous