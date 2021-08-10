//
// Created by Zero on 2021/5/16.
//

#include "cpu_scene.h"
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
            init_accel();
        }

        void CPUScene::init_accel() {
            EmbreeAccel::init_device();
            _embree_accel = std::make_unique<EmbreeAccel>(this);
            build_accel();
        }

        void CPUScene::preload_textures(const SP<SceneGraph> &scene_graph) {
            // todo implement
        }

        void CPUScene::build_accel() {
            _embree_accel->build_bvh(_positions, _triangles, _meshes, _inst_to_mesh_idx,
                                     _transforms, _inst_to_transform_idx);

        }

    } // luminous::cpu
} // luminous