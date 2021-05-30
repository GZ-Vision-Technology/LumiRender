//
// Created by Zero on 2021/5/16.
//

#include "embree_accel.h"
#include "cpu_scene.h"

namespace luminous {
    inline namespace cpu {


        EmbreeAccel::EmbreeAccel(const CPUScene *cpu_scene) {

        }

        std::string EmbreeAccel::description() const {
            return nullptr;
        }

        void EmbreeAccel::build_bvh(const vector<const float3> &positions,
                                    const vector<const TriangleHandle> &triangles,
                                    const vector<MeshHandle> &meshes, const vector<uint> &instance_list,
                                    const vector<Transform> &transform_list, const vector<uint> &inst_to_transform) {

        }
    } // luminous::cpu
} // luminous