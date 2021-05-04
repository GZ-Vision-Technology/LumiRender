////
//// Created by Zero on 2021/5/4.
////
//
//
//#ifndef IS_GPU_CODE
//#include "shader_data.h"
//#include "distribution.h"
//#include "render/materials/material.h"
//
//namespace luminous {
//    inline namespace render {
//
//        const Texture &HitGroupData::get_texture(index_t idx) const {
//            return textures[idx];
//        }
//
//        const Material &HitGroupData::get_material(index_t inst_id) const {
//            MeshHandle mesh = get_mesh(inst_id);
//            return materials[mesh.material_idx];
//        }
//
//        const Distribution1D &HitGroupData::get_distrib(index_t inst_id) const {
//            MeshHandle mesh = get_mesh(inst_id);
//            return emission_distributions[mesh.distribute_idx];
//        }
//    }
//}
//#endif