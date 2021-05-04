//
// Created by Zero on 2021/4/12.
//


#pragma once

#include "graphics/math/common.h"
#include "core/backend/buffer_view.h"
#include "vector_types.h"

namespace luminous {
    inline namespace render {

        struct RayGenData {

        };

        struct MissData {
            float3 bg_color;
        };

        class LightSampler;

        class Distribution1D;

        class Texture;

        class Material;

        struct HitGroupData {
            // instance data
            BufferView<const index_t> inst_to_mesh_idx;
            BufferView<const index_t> inst_to_transform_idx;
            BufferView<const Transform> transforms;

            // mesh data
            BufferView<const MeshHandle> meshes;
            BufferView<const float3> positions;
            BufferView<const float3> normals;
            BufferView<const float2> tex_coords;
            BufferView<const TriangleHandle> triangles;

            // texture data
            BufferView<const Texture> textures;

            // material data
            BufferView<const Material> materials;

            // light data
            const LightSampler *light_sampler;
            BufferView<const Distribution1D> emission_distributions;

#ifdef IS_GPU_CODE
            NDSC_XPU_INLINE const Texture* get_texture(index_t idx) const {
                return const_cast<Texture *>(&textures[idx]);
            }
#endif
//
//            NDSC_XPU_INLINE const MeshHandle &get_mesh(index_t inst_idx) const {
//                index_t mesh_idx = inst_to_mesh_idx[inst_idx];
//                return meshes[mesh_idx];
//            }
//
//            NDSC_XPU_INLINE const Transform &get_transform(index_t inst_id) const {
//                index_t transform_idx = inst_to_transform_idx[inst_id];
//                return transforms[transform_idx];
//            }
//
//            NDSC_XPU_INLINE const Material &get_material(index_t inst_id) const {
//                auto mesh = get_mesh(inst_id);
//                return materials[mesh.material_idx];
//            }
//
//            NDSC_XPU_INLINE const Distribution1D &get_distrib(index_t inst_id) const {
//                auto mesh = get_mesh(inst_id);
//                return emission_distributions[mesh.distribute_idx];
//            }

#define GEN_GET_FUNCTION(attribute)                                                     \
            NDSC_XPU_INLINE auto get_##attribute(const MeshHandle &mesh) const {        \
                return attribute.sub_view(mesh.vertex_offset, mesh.vertex_count);       \
            }                                                                           \
            NDSC_XPU_INLINE auto get_##attribute##_by_mesh_idx(index_t mesh_idx) const {\
                MeshHandle mesh = meshes[mesh_idx];                                     \
                return get_##attribute(mesh);                                             \
            }                                                                           \
            NDSC_XPU_INLINE auto get_##attribute(index_t inst_idx) const {              \
                auto mesh_idx = inst_to_mesh_idx[inst_idx];                             \
                return get_##attribute##_by_mesh_idx(mesh_idx);                             \
            }

            GEN_GET_FUNCTION(positions)

            GEN_GET_FUNCTION(normals)

            GEN_GET_FUNCTION(tex_coords)

#undef GEN_GET_FUNCTION

        };
    }
}