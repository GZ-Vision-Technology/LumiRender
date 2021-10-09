//
// Created by Zero on 2021/4/12.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/geometry/common.h"
#include "render/include/interaction.h"

namespace luminous {

    inline namespace sampling {
        class Distribution1D;

        class Distribution2D;
    }

    inline namespace render {
        class LightSampler;

        class Texture;

        class Material;

        struct RayGenData {

        };

        struct SceneData {
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
            const LightSampler *light_sampler{};
            BufferView<const Distribution1D> distributions;
            BufferView<const Distribution2D> distribution2ds;

#define GEN_GET_FUNCTION(attribute)                                                     \
            ND_XPU_INLINE auto get_##attribute(const MeshHandle &mesh) const {        \
                return (attribute).sub_view(mesh.vertex_offset, mesh.vertex_count);     \
            }                                                                           \
            ND_XPU_INLINE auto get_##attribute##_by_mesh_idx(index_t mesh_idx) const {\
                MeshHandle mesh = meshes[mesh_idx];                                     \
                return get_##attribute(mesh);                                           \
            }                                                                           \
            ND_XPU_INLINE auto get_##attribute(index_t inst_idx) const {              \
                auto mesh_idx = inst_to_mesh_idx[inst_idx];                             \
                return get_##attribute##_by_mesh_idx(mesh_idx);                         \
            }

            GEN_GET_FUNCTION(positions)

            GEN_GET_FUNCTION(normals)

            GEN_GET_FUNCTION(tex_coords)

#undef GEN_GET_FUNCTION

            ND_XPU_INLINE const MeshHandle &get_mesh(index_t inst_idx) const {
                index_t mesh_idx = inst_to_mesh_idx[inst_idx];
                return meshes[mesh_idx];
            }

            ND_XPU_INLINE const Transform &get_transform(index_t inst_id) const {
                index_t transform_idx = inst_to_transform_idx[inst_id];
                return transforms[transform_idx];
            }

            ND_XPU_INLINE const TriangleHandle &get_triangle(const MeshHandle &mesh, index_t triangle_id) const {
                return triangles[mesh.triangle_offset + triangle_id];
            }

            ND_XPU_INLINE const TriangleHandle &get_triangle(const HitPoint &closest_hit) const {
                auto mesh = get_mesh(closest_hit.instance_id);
                return get_triangle(mesh, closest_hit.triangle_id);
            }

            LM_XPU SurfaceInteraction compute_surface_interaction(index_t inst_id,
                                                                  index_t tri_id,
                                                                  float2 bary) const;

            LM_NODISCARD LM_XPU_INLINE SurfaceInteraction compute_surface_interaction(const HitPoint &closest_hit) const {
                return compute_surface_interaction(closest_hit.instance_id, closest_hit.triangle_id, closest_hit.bary);
            }

            LM_XPU void fill_attribute(index_t inst_id, index_t tri_id, float2 bary,
                                       float3 *world_p, float3 *world_ng = nullptr,
                                       float3 *world_ns = nullptr, float2 *tex_coord = nullptr) const;

            LM_XPU_INLINE void fill_attribute(const HitPoint &closest_hit,
                                              float3 *world_p, float3 *world_ng = nullptr,
                                              float3 *world_ns = nullptr, float2 *tex_coord = nullptr) const {
                fill_attribute(closest_hit.instance_id, closest_hit.triangle_id, closest_hit.bary,
                               world_p, world_ng, world_ns, tex_coord);
            }

            LM_ND_XPU const Material &get_material(index_t inst_id) const;

            LM_ND_XPU const Texture &get_texture(index_t idx) const;

            LM_ND_XPU const Distribution1D &get_distrib(index_t inst_id) const;

            LM_ND_XPU const Distribution2D &get_distribution2d(index_t idx) const;
        };
    }
}