//
// Created by Zero on 2021/4/12.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/geometry/common.h"
#include "render/lights/light_util.h"


namespace luminous {

    inline namespace sampling {
        struct DistribData;

        template<typename T>
        class TDistribution;

        using Distribution1D = TDistribution<DistribData>;

        struct Distribution2DData;

        template<typename T>
        class TDistribution2D;

        using Distribution2D = TDistribution2D<Distribution2DData>;
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

            ND_XPU_INLINE const TriangleHandle &get_triangle(const HitInfo &closest_hit) const {
                auto mesh = get_mesh(closest_hit.instance_id);
                return get_triangle(mesh, closest_hit.prim_id);
            }

            LM_ND_XPU float compute_prim_PMF(HitInfo hit_info) const {
                return compute_prim_PMF(hit_info.instance_id, hit_info.prim_id);
            }

            LM_ND_XPU float compute_prim_PMF(index_t inst_id, index_t tri_id) const;

            LM_ND_XPU SurfaceInteraction compute_surface_interaction(index_t inst_id,
                                                                     index_t tri_id,
                                                                     luminous::float2 bary) const;

            LM_ND_XPU SurfaceInteraction compute_surface_interaction(const HitInfo &hit_info) const {
                return compute_surface_interaction(hit_info.instance_id, hit_info.prim_id, hit_info.bary);
            }

            LM_ND_XPU bool has_emission(index_t inst_id) const {
                auto mesh = get_mesh(inst_id);
                return mesh.has_emission();
            }

            LM_ND_XPU bool has_emission(const HitInfo &hit_point) const {
                return has_emission(hit_point.instance_id);
            }

            LM_ND_XPU bool has_material(index_t inst_id) const {
                auto mesh = get_mesh(inst_id);
                return mesh.has_material();
            }

            LM_ND_XPU bool has_material(const HitInfo &hit_point) const {
                return has_material(hit_point.instance_id);
            }

            LM_ND_XPU LightEvalContext compute_light_eval_context(index_t inst_id,
                                                                  index_t tri_id,
                                                                  luminous::float2 bary) const;

            LM_ND_XPU LightEvalContext compute_light_eval_context(const HitInfo &hit_point) const {
                return compute_light_eval_context(hit_point.instance_id, hit_point.prim_id, hit_point.bary);
            }

            LM_XPU MeshHandle fill_attribute(index_t inst_id, index_t tri_id, float2 bary,
                                             float3 *world_p, float3 *world_ng_un = nullptr,
                                             float2 *tex_coord = nullptr,
                                             float3 *world_ns_un = nullptr,
                                             SurfaceInteraction *si = nullptr) const;

            LM_XPU_INLINE MeshHandle fill_attribute(const HitInfo &closest_hit,
                                                    float3 *world_p, float3 *world_ng = nullptr,
                                                    float2 *tex_coord = nullptr,
                                                    float3 *world_ns = nullptr,
                                                    SurfaceInteraction *si = nullptr) const {
                return fill_attribute(closest_hit.instance_id, closest_hit.prim_id, closest_hit.bary,
                                      world_p, world_ng, tex_coord, world_ns, si);
            }

            LM_ND_XPU const Material *get_material(index_t inst_id) const;

            LM_ND_XPU const Light *get_light(index_t inst_id) const;

            LM_ND_XPU const Texture &get_texture(index_t idx) const;

            LM_ND_XPU const Distribution1D &get_distribution(index_t inst_id) const;

            LM_ND_XPU const Distribution2D &get_distribution2d(index_t idx) const;
        };
    }
}