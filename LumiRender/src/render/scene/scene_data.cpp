//
// Created by Zero on 2021/5/5.
//

#include "scene_data.h"
#include "base_libs/sampling/distribution.h"
#include "render/materials/material.h"
#include "render/light_samplers/shader_include.h"
#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {

        SurfaceInteraction SceneData::compute_surface_interaction(index_t inst_id,
                                                                  index_t tri_id,
                                                                  luminous::float2 bary) const {
            auto mesh = get_mesh(inst_id);
            TriangleHandle tri = get_triangle(mesh, tri_id);
            const auto &mesh_tex_coords = get_tex_coords(mesh);
            const auto &mesh_positions = get_positions(mesh);
            const auto &mesh_normals = get_normals(mesh);
            const auto &o2w = get_transform(inst_id);

            SurfaceInteraction si;
            luminous::float2 tex_coord0 = mesh_tex_coords[tri.i];
            luminous::float2 tex_coord1 = mesh_tex_coords[tri.j];
            luminous::float2 tex_coord2 = mesh_tex_coords[tri.k];
            if (is_zero(tex_coord0) && is_zero(tex_coord1) && is_zero(tex_coord2)) {
                tex_coord0 = luminous::make_float2(0, 0);
                tex_coord1 = luminous::make_float2(1, 0);
                tex_coord2 = luminous::make_float2(1, 1);
            }
            si.uv = triangle_lerp(bary, tex_coord0, tex_coord1, tex_coord2);

            {
                // compute pos
                luminous::float3 p0 = o2w.apply_point(mesh_positions[tri.i]);
                luminous::float3 p1 = o2w.apply_point(mesh_positions[tri.j]);
                luminous::float3 p2 = o2w.apply_point(mesh_positions[tri.k]);
                luminous::float3 pos = triangle_lerp(bary, p0, p1, p2);
                si.pos = pos;

                // compute geometry uvn
                luminous::float3 dp02 = p0 - p2;
                luminous::float3 dp12 = p1 - p2;
                luminous::float3 ng = cross(dp02, dp12);
                si.prim_area = 0.5f * length(ng);

                luminous::float2 duv02 = tex_coord0 - tex_coord2;
                luminous::float2 duv12 = tex_coord1 - tex_coord2;
                float det = duv02[0] * duv12[1] - duv02[1] * duv12[0];
                float inv_det = 1 / det;

                luminous::float3 dp_du = (duv12[1] * dp02 - duv02[1] * dp12) * inv_det;
                luminous::float3 dp_dv = (-duv12[0] * dp02 + duv02[0] * dp12) * inv_det;
                si.g_uvn.set(normalize(dp_du), normalize(dp_dv), normalize(ng));
            }

            {
                // compute shading uvn
                auto n0 = mesh_normals[tri.i];
                auto n1 = mesh_normals[tri.j];
                auto n2 = mesh_normals[tri.k];
                auto normal = triangle_lerp(bary, n0, n1, n2);
                luminous::float3 ns = normalize(o2w.apply_normal(normal));
                luminous::float3 ss = si.g_uvn.dp_du;
                luminous::float3 st = normalize(cross(ns, ss));
                ss = cross(st, ns);
                si.s_uvn.set(ss, st, ns);
            }
            if (mesh.has_emission()) {
                si.light = &light_sampler->light_at(mesh.light_idx);
            }
            if (mesh.has_material()) {
                si.material = &materials[mesh.material_idx];
            }
            return si;
        }

        LightEvalContext SceneData::compute_light_eval_context(index_t inst_id,
                                                               index_t tri_id,
                                                               luminous::float2 bary) const {
            float3 pos, ng;
            float2 uv;
            fill_attribute(inst_id, tri_id, bary, &pos, &ng, &uv);
            float prim_area = 0.5f * length(ng);
            float PMF = compute_prim_PMF(inst_id, tri_id);
            float PDF_pos = PMF / prim_area;
            return LightEvalContext{pos, normalize(ng), uv, PDF_pos};
        }

        void SceneData::fill_attribute(index_t inst_id, index_t tri_id, float2 bary,
                                       float3 *world_p, float3 *world_ng,float2 *tex_coord,
                                       float3 *world_ns ) const {
            auto mesh = get_mesh(inst_id);
            TriangleHandle tri = get_triangle(mesh, tri_id);

            const auto &o2w = get_transform(inst_id);
            // compute pos
            const auto &mesh_positions = get_positions(mesh);
            luminous::float3 p0 = mesh_positions[tri.i];
            luminous::float3 p1 = mesh_positions[tri.j];
            luminous::float3 p2 = mesh_positions[tri.k];
            *world_p = o2w.apply_point(triangle_lerp(bary, p0, p1, p2));

            if (world_ng) {
                luminous::float3 dp02 = p0 - p2;
                luminous::float3 dp12 = p1 - p2;
                *world_ng = o2w.apply_normal(cross(dp02, dp12));
            }

            if (world_ns) {
                const auto &mesh_normals = get_normals(mesh);
                auto n0 = mesh_normals[tri.i];
                auto n1 = mesh_normals[tri.j];
                auto n2 = mesh_normals[tri.k];
                *world_ns = o2w.apply_normal(triangle_lerp(bary, n0, n1, n2));
            }

            if (tex_coord) {
                const auto &mesh_tex_coords = get_tex_coords(mesh);
                luminous::float2 tex_coord0 = mesh_tex_coords[tri.i];
                luminous::float2 tex_coord1 = mesh_tex_coords[tri.j];
                luminous::float2 tex_coord2 = mesh_tex_coords[tri.k];
                if (is_zero(tex_coord0) && is_zero(tex_coord1) && is_zero(tex_coord2)) {
                    tex_coord0 = luminous::make_float2(0, 0);
                    tex_coord1 = luminous::make_float2(1, 0);
                    tex_coord2 = luminous::make_float2(1, 1);
                }
                *tex_coord = triangle_lerp(bary, tex_coord0, tex_coord1, tex_coord2);
            }
        }

        const Distribution1D &SceneData::get_distrib(index_t inst_id) const {
            auto mesh = get_mesh(inst_id);
            return distributions[mesh.distribute_idx];
        }

        const Distribution2D &SceneData::get_distribution2d(index_t idx) const {
            return distribution2ds[idx];
        }

        const Texture &SceneData::get_texture(index_t idx) const {
            return textures[idx];
        }

        const Material *SceneData::get_material(index_t inst_id) const {
            auto mesh = get_mesh(inst_id);
            return &materials[mesh.material_idx];
        }

        float SceneData::compute_prim_PMF(index_t inst_id, index_t tri_id) const {
            auto mesh = get_mesh(inst_id);
            const Distribution1D &distrib = get_distrib(mesh.distribute_idx);
            return distrib.PMF(tri_id);
        }

        const Light *SceneData::get_light(index_t inst_id) const {
            auto mesh = get_mesh(inst_id);
            return &light_sampler->light_at(mesh.light_idx);
        }
    } // luminous::render
} // luminous
