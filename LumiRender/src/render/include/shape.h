//
// Created by Zero on 2021/1/30.
//


#pragma once

#include "graphics/geometry/common.h"
#include <vector>
#include "core/concepts.h"
#include "interaction.h"
#include "config.h"


namespace luminous {

    inline namespace render {
        using std::vector;
        using std::string;

        class Shape : public IObject {

        };

        struct Mesh : public Shape {
            Mesh(vector <float3> P,
                 vector <float3> N,
                 vector <float2> uv,
                 vector <TriangleHandle> T,
                 Box3f aabb) :
                    positions(std::move(P)),
                    normals(std::move(N)),
                    tex_coords(std::move(uv)),
                    triangles(std::move(T)),
                    aabb(aabb) {}

            vector <float3> normals;
            vector <float3> positions;
            vector <float2> tex_coords;
            vector <TriangleHandle> triangles;
            Box3f aabb;
            mutable uint idx_in_meshes;
        };

        struct Model {
            Model(const std::filesystem::path &fn, uint subdiv_level = 0);

            Model() = default;

            vector <std::shared_ptr<const Mesh>> meshes;
            vector <MaterialConfig> materials;
            string key;
        };

        struct ModelInstance {
            ModelInstance(uint idx, const Transform &t, const char *n,
                          float3 emission = make_float3(0))
                    : model_idx(idx),
                      o2w(t),
                      name(n),
                      emission(emission) {}

            const char *name;
            const uint model_idx;
            const Transform o2w;
            float3 emission = make_float3(0.f);
        };
    }
}