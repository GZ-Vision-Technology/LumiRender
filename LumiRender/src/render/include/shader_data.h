//
// Created by Zero on 2021/4/12.
//


#pragma once

#include "graphics/math/common.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {

        struct RayGenData {

        };


        struct MissData {
            float3 bg_color;
        };

        class LightSampler;

        struct HitGroupData {
            // instance data
            BufferView<const uint> inst_to_mesh_idx;
            BufferView<const uint> inst_to_transform_idx;
            BufferView<const float4x4> transforms;
            // mesh data
            BufferView<const MeshHandle> meshes;
            BufferView<const float3> positions;
            BufferView<const float3> normals;
            BufferView<const float2> tex_coords;
            BufferView<const TriangleHandle> triangles;

            // light data
            const LightSampler *light_sampler;
        };
    }
}