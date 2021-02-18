//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "backend/buffer.h"
#include "concepts.h"
#include "graphics/math/common.h"
#include "model.h"

namespace luminous {
    struct Scene : public Noncopyable {
        Buffer<float3> _positions;
        Buffer<float3> _normals;
        Buffer<float2> _tex_coords;
        Buffer<TriangleHandle> _triangles;
        Buffer<ModelHandle> _models;
        Buffer<uint> _models_triangle_counts;
        Buffer<uint> _instance_to_model_id;
        Buffer<float4x4> _instance_transforms;
    };
}