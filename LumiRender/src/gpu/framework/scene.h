//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/backend/buffer.h"
#include "core/concepts.h"
#include "graphics/math/common.h"
#include "render/include/sensor.h"

namespace luminous {
    inline namespace gpu {
        class Scene : public Noncopyable {
        private:
            Buffer<float3> _positions{nullptr};
            Buffer<float3> _normals{nullptr};
            Buffer<float2> _tex_coords{nullptr};
            Buffer<TriangleHandle> _triangles{nullptr};
            Buffer<ModelHandle> _models{nullptr};
            Buffer<uint> _models_triangle_counts{nullptr};
            Buffer<uint> _instance_to_model_id{nullptr};
            Buffer<float4x4> _instance_transforms{nullptr};
        public:
            Scene(UP<SceneGraph> scene_graph);

        };
    }
}