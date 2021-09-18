//
// Created by Zero on 30/08/2021.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {

        class Scene;

        class Accelerator : public Noncopyable {
        protected:
            size_t _bvh_size_in_bytes{0u};
            const Scene *_scene{};
        public:
            explicit Accelerator(const Scene *scene) : _scene(scene) {}

            virtual ~Accelerator() = default;

            _NODISCARD virtual uint64_t handle() const = 0;

            _NODISCARD size_t bvh_size_in_bytes() const { return _bvh_size_in_bytes; }

            virtual void clear() = 0;

            _NODISCARD virtual  std::string description() const = 0;

            virtual void build_bvh(const Managed <float3> &positions, const Managed <TriangleHandle> &triangles,
                                   const Managed <MeshHandle> &meshes, const Managed <uint> &instance_list,
                                   const Managed <Transform> &transform_list,
                                   const Managed <uint> &inst_to_transform) = 0;
        };
    }
}