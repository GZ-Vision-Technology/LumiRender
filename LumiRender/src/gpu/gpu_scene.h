//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/concepts.h"
#include "gpu/accel/megakernel_optix_accel.h"
#include "render/include/scene.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene : public Scene {
        private:
            UP<MegakernelOptixAccel> _optix_accel;

            friend class MegakernelOptixAccel;
            friend class OptixAccel;
            friend class ShaderWrapper;

        public:
            GPUScene(const SP<Device> &device, Context *context);

            void init(const SP<SceneGraph> &scene_graph) override;

            void init_accel() override;

            NDSC size_t size_in_bytes() const override;

            void create_device_memory() override;

            void synchronize_to_gpu();

            template<typename... Args>
            void launch(Args &&...args) {
                _optix_accel->launch(std::forward<Args>(args)...);
            }

            void clear() override;

            void build_accel();

            NDSC std::string description() const override;
        };
    }
}