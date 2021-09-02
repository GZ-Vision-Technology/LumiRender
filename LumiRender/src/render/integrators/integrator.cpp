//
// Created by Zero on 02/09/2021.
//

#include "integrator.h"
#include "render/include/scene.h"


namespace luminous {
    inline namespace render {
        const SceneData* Integrator::scene_data() const { return _scene->scene_data();}
    }
}