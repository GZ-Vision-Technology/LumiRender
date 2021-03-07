//
// Created by Zero on 2021/2/18.
//

#include "cuda_pipeline.h"

namespace luminous {
    inline namespace gpu {

        void CUDAPipeline::init(const Parser &parser) {
            auto scene_graph = parser.parse();
            scene_graph->create_shapes();
            _scene = make_unique<Scene>(_device);
            _scene->convert_geometry_data(scene_graph);
        }
    }
}