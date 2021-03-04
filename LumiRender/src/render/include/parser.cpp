//
// Created by Zero on 2021/2/16.
//

#include "parser.h"
#include "parameter_set.h"
#include "graphics/string_util.h"
#include <iomanip>

namespace luminous {
    inline namespace render {
        DataWrap create_json_from_file(const std::filesystem::path &fn) {
            std::ifstream fst;
            fst.open(fn.c_str());
            std::stringstream buffer;
            buffer << fst.rdbuf();
            std::string str = buffer.str();
            str = jsonc_to_json(str);
//        LUMINOUS_INFO(str);
            if (fn.extension() == "bson") {
                return DataWrap::from_bson(str);
            } else {
                return DataWrap::parse(str);
            }
        }
    }

    TransformConfig parse_transform(const ParameterSet &ps) {
        TransformConfig ret;
        ret.type = ps["type"].as_string();
        auto param = ps["param"];
        if (ret.type == "matrix4x4") {
            ret.mat4x4 = param["matrix4x4"].as_float4x4();
            cout << ret.mat4x4.to_string() << endl;
        } else if (ret.type == "trs") {
            ret.t = param["t"].as_float3();
            ret.r = param["r"].as_float4();
            ret.s = param["s"].as_float3();
        }
        return ret;
    }

    UP<SceneGraph> Parser::load_from_json(const std::filesystem::path &fn) {
        _data = create_json_from_file(fn);
        auto shapes = _data["shapes"];
        using namespace std;
        auto scene_graph = make_unique<SceneGraph>(_context);
        scene_graph->shape_configs.reserve(shapes.size());
        for (auto &shape : shapes) {
            ShapeConfig shape_config;
            shape_config.type = shape["type"];
            shape_config.fn = string(shape["params"]["fn"]);
            shape_config.o2w = parse_transform(ParameterSet(shape["params"]["transform"]));
        }

        return move(scene_graph);
    }
}