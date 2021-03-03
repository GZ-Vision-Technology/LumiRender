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
        if (ret.type == "translate") {
            ret.vec4 = make_float4(param.as_float3(), 0);
        } else if (ret.type == "rotate_x") {
            auto deg = param.as_float();
            ret.vec4 = make_float4(1,0,0,deg);
        } else if (ret.type == "rotate_y") {
            auto deg = param.as_float();
            ret.vec4 = make_float4(0,1,0,deg);
        } else if (ret.type == "rotate_z") {
            auto deg = param.as_float();
            ret.vec4 = make_float4(0,0,1,deg);
        } else if (ret.type == "rotate") {
            auto vec4 = param.as_float4();
            ret.vec4 = make_float4(make_float3(vec4), vec4.w);
        } else if (ret.type == "scale") {
            auto scale = param.as_float3();
            ret.vec4 = make_float4(scale, 0.f);
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