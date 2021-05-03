//
// Created by Zero on 2021/2/16.
//

#include "parser.h"
#include "parameter_set.h"
#include "render/textures/texture.h"
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
            LUMINOUS_DEBUG(str);
            if (fn.extension() == "bson") {
                return DataWrap::from_bson(str);
            } else {
                return DataWrap::parse(str);
            }
        }

        //"transform" : {
        //					"type" : "matrix4x4",
        //					"param" : {
        //						"matrix4x4" : [
        //							1,0,0,0,
        //							0,1,0,0,
        //							0,0,1,0,
        //							0,0,0,1
        //						]
        //					}
        //				}
        TransformConfig parse_transform(const ParameterSet &ps) {
            TransformConfig ret;
            ret.set_type(ps["type"].as_string("matrix4x4"));
            auto param = ps["param"];
            if (ret.type() == "matrix4x4") {
                ret.mat4x4 = param["matrix4x4"].as_float4x4();
            } else if (ret.type() == "trs") {
                ret.t = param["t"].as_float3();
                ret.r = param["r"].as_float4(make_float4(1, 0, 0, 0));
                ret.s = param["s"].as_float3(make_float3(1, 1, 1));
            } else {
                // yaw pitch position
                ret.yaw = param["yaw"].as_float();
                ret.pitch = param["pitch"].as_float();
                ret.position = param["position"].as_float3();
            }
            return ret;
        }

        ShapeConfig parse_shape(const DataWrap &shape) {
            ShapeConfig shape_config;
            shape_config.set_type(string(shape["type"]));
            shape_config.name = shape["name"];
            ParameterSet param(shape["param"]);
            if (shape_config.type() == "model") {
                shape_config.subdiv_level = param["subdiv_level"].as_uint(0u);
                shape_config.fn = param["fn"].as_string();
                shape_config.smooth = param["smooth"].as_bool(true);
            } else if (shape_config.type() == "quad") {
                shape_config.width = param["width"].as_float(1);
                shape_config.height = param["height"].as_float(1);
            }
            shape_config.material_name = param["material"].as_string();
            shape_config.name = string(shape["name"]);
            shape_config.o2w = parse_transform(param["transform"]);
            if (param.contains("emission")) {
                shape_config.emission = param["emission"].as_float3(make_float3(0.f));
            }
            return shape_config;
        }

        //		{
        //			"name" : "c_box",
        //			"type": "model", or "quad"
        //			"params" : {
        //				"fn": "cornell_box.obj",
        //				"transform" : {
        //					"type" : "trs",
        //					"param": {
        //						"t": [1,1,1],
        //						"r": [1,1,1,60],
        //						"s": [2,2,2]
        //					}
        //				}
        //			}
        //		}
        std::vector<ShapeConfig> parse_shapes(const DataWrap &shapes) {
            std::vector<ShapeConfig> ret;
            ret.reserve(shapes.size());
            for (auto &shape : shapes) {
                ShapeConfig shape_config = parse_shape(shape);
                ret.push_back(shape_config);
            }
            return move(ret);
        }

        //"sampler" : {
        //		"type" : "LCGSampler",
        //		"param" : {
        //			"spp" : 16
        //		}
        //	}
        SamplerConfig parse_sampler(const ParameterSet &ps) {
            SamplerConfig ret;
            ret.set_full_type(ps["type"].as_string());
            ret.spp = ps["param"]["spp"].as_uint();
            return ret;
        }

        FilmConfig parse_film(const ParameterSet &ps) {
            FilmConfig fc;
            fc.set_full_type("RGBFilm");
            ParameterSet param(ps["param"]);
            fc.resolution = param["resolution"].as_uint2(make_uint2(500, 500));
            fc.file_name = param["file_name"].as_string("luminous.png");
            return fc;
        }

        //	"camera" : {
        //		"type" : "PinholeCamera",
        //		"param" : {
        //			"fov_y" : 20,
        //			"velocity" : 20,
        //			"transform" : {
        //				"type" : "yaw_pitch",
        //				"param" : {
        //					"yaw" : 10,
        //					"pitch": 20,
        //					"position": [1,1,1]
        //				}
        //			}
        //		}
        //	},
        SensorConfig parse_sensor(const ParameterSet &ps) {
            SensorConfig ret;
            ret.set_full_type(ps["type"].as_string());
            ParameterSet param(ps["param"]);
            ret.fov_y = param["fov_y"].as_float();
            ret.velocity = param["velocity"].as_float();
            ret.transform_config = parse_transform(param["transform"]);
            ret.film_config = parse_film(param["film"]);
            return ret;
        }

        //    {
        //        "type": "PointLight",
        //            "param": {
        //            "pos": [10,10,10],
        //            "intensity": [10,1,6]
        //        }
        //    }
        LightConfig parse_light(const ParameterSet &ps) {
            LightConfig ret;
            ret.set_full_type(ps["type"].as_string("PointLight"));
            ParameterSet param = ps["param"];
            ret.position = param["pos"].as_float3(make_float3(0.f));
            ret.intensity = param["intensity"].as_float3(make_float3(0.f));
            return ret;
        }

        std::vector<LightConfig> parse_lights(const DataWrap &lights) {
            std::vector<LightConfig> ret;
            if (!lights.is_array()) {
                return ret;
            }
            ret.reserve(lights.size());
            for (const auto &light : lights) {
                LightConfig lc = parse_light(ParameterSet(light));
                ret.push_back(lc);
            }
            return ret;
        }

        LightSamplerConfig parse_light_sampler(const ParameterSet &ps) {
            LightSamplerConfig ret;
            ret.set_full_type(ps["type"].as_string("UniformLightSampler"));
            return ret;
        }

        void Parser::load_from_json(const std::filesystem::path &fn) {
            _data = create_json_from_file(fn);
        }

        TextureConfig parse_texture(const ParameterSet &ps) {
            std::string type;
            type = ps["type"].as_string("ConstantTexture");
            TextureConfig tc;
            auto param = ps["param"];
            tc.set_full_type(type);
            if (type == "ConstantTexture") {
                tc.val = param["val"].as_float4(make_float4(1.f));
            } else {
                tc.fn = param["fn"].as_string();
            }
            tc.name = ps["name"].as_string();
            string color_space = param["color_space"].as_string("SRGB");
            if (color_space == "SRGB") {
                tc.color_space = SRGB;
            } else {
                tc.color_space = LINEAR;
            }
            return tc;
        }

        std::vector<TextureConfig> parse_textures(const DataWrap &textures) {
            std::vector<TextureConfig> ret;
            for (const auto &texture : textures) {
                ret.push_back(parse_texture(ParameterSet(texture)));
            }
            return ret;
        }

        MaterialConfig parse_material(const ParameterSet &ps) {
            std::string type;
            type = ps["type"].as_string("MatteMaterial");
            MaterialConfig ret;
            ret.set_full_type(type);
            if (type == "MatteMaterial") {
                ret.diffuse_tex.name = ps["param"]["diffuse"].as_string();
            }
            ret.name = ps["name"].as_string();
            return ret;
        }

        std::vector<MaterialConfig> parse_materials(const DataWrap &materials) {
            std::vector<MaterialConfig> ret;
            for (const auto &material : materials) {
                ret.push_back(parse_material(ParameterSet(material)));
            }
            return ret;
        }

        SP<SceneGraph> Parser::parse() const {
            auto shapes = _data["shapes"];
            auto scene_graph = make_shared<SceneGraph>(_context);
            scene_graph->shape_configs = parse_shapes(shapes);
            scene_graph->sensor_config = parse_sensor(ParameterSet(_data["camera"]));
            scene_graph->sampler_config = parse_sampler(ParameterSet(_data["sampler"]));
            scene_graph->light_configs = parse_lights(_data.value("lights", DataWrap()));
            scene_graph->light_sampler_config = parse_light_sampler(ParameterSet(_data["light_sampler"]));
            scene_graph->tex_configs = parse_textures(_data["textures"]);
            scene_graph->material_configs = parse_materials(_data["materials"]);
            return scene_graph;
        }
    }
}