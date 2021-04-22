//
// Created by Zero on 2021/2/16.
//

#include "parser.h"
#include "parameter_set.h"
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
            ret.type = ps["type"].as_string("matrix4x4");
            auto param = ps["param"];
            if (ret.type == "matrix4x4") {
                ret.mat4x4 = param["matrix4x4"].as_float4x4();
            } else if (ret.type == "trs") {
                ret.t = param["t"].as_float3();
                ret.r = param["r"].as_float4();
                ret.s = param["s"].as_float3();
            } else {
                // yaw pitch position
                ret.yaw = param["yaw"].as_float();
                ret.pitch = param["pitch"].as_float();
                ret.position = param["position"].as_float3();
            }
            return ret;
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
                ShapeConfig shape_config;
                shape_config.type = string(shape["type"]);
                shape_config.name = shape["name"];
                ParameterSet param(shape["param"]);
                if (shape_config.type == "model"){
                    shape_config.subdiv_level = param["subdiv_level"].as_uint(0u);
                    shape_config.fn = param["fn"].as_string();
                } else if (shape_config.type == "quad") {
                    shape_config.width = param["width"].as_float(1);
                    shape_config.height = param["height"].as_float(1);
                }
                shape_config.name = string(shape["name"]);
                shape_config.o2w = parse_transform(param["transform"]);
                if (param.contains("emission")) {
                    shape_config.emission = param["emission"].as_float3(make_float3(0.f));
                }
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
            ret.type = ps["type"].as_string();
            ret.spp = ps["param"]["spp"].as_uint();
            return ret;
        }

        FilmConfig parse_film(const ParameterSet &ps) {
            FilmConfig fc;
            fc.type = "RGBFilm";
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
            ret.type = ps["type"].as_string();
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
            ret.type = ps["type"].as_string("PointLight");
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
            ret.type = ps["type"].as_string("UniformLightSampler");
            return ret;
        }

        void Parser::load_from_json(const std::filesystem::path &fn) {
            _data = create_json_from_file(fn);
        }

        template<typename T>
        TextureConfig<T> parse_texture(const ParameterSet &ps) {
            std::string type;
            type = ps["type"].as_string("ConstantTexture");
            TextureConfig<T> tc;
            tc.type = type;
            auto param = ps["param"];
            if (type == "ConstantTexture") {
                tc.val = param["val"].template as<T>();
            } else {
                tc.fn = param["fn"].as_string();
            }
            string color_space = param["color_space"].as_string("SRGB");
            if (color_space == "SRGB") {
                tc.color_space = SRGB;
            } else {
                tc.color_space = LINEAR;
            }
            if constexpr (std::is_same_v<T, float>) {
                tc.type = tc.type + "<float>";
            } else {
                tc.type = tc.type + "<float4>";
            }
            return tc;
        }

        template<typename T>
        std::vector<TextureConfig<T>> parse_textures(const DataWrap &textures) {
            std::vector<TextureConfig<T>> ret;
            for (const auto &texture : textures) {
                ret.push_back(parse_texture<T>(ParameterSet(texture)));
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
            scene_graph->tex_scalar_configs = parse_textures<float>(_data["tex_scalars"]);
            scene_graph->tex_vector_configs = parse_textures<float4>(_data["tex_vectors"]);
            return scene_graph;
        }
    }
}