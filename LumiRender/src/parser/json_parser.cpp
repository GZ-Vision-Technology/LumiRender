//
// Created by Zero on 2021/2/16.
//

#include "json_parser.h"
#include "parameter_set.h"
#include <iomanip>

namespace luminous {
    inline namespace utility {
        DataWrap create_json_from_file(const luminous_fs::path &fn) {
            std::ifstream fst;
            fst.open(fn.c_str());
            std::stringstream buffer;
            buffer << fst.rdbuf();
            std::string str = buffer.str();
            str = jsonc_to_json(str);
            LUMINOUS_DEBUG(str);
            if (fn.extension() == ".bson") {
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
                shape_config.swap_handed = param["swap_handed"].as_bool(false);
            } else if (shape_config.type() == "quad") {
                shape_config.width = param["width"].as_float(1);
                shape_config.height = param["height"].as_float(1);
            } else if (shape_config.type() == "mesh") {
                shape_config.positions = param["positions"].as_vector<float3>();
                shape_config.normals = param["normals"].as_vector<float3>();
                shape_config.tex_coords = param["tex_coords"].as_vector<float2>();
                shape_config.triangles = param["triangles"].as_vector<TriangleHandle>();
            }

            shape_config.name = string(shape["name"]);
            shape_config.o2w = parse_transform(param["transform"]);
            if (param.contains("emission")) {
                auto scale = param["scale"].as_float(1);
                shape_config.emission = param["emission"].as_float3(make_float3(0.f)) * scale;
            }
            if (is_zero(shape_config.emission)) {
                shape_config.material_name = param["material"].as_string();
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
            ParameterSet param(ps["param"]);
            fc.resolution = param["resolution"].as_uint2(make_uint2(500, 500));
            fc.state = param["fb_state"].as_int(0);
            return fc;
        }

        //"filter": {
        //    "type": "BoxFilter",
        //    "param": {
        //        "radius": [1,1]
        //    }
        //}
        FilterConfig parse_filter(const ParameterSet &ps) {
            FilterConfig fc;
            std::string type = ps["type"].as_string("BoxFilter");
            fc.set_full_type(type);
            ParameterSet param(ps["param"]);
            if (type == "BoxFilter") {
                fc.radius = param["radius"].as_float2(make_float2(0.5, 0.5));
            } else if (type == "TriangleFilter") {
                fc.radius = param["radius"].as_float2(make_float2(2.f, 2.f));
            } else if (type == "GaussianFilter") {
                fc.radius = param["radius"].as_float2(make_float2(1.5f, 1.5f));
                fc.sigma = param["radius"].as_float(0.5f);
            }
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
            auto type = ps["type"].as_string();
            ret.set_full_type(type);
            ParameterSet param(ps["param"]);
            ret.fov_y = param["fov_y"].as_float();
            ret.velocity = param["velocity"].as_float();
            ret.transform_config = parse_transform(param["transform"]);
            if (type == "ThinLensCamera") {
                ret.focal_distance = param["focal_distance"].as_float(0);
                ret.lens_radius = param["lens_radius"].as_float(0);
            }
            ret.film_config = parse_film(param["film"]);
            ret.filter_config = parse_filter(param["filter"]);
            return ret;
        }

        //    {
        //        "type": "PointLight",
        //         "param": {
        //            "pos": [10,10,10],
        //            "intensity": [10,1,6]
        //        }
        //    }
        LightConfig parse_light(const ParameterSet &ps) {
            LightConfig ret;
            std::string type = ps["type"].as_string("PointLight");
            ret.set_full_type(type);
            ParameterSet param = ps["param"];
            ret.miss_color = param["miss_color"].as_float3(make_float3(0.f));
            if (type == "PointLight") {
                ret.position = param["pos"].as_float3(make_float3(0.f));
                ret.intensity = param["intensity"].as_float3(make_float3(0.f));
            } else if (type == "Envmap") {
                ret.texture_config.name = param["key"].as_string("");
                ret.scale = param["scale"].as_float3(make_float3(1.f));
                ret.o2w_config = parse_transform(param["transform"]);
            } else if (type == "SpotLight") {
                ret.position = param["pos"].as_float3(make_float3(0.f));
                ret.intensity = param["intensity"].as_float3(make_float3(0.f));
                ret.theta_o = param["theta_o"].as_float(60);
                ret.theta_i = param["theta_i"].as_float(45);
            }
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

        void JsonParser::load(const luminous_fs::path &fn) {
            auto a = fn.extension();
            if (fn.extension() == ".json" || fn.extension() == ".bson") {
                _data = create_json_from_file(fn);
            }
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
            tc.scale = param["scale"].as_float3(make_float3(1.f));
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
                TextureConfig config = parse_texture(ParameterSet(texture));
                config.tex_idx = ret.size();
                ret.push_back(config);
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

        IntegratorConfig parse_integrator(const ParameterSet &ps) {
            IntegratorConfig ret;
            auto param = ps["param"];
            ret.set_type(ps["type"].as_string("PT"));
            ret.max_depth = param["max_depth"].as_uint(10);
            ret.rr_threshold = param["rr_threshold"].as_float(1);
            return ret;
        }

        OutputConfig parse_output(const ParameterSet &ps) {
            OutputConfig ret;
            ret.fn = ps["fn"].as_string("luminous.png");
            ret.frame_num = ps["frame_num"].as_int(0);
            return ret;
        }

        SP<SceneGraph> JsonParser::parse() const {
            auto shapes = _data["shapes"];
            auto scene_graph = std::make_shared<SceneGraph>(_context);
            scene_graph->shape_configs = parse_shapes(shapes);
            scene_graph->sensor_config = parse_sensor(ParameterSet(_data["camera"]));
            scene_graph->sampler_config = parse_sampler(ParameterSet(_data["sampler"]));
            scene_graph->light_configs = parse_lights(_data.value("lights", DataWrap()));
            scene_graph->light_sampler_config = parse_light_sampler(ParameterSet(_data["light_sampler"]));
            scene_graph->tex_configs = parse_textures(_data["textures"]);
            scene_graph->material_configs = parse_materials(_data["materials"]);
            scene_graph->integrator_config = parse_integrator(ParameterSet(_data["integrator"]));
            scene_graph->output_config = parse_output(ParameterSet(_data["output"]));
            scene_graph->create_shapes();
            return scene_graph;
        }
    }
}