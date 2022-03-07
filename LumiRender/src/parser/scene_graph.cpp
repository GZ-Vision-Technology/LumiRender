//
// Created by Zero on 2021/2/21.
//

#include "scene_graph.h"
#include "util/stats.h"

namespace luminous {
    inline namespace render {

        string gen_key(const string &fn, uint subdiv_level = 0) {
            return string_printf("%s_subdiv_%u", fn.c_str(), subdiv_level);
        }

        string gen_key(const ShapeConfig &sc) {
            if (sc.type() == "model") {
                return string_printf("%s_subdiv_%u", sc.fn.c_str(), sc.subdiv_level);
            } else {
                return sc.name;
            }
        }


        Model SceneGraph::create_sphere(const ShapeConfig &config) {
            float radius = config.radius;
            vector<float3> positions;
            vector<float3> normals;
            vector<float2> tex_coords;
            vector<TriangleHandle> triangles;
            Box3f aabb(make_float3(-radius), float3(radius));
            uint theta_div = config.sub_div;
            uint phi_div = 2 * theta_div;
            positions.push_back(make_float3(0, radius, 0));
            normals.push_back(make_float3(0, 1, 0));
            tex_coords.push_back(make_float2(0, 0));
            for (uint i = 1; i < theta_div; ++i) {
                float v = float(i) / theta_div;
                float theta = Pi * v;
                float y = radius * cos(theta);
                float r = radius * sin(theta);
                float3 p0 = make_float3(r, y, 0.f);
                positions.push_back(p0);
                float2 t0 = make_float2(0, v);
                for (uint j = 1; j < phi_div; ++j) {
                    float u = float(j) / phi_div;
                    float phi = u * _2Pi;
                    float x = cos(phi) * r;
                    float z = sin(phi) * r;
                    float3 p = make_float3(x, y, z);
                    positions.push_back(p);
                    float2 t = make_float2(u, v);
                    tex_coords.push_back(t);
                    normals.push_back(normalize(p));
                }
            }
            positions.push_back(make_float3(0, -radius, 0));
            normals.push_back(make_float3(0, -1, 0));
            tex_coords.push_back(make_float2(0, 1));
            Model model;
            model.custom_material_name = config.material_name;
            uint tri_count = phi_div * 2 + (theta_div - 2) * phi_div * 2;
            triangles.reserve(tri_count);

            for (uint i = 0; i < phi_div; ++i) {
                TriangleHandle tri{0, (i + 1) % phi_div + 1, i + 1};
                triangles.push_back(tri);
            }
            for (uint i = 0; i < theta_div - 2; ++i) {
                uint vert_start = 1 + i * phi_div;
                for (int j = 0; j < phi_div; ++j) {
                    if (j != phi_div - 1) {
                        TriangleHandle tri{vert_start, vert_start + 1, vert_start + phi_div};
                        triangles.push_back(tri);
                        TriangleHandle tri2{vert_start + 1, vert_start + phi_div + 1, vert_start + phi_div};
                        triangles.push_back(tri2);
                    } else {
                        TriangleHandle tri{vert_start, vert_start + 1 - phi_div, vert_start + phi_div};
                        triangles.push_back(tri);
                        TriangleHandle tri2{vert_start + 1 - phi_div, vert_start + 1, vert_start + phi_div};
                        triangles.push_back(tri2);
                    }
                    vert_start ++;
                }
            }
            uint vert_start = (theta_div - 1) * phi_div + 2;
            uint vert_end = positions.size() - 1;
            for (uint i = 0; i < phi_div; ++i) {
                uint idx1 = i + 1;
                uint idx2 = (1 + i) % phi_div + 1;
                TriangleHandle tri{vert_end, vert_end - idx2, vert_end - idx1};
                triangles.push_back(tri);
            }

            auto mesh = Mesh(move(positions), move(normals), move(tex_coords), move(triangles), Box3f());
            model.meshes.push_back(mesh);
            return model;
        }

        Model SceneGraph::create_quad_y(const ShapeConfig &config) {
            auto model = Model();
            auto width = config.width / 2;
            auto height = config.height / 2;
            Box3f aabb;
            vector<float3> P{make_float3(width, 0, height),
                             make_float3(width, 0, -height),
                             make_float3(-width, 0, height),
                             make_float3(-width, 0, -height)};

            for (auto p : P) {
                aabb.extend(p);
            }

            vector<float3> N(4, make_float3(0, 0, 0));

            vector<float2> UV{make_float2(1, 1),
                              make_float2(1, 0),
                              make_float2(0, 1),
                              make_float2(0, 0)};

            vector<TriangleHandle> triangles{TriangleHandle{1, 0, 2},
                                             TriangleHandle{1, 2, 3}};

            auto mesh = Mesh(move(P), move(N), move(UV), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            return model;
        }

        Model SceneGraph::create_quad(const ShapeConfig &config) {
            auto model = Model();
            auto width = config.width / 2;
            auto height = config.height / 2;
            Box3f aabb;
            vector<float3> P{make_float3(width, height, 0),
                             make_float3(width, -height, 0),
                             make_float3(-width, height, 0),
                             make_float3(-width, -height, 0)};

            for (auto p : P) {
                aabb.extend(p);
            }

            vector<float3> N(4, make_float3(0, 0, 0));

            vector<float2> UV{make_float2(1, 1),
                              make_float2(1, 0),
                              make_float2(0, 1),
                              make_float2(0, 0)};

            vector<TriangleHandle> triangles{TriangleHandle{1, 0, 2},
                                             TriangleHandle{1, 2, 3}};

            auto mesh = Mesh(move(P), move(N), move(UV), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            return model;
        }

        Model SceneGraph::create_cube(const ShapeConfig &config) {
            float x = config.x;
            float y = config.y;
            float z = config.z;
            y = y == 0 ? x : y;
            z = z == 0 ? y : z;
            x = x / 2.f;
            y = y / 2.f;
            z = z / 2.f;
            auto P = vector<float3>{
                    float3(-x, -y, z), float3(x, -y, z), float3(-x, y, z), float3(x, y, z), // +z
                    float3(-x, y, -z), float3(x, y, -z), float3(-x, -y, -z), float3(x, -y, -z), // -z
                    float3(-x, y, z), float3(x, y, z), float3(-x, y, -z), float3(x, y, -z),  // +y
                    float3(-x, -y, z), float3(x, -y, z), float3(-x, -y, -z), float3(x, -y, -z), // -y
                    float3(x, -y, z), float3(x, y, z), float3(x, y, -z), float3(x, -y, -z), // +x
                    float3(-x, -y, z), float3(-x, y, z), float3(-x, y, -z), float3(-x, -y, -z), // -x
            };
            auto N = vector<float3>{
                    float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1),
                    float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1), float3(0, 0, -1),
                    float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0), float3(0, 1, 0),
                    float3(0, -1, 0), float3(0, -1, 0), float3(0, -1, 0), float3(0, -1, 0),
                    float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0),
                    float3(-1, 0, 0), float3(-1, 0, 0), float3(-1, 0, 0), float3(-1, 0, 0),
            };
            auto UVs = vector<float2>{
                    float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1),
                    float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0),
                    float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0),
                    float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0),
                    float2(0, 1), float2(1, 1), float2(1, 0), float2(0, 0),
                    float2(0, 1), float2(1, 1), float2(1, 0), float2(0, 0),
            };
            auto triangles = vector<TriangleHandle>{
                    TriangleHandle(0, 1, 3), TriangleHandle(0, 3, 2),
                    TriangleHandle(6, 5, 7), TriangleHandle(4, 5, 6),
                    TriangleHandle(10, 9, 11), TriangleHandle(8, 9, 10),
                    TriangleHandle(13, 14, 15), TriangleHandle(13, 12, 14),
                    TriangleHandle(18, 17, 19), TriangleHandle(17, 16, 19),
                    TriangleHandle(21, 22, 23), TriangleHandle(20, 21, 23),
            };
            Model model;
            Box3f aabb;
            for (auto p : P) {
                aabb.extend(p);
            }
            auto mesh = Mesh(move(P), move(N), move(UVs), move(triangles), aabb);
            model.meshes.push_back(mesh);
            model.custom_material_name = config.material_name;
            return model;
        }

        void swap_handed(Model &model, int dim = 0) {
            for (auto &mesh : model.meshes) {
                for (auto &pos : mesh.positions) {
                    pos[dim] *= -1.f;
                }
                for (auto &normal : mesh.normals) {
                    normal[dim] *= -1.f;
                }
                for (auto &tri : mesh.triangles) {
                    std::swap(tri.i, tri.j);
                }
            }
        }

        Model SceneGraph::create_shape(const ShapeConfig &config) {
            Model ret;
            if (config.type() == "model") {
                config.fn = (_context->scene_path() / config.fn).string();
                ret = Model(config);
            } else if (config.type() == "quad") {
                ret = create_quad(config);
            } else if (config.type() == "quad_y") {
                ret = create_quad_y(config);
            } else if (config.type() == "mesh") {
                Box3f aabb;
                for (auto pos : config.positions) {
                    aabb.extend(pos);
                }
                Mesh mesh(move(config.positions), move(config.normals),
                          move(config.tex_coords), move(config.triangles), aabb);
                ret.meshes.push_back(mesh);
                ret.custom_material_name = config.material_name;
            } else if (config.type() == "cube") {
                ret = create_cube(config);
            } else if (config.type() == "sphere") {
                ret = create_sphere(config);
            } else {
                LUMINOUS_ERROR("unknown shape type !")
            }
            update_counter(ret);
            return ret;
        }

        void SceneGraph::create_shape_instance(const ShapeConfig &config) {
            auto key = gen_key(config);
            if (!is_contain(key)) {
                auto mp = create_shape(config);
                mp.key = key;
                _key_to_idx[key] = model_list.size();
                model_list.push_back(mp);
            }
            uint idx = _key_to_idx[key];
            Transform o2w = config.o2w.create();
            auto instance = ModelInstance(idx, o2w, config.name.c_str(), config.emission, config.two_sided);
            instance_num += model_list[instance.model_idx].meshes.size();
            instance_list.push_back(instance);
        }

        void SceneGraph::create_shapes() {
            TASK_TAG("create shapes")
            for (const auto &shape_config : shape_configs) {
                create_shape_instance(shape_config);
            }
        }
    }
}