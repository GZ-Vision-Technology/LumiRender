
import bpy
import os
import shutil
import json
import math

import numpy as np
from math import radians
# 工具

# #####################################################
# 工具、计算变换矩阵
# #####################################################


def to_mat(matrix4x4):
    items = []
    for col in matrix4x4.col:
        items.extend(col)
    mat = np.array(items).reshape(4, 4)
    return mat


def rotate_x(theta):
    theta = radians(theta)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    mat = [
        [1, 0,        0,         0],
        [0, cosTheta, sinTheta, 0],
        [0, -sinTheta, cosTheta,  0],
        [0, 0,        0,         1]
    ]
    return np.array(mat)


def scale(s):
    mat = [
        [s[0], 0, 0, 0],
        [0, s[1], 0, 0],
        [0, 0, s[2], 0],
        [0, 0, 0,     1]
    ]
    return np.array(mat)


def to_luminous():
    r = rotate_x(90)
    s = scale([1, 1, -1])
    t = np.matmul(s, r)
    return t


def model_to_luminous():
    r = rotate_x(0)
    s = scale([1, 1, 1])
    t = np.matmul(s, r)
    return t

# #####################################################
# create
# #####################################################


# def export_material(scene, scene_json, object, slot_idx):
#     mat = object.material_slots[slot_idx].material
#     if not mat or not mat.use_nodes:
#         return

#     def is_custom_node(node):
#         return isinstance(node, material_nodes.MyCustomTreeNode)

#     print('\nMat name: ', mat.name)
#     for node in mat.node_tree.nodes:
#         if not is_custom_node(node):
#             continue
#         if node.bl_idname == 'CustomNodeTypeMatte':
#             export_matte(scene_json, node, mat.name)
#     return mat.name

def export_material(scene, scene_json, object, slot_idx):

    mat = object.material_slots[slot_idx].material


def export_meshes(scene, scene_json):
    directory_path = bpy.path.abspath(scene.exportpath)

    def skip(object):
        return object is None or object.type == 'CAMERA' or object.type != 'MESH'
    # 一个mesh含有多个object
    for i, object in enumerate(scene.objects):
        if skip(object):
            continue
        mesh_name = object.name
        print(len(object.material_slots))
        # 每一个object 可能含有多个material
        for slotIndex in range(len(object.material_slots)):
            if len(object.material_slots) == 0:
                print("no material on object")
                continue
            mat = object.data.materials[slotIndex]
            if mat == None:
                textureName = ""
                links = mat.links
                print('Number of links: ')
                print(len(links))

        r = rotate_x(0)
        s = scale([1, 1, 1])
        t = np.matmul(s, r)
        # https://docs.blender.org/api/current/bpy.ops.export_scene.html?highlight=bpy%20ops%20export_scene%20obj#bpy.ops.export_scene.obj
        # bpy.ops.export_scene.obj(filepath=obj_filepath,
        #                          use_selection=False, path_mode='COPY')
        # 这里函数也会导出贴图
        bpy.ops.export_scene.gltf(filepath=directory_path+'/'+mesh_name+'.gltf',
                                  export_format='GLTF_SEPARATE', export_materials='EXPORT', export_colors=True)
        scene_json['shapes'].append({
            'name': mesh_name,
            'type': 'model',
            'param': {
                'fn': mesh_name+'.gltf',
                'smooth': False,
                'swap_handed': True,
                'subdiv_level': 0,
                'transform': {
                    'type': 'matrix4x4',
                    'param': {
                        'matrix4x4':  t.tolist()
                    }
                }
            }
        })


def export_meshes_all(scene, scene_json):
    directory_path = bpy.path.abspath(scene.exportpath)
    filepath = directory_path + '/all.gltf'
    # print(f'[info] export mesh to {obj_filepath}')
    # create_directory_if_needed('')

    # def skip(object):
    #     return object is None or object.type == 'CAMERA' or object.type != 'MESH'

    # for i, object in enumerate(scene.objects):
    #     if skip(object):
    #         continue
    #     mat_name = ""
    #     for i in range(len(object.material_slots)):
    #         mat_name = export_material(scene, scene_json, object, i)
    #     export_mesh(scene, scene_json, object, mat_name, i)

    r = rotate_x(0)
    s = scale([1, 1, 1])
    t = np.matmul(s, r)
    # https://docs.blender.org/api/current/bpy.ops.export_scene.html?highlight=bpy%20ops%20export_scene%20obj#bpy.ops.export_scene.obj
    # bpy.ops.export_scene.obj(filepath=obj_filepath,
    #                          use_selection=False, path_mode='COPY')

    bpy.ops.export_scene.gltf(filepath=filepath,
                              export_format='GLTF_SEPARATE', export_materials='EXPORT', export_colors=True)
    scene_json['shapes'].append({
        'name': 'mesh',
        'type': 'model',
        'param': {
            'fn': 'test.gltf',
            'smooth': False,
            'swap_handed': True,
            'subdiv_level': 0,
            'transform': {
                'type': 'matrix4x4',
                'param': {
                    'matrix4x4':  t.tolist()
                }
            }
        }
    })


def create_texture(type, name, color, color_sapce):
    return {
        "type": type,
        "name": name,
        "param": {"val": color, "color_space": color_sapce},
    }


# type: MatteMaterial
def create_material(type, name, param):
    return {
        "type": type,
        "name": name,
        "param": param,
    }


def create_shape_param(width, height, emission, scale, transform):
    return {
        "width": width,
        "height": height,
        "emission": emission,
        "scale": scale,
        "transform": transform,
    }


# type:yaw_pitch/rts
def create_transform(type, yaw, pitch, position):
    return {"type": type, "param": {"yaw": yaw, "pitch": pitch, "position": position}}


# type: quad/ model
def create_shape(type, name, shape_param):
    return {
        "type": type,
        "name": name,
        "param": shape_param,
    }


# type:
# def create_light():
#     return


def create_integrator(type, max_depth, rr_threshold):
    return {
        "type": type,
        "param": {"max_depth": max_depth, "rr_threshold": rr_threshold},
    }


def create_light_sampler(type):
    return {"type": type}


def export_area_lights(scene, scene_json):
    lights = []
    allow_light_shapes = ['RECTANGLE', 'SQUARE']
    for obj in scene.objects:
        if obj.type != 'LIGHT' or obj.data.type != 'AREA' or obj.data.shape not in allow_light_shapes:
            continue
        light_obj = obj
        light_data = obj.data

        if light_data.shape == 'SQUARE':
            width = light_data.size
            height = light_data.size
        else:
            width = light_data.size
            height = light_data.size_y

        mat = to_mat(light_obj.matrix_world)

        mat = np.matmul(mat, to_luminous())

        light = {
            'name': 'light_' + light_obj.name,
            'type': 'quad',
            'param': {
                'width': width,
                'height': height,
                'emission': list(light_data.color),
                'scale': light_data.energy * 0.00002,
                'transform': {
                    'type': 'matrix4x4',
                    'param': {
                        'matrix4x4': mat.tolist()

                    }
                },
            }
        }
        lights.append(light)

    if 'shapes' in scene_json:
        scene_json['shapes'].extend(lights)
    else:
        scene_json['shapes'] = lights


def create_camera_param(
    fov_y,
    velocity,
    focal_distance,
    lens_radius,
    transform,
    width,
    height,
    fb_state,
    filter_type,
    filter_radius,
):
    return {
        "fov_y": fov_y,
        "velocity": velocity,
        "focal_distance": focal_distance,
        "lens_radius": lens_radius,
        "transform": transform,
        "film": {"param": {"resolution": [width, height], "fb_state": fb_state}},
        "filter": {"type": filter_type, "param": {"radius": filter_radius}, },
    }


def create_camera(scene):
    camera_obj_blender = None
    camera_data_blender = None
    for camera_blender in bpy.data.cameras:
        if camera_blender.type == 'PERSP':
            camera_obj_blender = bpy.data.objects[camera_blender.name]
            camera_data_blender = camera_blender
            break
    if camera_obj_blender is None:
        print('[error]: no PERSP camera')
        return
    # scene.render.resolution_x是blender的渲染分辨率，把我们的渲染器跟blender的渲染器分开会好些

    angle_rad = camera_data_blender.angle_y
    tab = {
        "render": 0,
        "normal": 1,
        "albedo": 2
    }
    mat = to_mat(camera_obj_blender.matrix_world)
    pos = mat[3][0:3]
    euler = camera_obj_blender.rotation_euler
    pitch = math.degrees(euler.x) - 90
    yaw = math.degrees(euler.z)
    return {
        'type': 'ThinLensCamera',
        'param': {
            # 需要和camera[0].fov绑定？不需要
            'fov_y': angle_rad * 180.0 / math.pi,
            'velocity': scene.cameravelocity,
            "focal_distance": camera_blender.dof.aperture_fstop,
            "lens_radius": 0.0,
            'transform': {
                "type": "yaw_pitch",
                "param": {
                    "yaw": yaw,
                    "pitch": pitch,
                    "position": [pos[0], pos[2], pos[1]]
                }
            },
            'film': {
                'param': {
                    'resolution': [scene.resolution_x, scene.resolution_y],
                    'fb_state': tab[scene.rendermode]
                }
            },
            "filter": {
                "type": scene.filterType,
                "param": {
                    "radius": [
                        scene.filter_radius_x,
                        scene.filter_radius_y
                    ]
                }
            }
        }
    }


def create_sampler(type, spp):
    return {"type": type, "param": {"spp": spp}}


def create_output(fn, frame_num):
    return {"fn": fn, "frame_num": frame_num}


# def export_scene(scene_json, filepath):
#     with open(filepath, "w") as outputfile:
#         json.dump(scene_json, outputfile, indent=4)


def export_test(scene):

    # 默认添加了output light_sampler 和一个纯色材质
    scene_json = {
        "textures": [{
            "type": "ConstantTexture",
            "name": "constant",
            "param": {
                "val": [
                     1,
                     1,
                     1,
                     0
                     ],
                "color_space": "SRGB"
            }
        }],
        "materials": [{
            "type": "MatteMaterial",
            "name": "quad",
            "param": {
                "color": "constant"
            }
        }],
        "shapes": [],
        "lights": [],
        "light_sampler": {
            "type": "UniformLightSampler"
        },
        "output": {
            "fn": "cornell-box.png",
            "frame_num": 0
        }
    }

    scene_json['camera'] = create_camera(scene)
    scene_json['sampler'] = create_sampler(scene.sampler, scene.spp)
    scene_json['integrator'] = create_integrator(
        scene.integrator, scene.max_depth, scene.rr_threshold)

    export_area_lights(scene, scene_json)

    export_meshes(scene, scene_json)

    # export
    exportpath = scene.exportpath  # "D:/code/blender2luminous/assets
    # 检查文件路径是否存在，因为在blender里面设置了，应该一般存在，如果存在则删除
    if not os.path.exists(exportpath):
        os.makedirs(exportpath)
    else:
        os.rmdir(exportpath)
        os.makedirs(exportpath)
    # 导出
    filename = scene.outputfilename  # output.json
    json_out = exportpath+'/'+filename
    print(scene_json)
    with open(json_out, "w") as outputfile:
        json.dump(scene_json, outputfile, indent=4)
