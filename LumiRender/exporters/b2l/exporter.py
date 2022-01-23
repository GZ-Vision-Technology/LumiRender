import bpy
import os
import json
import math
import numpy as np
from math import radians
import shutil
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


def create_imagetex(name, tex):
    return {
        "name": name,
        "type": "ImageTexture",
        "param": {
            "fn": 'textures/'+tex,
            "color_space": "LINEAR"
        }
    }
# https://blenderartists.org/t/simple-exporter-script-need-some-python-advice/1238345/6


def export_texture_from_input(inputSlot, scene_json, mesh_name):
    tex_name = ""
    links = inputSlot.links
    print('Number of links: ')
    print(len(links))
    for x in inputSlot.links:
        tex_name = x.from_node.image.name
        print(tex_name)
        src_texture_path = bpy.path.abspath(x.from_node.image.filepath)
        dst_texture_path = bpy.path.abspath(
            bpy.data.scenes[0].exportpath + '/textures/' + tex_name)
        # 自己拷贝纹理贴图
        shutil.copyfile(src_texture_path, dst_texture_path)
        # 写进json文件
        tex_data = create_imagetex(tex_name, tex_name)
        scene_json["textures"].append(tex_data)
        print("Copy %s  from %s to %s, " %
              (tex_name, src_texture_path, dst_texture_path))
    return tex_name


def export_matte_material(mat, scene_json, mesh_name):
    print('Currently exporting B2L Matte material')
    kdTextureName = ""
    material_name = mesh_name + "_" + mat.name
    kdTextureName = export_texture_from_input(
        mat.inputs[0], scene_json, mesh_name)
    m = {}  # create a dictionary
    m["name"] = material_name  # set material name
    m["type"] = "MatteMaterial"
    # 如果有贴图写贴图，没贴图写数值
    if kdTextureName != "":
        m["param"] = {"color": kdTextureName}
    else:
        m["param"] = {"color": [mat.Kd[0], mat.Kd[1], mat.Kd[2]]}
    # list
    scene_json['materials'].append(m)  # appending to list
    return material_name


def export_fake_metal_material(mat, scene_json, mesh_name):
    print('Currently exporting B2L fake metal material')
    kdTextureName = ""
    material_name = mesh_name + "_" + mat.name
    kdTextureName = export_texture_from_input(
        mat.inputs[0], scene_json, mesh_name)
    m = {}  # create a dictionary
    m["name"] = material_name  # set material name
    m["type"] = "FakeMetalMaterial"
    # 如果有贴图写贴图，没贴图写数值
    if kdTextureName != "":
        m["param"] = {"color": kdTextureName}
    else:
        m["param"] = {"color": [mat.color[0], mat.color[1], mat.color[2]]}
    m["param"]["roughness"] = [mat.roughness, mat.roughness]
    scene_json['materials'].append(m)  # appending to list
    return material_name

def exporter_materials(scene, scene_json):
    # 在blender由于需要使用normal_map节点对法线贴图进行采样，需要特殊处理
    def write_normal(node):
        if len(node.links) == 1:
            Normal_Map = node.links[0].from_node
            texture_node = Normal_Map.inputs.get("Color").links[0].from_node
            tex_name = texture_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name

    # color 节点返回float4
    def write_color(node):
        if len(node.links) == 1 and node.links[0].from_node.type == 'TEX_IMAGE':
            tex_name = node.links[0].from_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name
        else:
            Kd = [node.default_value[0],
                  node.default_value[1], node.default_value[2], node.default_value[3]]
            return Kd

    def write_material(node):
        # ensure input node is an image
        if len(node.links) == 1 and node.links[0].from_node.type == 'TEX_IMAGE':
            tex_name = node.links[0].from_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name
        else:
            val = node.default_value
            return val

    # iterating all of the materials
    for mat in bpy.data.materials:
        # get the bsdf node
        bsdf = None
        if mat.node_tree != None:  # should have node tree
            for n in mat.node_tree.nodes.keys():  # searching all nodes
                node = mat.node_tree.nodes[n]
                # 使用type进行判断
                if node.type == 'BSDF_PRINCIPLED':  # node type should be bdsf
                    bsdf = node  # setting the node
                    break  # exit node tree

        # bsdf not found, skipping
        if bsdf == None:
            continue
        # dictionary
        m = {}  # create a dictionary
        m["name"] = mat.name  # set material name
        m["type"] = "DisneyMaterial"
        m["param"] = {"color": 0.0, "roughness": 1.0, "eta": 1.5, "matallic": 0.0, "specular_tint": 0.0,
                      "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0, "clearcoat": 0.0, "clearcoat_gloss": 0.0}

        # get socket
        # normal = bsdf.inputs.get("Normal")
        m["param"]["color"] = write_color(
            bsdf.inputs.get("Base Color"), mat.name)
        m["param"]["roughness"] = write_material(bsdf.inputs.get("Roughness"))
        m["param"]["eta"] = write_material(bsdf.inputs.get("Subsurface IOR"))
        m["param"]["matallic"] = write_material(bsdf.inputs.get("Metallic"))
        m["param"]["specular_tint"] = write_material(
            bsdf.inputs.get("Specular Tint"))
        m["param"]["anisotropic"] = write_material(
            bsdf.inputs.get("Anisotropic"))
        m["param"]["sheen"] = write_material(bsdf.inputs.get("Sheen"))
        m["param"]["sheen_tint"] = write_material(
            bsdf.inputs.get("Sheen Tint"))
        m["param"]["clearcoat"] = write_material(bsdf.inputs.get("Clearcoat"))
        m["param"]["clearcoat_gloss"] = write_material(
            bsdf.inputs.get("Clearcoat Roughness"))
        m["param"]["normal"] = write_normal(
            bsdf.inputs.get("Normal"))
        # list
        scene_json['materials'].append(m)  # appending to list


def export_builtin_disney_material(bsdf, scene_json, mesh_name, mat):
    def write_normal(node):
        if len(node.links) == 1:
            Normal_Map = node.links[0].from_node
            texture_node = Normal_Map.inputs.get("Color").links[0].from_node
            tex_name = texture_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name

    # color 节点返回float4
    def write_float4(node):
        if len(node.links) == 1 and node.links[0].from_node.type == 'TEX_IMAGE':
            tex_name = node.links[0].from_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name
        else:
            Kd = [node.default_value[0],
                  node.default_value[1], node.default_value[2], node.default_value[3]]
            return Kd

    def write_float1(node):
        # ensure input node is an image
        if len(node.links) == 1 and node.links[0].from_node.type == 'TEX_IMAGE':
            tex_name = node.links[0].from_node.image.filepath.split("\\")[-1]
            tex_data = create_imagetex(tex_name, tex_name)
            scene_json["textures"].append(tex_data)
            return tex_name
        else:
            val = node.default_value
            return val
        
    material_name = mesh_name + "_" + mat.name
    m = {}
    m["name"] = material_name  # set material name
    m["type"] = "DisneyMaterial"
    
    param = {}
    param["color"] = write_float4(bsdf.inputs.get("Base Color"))
    param["roughness"] = write_float1(bsdf.inputs.get("Roughness"))
    param["eta"] = write_float1(bsdf.inputs.get("Subsurface IOR"))
    param["matallic"] = write_float1(bsdf.inputs.get("Metallic"))
    param["specular_tint"] = write_float1(
        bsdf.inputs.get("Specular Tint"))
    param["anisotropic"] = write_float1(
        bsdf.inputs.get("Anisotropic"))
    param["sheen"] = write_float1(bsdf.inputs.get("Sheen"))
    param["sheen_tint"] = write_float1(
        bsdf.inputs.get("Sheen Tint"))
    param["clearcoat"] = write_float1(bsdf.inputs.get("Clearcoat"))
    param["clearcoat_roughness"] = write_float1(
        bsdf.inputs.get("Clearcoat Roughness"))
    param["normal"] = write_normal(
        bsdf.inputs.get("Normal"))
    
    m["param"] = param
    
    scene_json['materials'].append(m)
    
    return material_name

def export_disney_material(mat, scene_json, mesh_name):
    print('Currently exporting B2L disney material')
    kdTextureName = ""
    material_name = mesh_name + "_" + mat.name
    kdTextureName = export_texture_from_input(
        mat.inputs[0], scene_json, mesh_name)
    m = {}  # create a dictionary
    m["name"] = material_name  # set material name
    m["type"] = "DisneyMaterial"
    param = {}
    if kdTextureName != "":
        param["color"] = kdTextureName
    else:
        param["color"] = [mat.color[0], mat.color[1], mat.color[2]]
    param["metallic"] = mat.metallic
    param["eta"] = mat.eta
    param["roughness"] = mat.roughness
    param["specular_tint"] = mat.specular_tint
    param["anisotropic"] = mat.anisotropic
    param["sheen"] = mat.sheen
    param["sheen_tint"] = mat.sheen_tint
    param["clearcoat"] = mat.clearcoat
    param["clearcoat_roughness"] = mat.clearcoat_roughness
    param["spec_trans"] = mat.spec_trans
    param["scatter_distance"] = [mat.scatter_distance[0],
                                 mat.scatter_distance[1],
                                 mat.scatter_distance[2]]
    param["flatness"] = mat.flatness
    param["diff_trans"] = mat.diff_trans
    param["thin"] = mat.thin

    m["param"] = param
    scene_json['materials'].append(m)

    return material_name


def export_material(scene, scene_json, object, mesh_name):
    from. import material_nodes
    # 一个mesh可能有多个材质
    for slot_index, material in enumerate(object.material_slots):
        print("slot_index", slot_index)
        # 如果使用Nodes,则使用blender内建的BSDF,交给gltf导出对应材质，否则使用自定义的材质节点
        mat = object.material_slots[slot_index].material
        if not(mat):
            return
        elif (mat.use_nodes):
            print("use blender bsdf--------", mat)
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    return export_builtin_disney_material(node, scene_json, mesh_name, mat)
            return
        else:
            for material in mat.node_tree.nodes:
                if material.bl_idname == material_nodes.B2L_Matte.bl_idname:
                    return export_matte_material(material, scene_json, mesh_name)
                elif material.bl_idname == material_nodes.B2L_FakeMetal.bl_idname:
                    return export_fake_metal_material(material, scene_json, mesh_name)
                elif material.bl_idname == material_nodes.B2L_Disney.bl_idname:
                    return export_disney_material(material, scene_json, mesh_name)


def export_mesh(scene, scene_json, object, directory_path):
    mesh_name = object.name
    # 模型变换矩阵
    r = rotate_x(0)
    s = scale([1, 1, 1])
    t = np.matmul(s, r)

    # 一个material可能会有多个材质
    # 返回当前material的name,export_material导出材质并导出贴图
    material_name = ""
    material_name = export_material(scene, scene_json, object, mesh_name)

    # export_materials ['EXPORT', 'PLACEHOLDER', 'NONE'] PLACEHOLDER保留material slot信息但不导出贴图
    bpy.ops.export_scene.gltf(filepath=directory_path+'/meshes/'+mesh_name+'.gltf',
                              export_texture_dir='../textures',
                              export_format='GLTF_SEPARATE',
                              export_materials='EXPORT',
                              export_colors=False,
                              use_selection=True)

    scene_json['shapes'].append({
        'name': mesh_name,
        'type': 'model',
        'param': {
            'fn': 'meshes/'+mesh_name+'.gltf',
            'smooth': False,
            'material': material_name,
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

# 每个模型单独导出.gltf


def export_meshes(scene, scene_json):

    directory_path = bpy.path.abspath(scene.exportpath)
    obs = [o for o in scene.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    viewlayer = bpy.context.view_layer
    for i, object in enumerate(obs):
        viewlayer.objects.active = object
        object.select_set(True)
        export_mesh(scene, scene_json, object, directory_path)
        object.select_set(False)


def create_integrator(scene):
    return {
        "type": scene.integrator,
        "param": {
            "min_depth": scene.min_depth,
            "max_depth": scene.max_depth,
            "rr_threshold": scene.rr_threshold
        },
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
        # blender和引擎之间的灯光有一个单位转换系数，这里暂定为0.00002
        lightscale = scene.lightscale
        light = {
            'name': 'light_' + light_obj.name,
            'type': 'quad',
            'param': {
                'width': width,
                'height': height,
                'emission': list(light_data.color),
                'scale': light_data.energy * lightscale,
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


def create_filter(scene):
    ret = {
        "type": scene.filterType,
        "param": {
            "radius": [
                scene.filter_radius_x,
                scene.filter_radius_y
            ]
        }
    }
    if scene.filterType == "LanczosSincFilter":
        ret["param"]["tau"] = scene.filter_tau
    elif scene.filterType == "GaussianFilter":
        ret["param"]["sigma"] = scene.filter_sigma
    elif scene.filterType == "MitchellFilter":
        ret["param"]["B"] = scene.filter_B
        ret["param"]["C"] = scene.filter_C
    return ret


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
            "filter": create_filter(scene)
        }
    }


def create_sampler(scene):
    return {"type": scene.sampler, "param": {"spp": scene.spp}}


def create_output(scene):
    return {"fn": scene.picture_name, "dispatch_num": scene.dispatch_num}


def write_scene(scene_json, filepath):
    # print(scene_json)
    with open(filepath, "w") as outputfile:
        json.dump(scene_json, outputfile, indent=4)

# 检查文件路径是否存在，因为在blender里面设置了，应该一般存在，如果存在则删除


def create_directories(exportpath):
    if not os.path.exists(exportpath):
        os.makedirs(exportpath)
        os.makedirs(exportpath+'/meshes')
        os.makedirs(exportpath+'/textures')
    else:
        shutil.rmtree(exportpath)
        os.makedirs(exportpath)
        os.makedirs(exportpath+'/meshes')
        os.makedirs(exportpath+'/textures')


def export_meshes_all(scene, scene_json):
    directory_path = bpy.path.abspath(scene.exportpath)
    filepath = directory_path + '/all.gltf'
    r = rotate_x(0)
    s = scale([1, 1, 1])
    t = np.matmul(s, r)
    bpy.ops.export_scene.gltf(filepath=directory_path+'/meshes/all.gltf',
                              export_texture_dir='../textures',
                              export_format='GLTF_SEPARATE',
                              export_materials='EXPORT',
                              export_colors=False,)
    scene_json['shapes'].append({
        'name': 'mesh',
        'type': 'model',
        'param': {
            'fn': 'all.gltf',
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


def export_test(scene, filepath_full):
    scene_json = {
        "textures": [],
        "materials": [],
        "shapes": [],
        "lights": [],
        "light_sampler": {
            "type": "UniformLightSampler"
        },
        "output": {}
    }
    # export
    exportpath = filepath_full  # default: output

    print("exportpath------------", exportpath)
    create_directories(exportpath)
    scene_json['camera'] = create_camera(scene)
    scene_json['sampler'] = create_sampler(scene)
    scene_json['integrator'] = create_integrator(scene)
    export_area_lights(scene, scene_json)
    if scene.meshtype == 'Single':
        export_meshes(scene, scene_json)
    else:
        export_meshes_all(scene, scene_json)
    # 自定义材质导出
    # exporter_custom_materials(scene, scene_json)
    # 原先的BSDF材质
    # exporter_materials(scene, scene_json)
    scene_json['output'] = create_output(scene)

    # 导出
    write_scene(scene_json, exportpath+'/'+scene.outputfilename)
