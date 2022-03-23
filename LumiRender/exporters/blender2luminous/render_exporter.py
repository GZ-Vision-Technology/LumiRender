import bpy
import bmesh
import os
import math
from math import *
import numpy as np
import mathutils
from mathutils import Vector
import shutil
import struct
from io_mesh_ply import export_ply
import json

from importlib import reload

if __name__ == "__main__":
    from blender2luminous import material_nodes
    reload(material_nodes)
else:
    from . import material_nodes

#render engine custom begin
class LuminousRenderEngine(bpy.types.RenderEngine):
    bl_idname = 'Luminous_Renderer'
    bl_label = 'Luminous_Renderer'
    bl_use_preview = False
    bl_use_material = True
    bl_use_shading_nodes = False
    bl_use_shading_nodes_custom = False
    bl_use_texture_preview = True
    bl_use_texture = True
    
    def render(self, scene):
        self.report({'ERROR'}, 'Use export function in PBRT panel.')
        
from bl_ui import properties_render
from bl_ui import properties_material
for member in dir(properties_render):
    subclass = getattr(properties_render, member)
    try:
        subclass.COMPAT_ENGINES.add('Luminous_Renderer')
    except:
        pass

for member in dir(properties_material):
    subclass = getattr(properties_material, member)
    try:
        subclass.COMPAT_ENGINES.add('Luminous_Renderer')
    except:
        pass

bpy.utils.register_class(LuminousRenderEngine)


def create_directory_if_needed(directory_filepath, force=False):
    if not os.path.exists(directory_filepath):
        os.makedirs(directory_filepath)
    elif force:
        os.rmdir(directory_filepath)
        os.makedirs(directory_filepath)

def get_filename(filepath):
    folderpath, filename = os.path.split(filepath)
    return filename

def scale(s):
    mat = [
        [s[0], 0, 0, 0],
        [0, s[1], 0, 0],
        [0, 0, s[2], 0],
        [0, 0, 0,     1]
    ]
    return np.array(mat)
    
def to_lh_cs():
    mat = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    return np.array(mat)

def convert_view(v):
    h_t = np.array([
        [0,0,1,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1]
    ])
    w = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return np.matmul(w, np.matmul(v, h_t))

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

def rotate_y(theta):
    theta = radians(theta)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    mat = [
        [cosTheta,  0, -sinTheta, 0],
        [0,         1, 0,        0],
        [sinTheta, 0, cosTheta, 0],
        [0,         0, 0,        1]
    ]
    return np.array(mat)

def rotate_z(theta):
    theta = radians(theta)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    mat = [
        [cosTheta,  sinTheta, 0, 0],
        [-sinTheta,  cosTheta, 0, 0],
        [0,         0,        1, 0],
        [0,         0,        0, 1]
    ]
    return np.array(mat)

def to_luminous():
    r = rotate_x(90)
    s = scale([1,1,-1])
    t = np.matmul(s, r)
    return t

def matrixToList_and_swaphanded(matrix4x4):
    items = []
    for col in matrix4x4.col:
        items.extend(col)

    mat = np.array(items).reshape(4,4)
    s = scale([-1,1,1])
    
    mat = np.matmul(s, mat)

    return mat.tolist()

def matrixToList_and_rotate(matrix4x4, angle):
    items = []
    for col in matrix4x4.col:
        items.extend(col)

    mat = np.array(items).reshape(4,4)
    rot = rotate_x(angle)
    
    mat = np.matmul(mat, rot)

    return mat.tolist() 

def to_mat(matrix4x4):
    items = []
    for col in matrix4x4.col:
        items.extend(col)
    mat = np.array(items).reshape(4,4)
    return mat

def getTextureInSlotName(textureSlotParam):
    srcfile = textureSlotParam
    head, tail = os.path.split(srcfile)
    print("File name is :")
    print(tail)

    return tail

def copy_image_to_dst_dir(textureSlotParam):
    srcfile = bpy.path.abspath(textureSlotParam)
    texturefilename = getTextureInSlotName(srcfile)

    target_file = bpy.path.abspath(bpy.data.scenes[0].exportpath + 'textures/' + texturefilename)
    dst_dir = os.path.dirname(target_file)
    if not os.path.exists(dst_dir):
         os.makedirs(dst_dir)
    shutil.copyfile(srcfile, target_file)
    return 'textures/' + texturefilename

def export_texture_from_input(inputSlot, mat):
    ret = ""
    for x in inputSlot.links:
        textureName = x.from_node.image.name
        ret = copy_image_to_dst_dir(x.from_node.image.filepath)
    return ret

def create_constant_tex(name, val):
    return {
        "type" : "ConstantTexture",
        "name" : name,
        "param" : {
            "val": val,
			"color_space": "SRGB"
        }
    }

def create_image_tex(path):
    return {
        "type" : "ImageTexture",
        "name" : path,
        "param" : {
            "fn" : path,
            "color_space": "SRGB"
        }
    }

def export_matte(scene_json, mat, mat_name):
    print("\nexport matte start !")

    Kd = [mat.Kd[0],mat.Kd[1],mat.Kd[2],mat.Kd[3]]

    image_path = export_texture_from_input(mat.inputs[0],mat)

    if image_path:
        tex_data = create_image_tex(image_path)
    else:
        tex_data = create_constant_tex(mat_name + "_constant", Kd)

    add_textures(scene_json, tex_data)

    tab = {
        "type": "MatteMaterial",
        "name": mat_name,
        "param": {
            "diffuse": tex_data["name"]
        }
    }
    add_material(scene_json, tab)

    print("export matte end !")

def export_material(scene, scene_json, object, slot_idx):
    mat = object.material_slots[slot_idx].material 
    if not mat or not mat.use_nodes:
        return
    
    def is_custom_node(node):
        return isinstance(node, material_nodes.MyCustomTreeNode)

    print('\nMat name: ', mat.name)
    for node in mat.node_tree.nodes:
        if not is_custom_node(node):
            continue
        if node.bl_idname == 'CustomNodeTypeMatte':
            export_matte(scene_json, node, mat.name)
    return mat.name
            
def model_transform(mat):
    mat = []

def export_mesh(scene, scene_json, object, mat_name, i):
    print('exporting object:' , object.name)
    bpy.context.view_layer.update()
    object.data.update()
    dg = bpy.context.evaluated_depsgraph_get()
    eval_obj = object.evaluated_get(dg)
    mesh = eval_obj.to_mesh()
    if not mesh.loop_triangles and mesh.polygons:
        mesh.calc_loop_triangles()

    mesh.calc_normals_split()

    objFolderPath = bpy.path.abspath(bpy.data.scenes[0].exportpath + 'meshes/')
    if not os.path.exists(objFolderPath):
        print('Meshes directory did not exist, creating: ')
        print(objFolderPath)
        os.makedirs(objFolderPath)

    objFilePath = objFolderPath + object.name + f'.ply' 
    objFilePathRel = 'meshes/' + object.name + f'.ply'


    export_ply.save_mesh(objFilePath, mesh, True, True, True, False)

    mat = to_mat(object.matrix_world)

    mat = np.matmul(mat, to_luminous())

    data = {
        "name" : object.name,
        "type" : "model",
        "param" : {
            "fn": objFilePathRel,
            "subdiv_level": 0,
             "material" : mat_name,
            'transform': {
                'type': 'matrix4x4',
                'param': {
                    'matrix4x4':  mat.tolist()
                }
            },
        }
    }
    scene_json["shapes"].append(data)


def export_meshes(scene, scene_json):
    obj_directory_path = bpy.path.abspath(scene.exportpath + 'meshes')
    obj_filepath =  obj_directory_path + '/meshes.gltf'
    # print(f'[info] export mesh to {obj_filepath}')
    create_directory_if_needed(obj_directory_path)

    if scene.file_format == ".ply":

        def skip(object):
            return object is None or object.type == 'CAMERA' or object.type != 'MESH'

        for i, object in enumerate(scene.objects):
            if skip(object):
                continue
            mat_name = ""
            for i in range(len(object.material_slots)):
                mat_name = export_material(scene, scene_json, object, i)
                break
            
            export_mesh(scene, scene_json, object, mat_name, i)

        return
    
    r = rotate_x(0)
    s = scale([1,1,1])
    t = np.matmul(s, r)

    bpy.ops.export_scene.gltf(filepath=obj_filepath,export_format="GLTF_SEPARATE" )
    scene_json['shapes'] = [{
        'name': 'mesh',
        'type': 'model',
        'param': {
            'fn': 'meshes/meshes.gltf',
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
    }]

def export_environmentmap(scene, scene_json):
    if scene.environmentmaptpath == '':
        print('export: environmentmap path is empyt')
        return
    environmentMapFileName = get_filename(scene.environmentmaptpath)
    srcfile = bpy.path.abspath(scene.environmentmaptpath)
    dstdir = bpy.path.abspath(scene.exportpath + 'textures')
    dstfile = dstdir + '/' + environmentMapFileName
    create_directory_if_needed(dstdir)
    shutil.copyfile(srcfile, dstfile)
    environmentmapscaleValue = scene.environmentmapscale
    environment_texture = {
        'name': 'envmap',
        'type': 'ImageTexture',
        'param': {
            'fn': 'textures/' + environmentMapFileName,
            'color_space': 'LINEAR'
        }
    }
    environment_light = {
        'type': 'Envmap',
        'param': {
            'transform' : {
                'type' : 'yaw_pitch',
                'param' : {
                    'yaw' : 0,
                    'pitch': 0,
                    'position': [0,0,0]
                }
            },
            'scale': [environmentmapscaleValue, environmentmapscaleValue, environmentmapscaleValue],
            'key' : 'envmap'
        }
    }
    if 'textures' in scene_json:
        scene_json['textures'].append(environment_texture)
    else:
        scene_json['textures'] = [environment_texture]
    if 'lights' in scene_json:
        scene_json['lights'].append(environment_light)
    else:
        scene_json['lights'] = [environment_light]

def export_point_lights(scene, scene_json):
    lights = []
    # 不能用bpy.data.objects[bpy.data.lights[0].name]这种形式来获取对象，name不是key，有可能两者不一致，如之前碰到的name=面光，key=g面光
    # for light_data in bpy.data.lights:
    #     light_obj = bpy.data.objects[light_data.name]
    
    for obj in scene.objects:
        if obj.type != 'LIGHT' or obj.data.type != 'POINT':
            continue
        light_obj = obj
        light_data = obj.data
        
        mat = to_mat(light_obj.matrix_world)
        
        r = rotate_x(90)
        s = scale([1,1,-1])

        t = np.matmul(s, r)

        mat = np.matmul(mat, t)
        
        light = {
            'type': 'PointLight',
            'param': {
                'transform': {
                    'type': 'matrix4x4',
                    'param': {
                        'matrix4x4':  mat.tolist()
                    }
                },
                'color': list(light_data.color)
            }
        }
        lights.append(light)

    if 'lights' in scene_json:
        scene_json['lights'].extend(lights)
    else:
        scene_json['lights'] = lights

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
                'scale': light_data.energy,
                'transform': {
                    'type': 'matrix4x4',
                    'param': {
                        'matrix4x4': mat.tolist()
                    }
                },
                'material': ''
            }
        }
        lights.append(light)

    if 'shapes' in scene_json:
        scene_json['shapes'].extend(lights)
    else:
        scene_json['shapes'] = lights

def yaw_pitch(m):
    pitch = math.degrees(math.atan2(m[1][2], (m[1][1])))
    yaw = math.degrees(math.atan2(m[2][0], m[0][0]))
    return yaw, pitch

def export_camera(scene, scene_json):
    camera = {}
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
    resolution_x = scene.resolution_x
    resolution_y = scene.resolution_y
    print('render res: ', resolution_x , ' x ', resolution_y)
    ratio = scene.render.resolution_y / scene.render.resolution_x
    angle_rad = camera_data_blender.angle_y

    tab = {
        "render" : 0,
        "normal" : 1,
        "albedo" : 2
    }

    mat = to_mat(camera_obj_blender.matrix_world)
    
    pos = mat[3][0:3]
    
    euler = camera_obj_blender.rotation_euler
    
    pitch = math.degrees(euler.x) - 90
    yaw = math.degrees(euler.z)
    
    scene_json['camera'] = {
        'type': 'ThinLensCamera',
        'param': {
            'fov_y': angle_rad * 180.0 / math.pi,
            'velocity': 20,
            'transform': {
                # 'type': 'matrix4x4',
                # 'param': {
                #     'matrix4x4': mat.tolist()
                # }
                "type" : "yaw_pitch",
                "param" : {
                    "yaw" : yaw,
                    "pitch" : pitch,
                    "position" : [pos[0], pos[2], pos[1]]
                }
            },
            'film': {
                'param': {
                    'resolution': [resolution_x, resolution_y],
                    'fb_state' : tab[scene.fb_state]
                }
            }
        }
    }

def export_integrator(scene, scene_json):
    scene_json['integrator'] = {
		'type' : scene.integrators,
		'param' : {
			'max_depth' : scene.maxdepth,
			'rr_threshold' : scene.rr_threshold
		}
	}

def export_light_sampler(scene, scene_json):
    scene_json['light_sampler'] = {
		'type': scene.light_sampler
	}

def export_sampler(scene, scene_json):
    scene_json['sampler'] = {
		'type' : scene.sampler,
		'param' : {
			'spp' : scene.spp
		}
	}

def export_filter(scene, scene_json):
    scene_json['filter'] = {
		'type' : scene.filterType,
		'param' : {
			'radius' : [scene.filter_x_width, scene.filter_y_width]
		}
	}

def export_render_output(scene, scene_json):
    scene_json['output'] = {
		'fn': scene.outputfilename,
		'frame_num' : scene.frame_num
    }

def export_scene(scene_json, filepath):
    with open(filepath, 'w') as outputfile:
        json.dump(scene_json, outputfile, indent=4)

def find_index(lst, key):
    for i, elm in enumerate(lst):
        if elm["name"] == key:
            return i
    return -1

def is_contain(lst, key):
    return find_index(lst, key) != -1


def create_texture(tex):
    ret = {}




def create_matte(mat, scene_json):
    ret = {}
    ret["type"] = "MatteMaterial"
    ret["name"] = mat.name
    ret["param"] = {
        "diffuse" : mat.Kd
    }
    return ret

def add_textures(scene_json, tex):
    index = find_index(scene_json["textures"], tex["name"])
    if index != -1:
        return index
    scene_json["textures"].append(tex)

def add_material(scene_json, mat):
    index = find_index(scene_json["materials"], mat["name"])
    if index != -1:
        return index
    scene_json["materials"].append(mat)


def export_luminous(filepath, scene):
    scene_json = {
        "textures":[],
        "materials":[],
        "shapes":[],
        "lights":[],
    }

    export_meshes(scene, scene_json)
    export_environmentmap(scene, scene_json)
    export_point_lights(scene, scene_json)
    export_area_lights(scene, scene_json)
    export_camera(scene, scene_json)
    export_integrator(scene, scene_json)
    export_light_sampler(scene, scene_json)
    export_sampler(scene, scene_json)
    export_filter(scene, scene_json)
    export_render_output(scene, scene_json)

    export_scene(scene_json, bpy.path.abspath(filepath + '/scene.json'))
    
