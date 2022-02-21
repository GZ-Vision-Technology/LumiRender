# -*- coding:utf-8 -*-

import json
from pathlib import Path
from posixpath import abspath
from sys import argv
import os
import glm

def try_add_textures(scene_output, name):
    textures = scene_output["textures"]
    for tex in textures:
        if tex["name"] == name:
            return
    textures.append({
        "name": name,
        "type": "ImageTexture",
        "param": {
            "fn": name,
            "color_space": "LINEAR"
        }
    })

def parse_attr(attr):
    if type(attr) == str:
        try_add_textures(attr)
    return attr

def convert_diffuse(mat_input):
    ret = {
        "type" : "MatteMaterial",
        "name" : mat_input["name"],
        "param" :{
            "color" : parse_attr(mat_input["albedo"]),
        }
    }
    return ret
    

def convert_materials(scene_input):
    mat_inputs = scene_input["bsdfs"]
    mat_outputs = []
    for mat_input in mat_inputs:
        if mat_input["type"] == "lambert":
            mat_outputs.append(convert_diffuse(mat_input))
            
    return mat_outputs

def rotateXYZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0))

def rotateYXZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.y, (0, 1, 0))

def rotateZXY(R):
    return glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.z, (0, 0, 1)) 

def rotateZYX(R):
    return glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.z, (0, 0, 1)) 

def convert_srt(S, R, T):
    return glm.translate(T) * rotateZXY(R) * glm.scale(S)

def convert_shpe_transform(transform):
    T = glm.vec3(transform.get("position", [0,0,0]))
    R = glm.radians(glm.vec3(transform.get("rotation", [0,0,0])))
    scale = transform.get("scale", [1,1,1])
    # scale[0] = -scale[0]
    S = glm.vec3(scale)
    M = glm.scale(glm.vec3([-1,1,1]))* convert_srt(S, R, T)
    matrix4x4 = []
    for i in M:
        arr = []
        matrix4x4.append(arr)
        for j in i:
            arr.append(j)
            
    ret = {
        "type" : "matrix4x4",
        "param" : {
            "matrix4x4" : matrix4x4
        }
    }
    return ret

def convert_quad(shape_input, index):
    ret = {
        "type" : "quad_y",
        "name" : "shape_" + str(index),
        "param" : {
            "width": 1.0,
            "height": 1.0,
            "material" : shape_input["bsdf"],
            "transform" : convert_shpe_transform(shape_input["transform"])
        }
    }
    return ret

def convert_cube(shape_input, index):
    ret = {
        "type" : "cube",
        "name" : "shape_" + str(index),
        "param" : {
            "x" : 1,
            "y" : 1,
            "z" : 1,
            "material" : shape_input["bsdf"],
            "transform" : convert_shpe_transform(shape_input["transform"])
        }
    }
    return ret

def convert_shapes(scene_input):
    shape_outputs = []
    shape_inputs = scene_input["primitives"]
    for i, shape_input in enumerate(shape_inputs):
        shape_output = None
        if shape_input["type"] == "quad":
            shape_output = convert_quad(shape_input, i)
        if shape_input["type"] == "cube":    
            shape_output = convert_cube(shape_input, i)
            
        if "emission" in shape_input:
            shape_output["param"]["emission"] = shape_input["emission"]
            shape_output["param"]["material"] = None
        shape_outputs.append(shape_output)
    return shape_outputs

def convert_camera(scene_input):
    camera_input = scene_input["camera"]
    transform = camera_input["transform"]
    ret = {
        "type" : "ThinLensCamera",
        "param" : {
            "fov_y" : camera_input["fov"],
            "velocity" : 20,
            "transform" : {
                "type" : "look_at",
                "param": {
                    "position" : transform["position"],
                    "up" : transform["up"],
                    "target_pos" : transform["look_at"]
                }
            },
            "film" : {
                "param" : {
                    "resolution" : camera_input.get("resolution", [768, 768]),
                    "fb_state": 0
                }
            },
            "filter": {
                "type": "BoxFilter",
                "param": {
                    "radius": [0.5,0.5]
                }
            }
        },
    }
    return ret

def convert_integrator(scene_input):
    integrator = scene_input["integrator"]
    ret = {
        "type" : "PT",
        "param" : {
			"min_depth" : integrator["min_bounces"],
			"max_depth" : integrator["max_bounces"],
			"rr_threshold" : 1
		}
    }
    return ret

def convert_light_sampler(scene_input):
    ret = {
        "type" : "UniformLightSampler"
    }
    return ret

def convert_sampler(scene_input):
    ret = {
        "type" : "PCGSampler",
		"param" : {
			"spp" : 1
		}
    }
    return ret

def convert_output_config(scene_input):
    renderer = scene_input["renderer"]
    ret = {
        "fn" : renderer.get("output_file", "scene.png"),
        "dispatch_num" : renderer.get("spp", 0)
    }
    return ret

def write_scene(scene_output, filepath):
    with open(filepath, "w") as outputfile:
        json.dump(scene_output, outputfile, indent=4)
    abspath = os.path.join(os.getcwd(), filepath)
    print("lumi scene save to:", abspath)

def main():
    fn = 'LumiRender\\res\\render_scene\\cornell-box\\tungsten_scene.json'
    parent = os.path.dirname(fn)
    output_fn = os.path.join(parent, "lumi_scene.json")
    # print()
    with open(fn) as file:
        scene_input = json.load(file)
    scene_output = {
        "textures" : [],
        "materials" : convert_materials(scene_input),
        "shapes" : convert_shapes(scene_input),
        "lights" : [],
        "light_sampler" : convert_light_sampler(scene_input),
        "sampler" : convert_sampler(scene_input),
        "integrator" : convert_integrator(scene_input),
        "camera" : convert_camera(scene_input),
        "output" : convert_output_config(scene_input),
    }
    write_scene(scene_output, output_fn)


if __name__ == "__main__":
    main()