# -*- coding:utf-8 -*-

import json
from pathlib import Path
from sys import argv
import os
import glm

def add_textures(scene_output, name):
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

def convert_diffuse(mat_input):
    ret = {
        "type" : "MatteMaterial",
        "name" : mat_input["name"],
        "param" :{
            "color" : mat_input["albedo"],
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

def convert_transform(transform):
    ret = {}
    return ret

def convert_quad(shape_input, index):
    ret = {
        "type" : "quad_y",
        "name" : "shape_" + str(index),
        "width": 1.0,
        "height": 1.0,
        "material" : shape_input["bsdf"],
        "transform" : convert_transform(shape_input["transform"])
    }
    return ret

def convert_shapes(scene_input):
    shape_outputs = []
    shape_inputs = scene_input["primitives"]
    for i, shape_input in enumerate(shape_inputs):
        if shape_input["type"] == "quad":
           shape_outputs.append(convert_quad(shape_input, i))
           
    return shape_outputs 

def write_scene(scene_output, filepath):
    with open(filepath, "w") as outputfile:
        json.dump(scene_output, outputfile, indent=4)

def main():
    help(glm.translate)
    fn = "LumiRender/res/render_scene/cornell-box/tungsten_scene.json"
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
        "light_sampler" : {},
        "sampler" : {},
        "output" : {},
        "camera" : {},
        "integrator" : {},
    }
    write_scene(scene_output, output_fn)


if __name__ == "__main__":
    main()