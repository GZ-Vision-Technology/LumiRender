# -*- coding:utf-8 -*-

import json
from pathlib import Path
from posixpath import abspath
from sys import argv
import os
import glm

complexIorList = {
    "a-C" :{ "eta": (2.9440999183, 2.2271502925, 1.9681668794), "k": (0.8874329109, 0.7993216383, 0.8152862927)},
    "Ag" :{ "eta": (0.1552646489, 0.1167232965, 0.1383806959), "k": (4.8283433224, 3.1222459278, 2.1469504455)},
    "Al" :{ "eta": (1.6574599595, 0.8803689579, 0.5212287346), "k": (9.2238691996, 6.2695232477, 4.8370012281)},
    "AlAs" :{ "eta": (3.6051023902, 3.2329365777, 2.2175611545), "k": (0.0006670247, -0.0004999400, 0.0074261204)},
    "AlSb" :{ "eta": (-0.0485225705, 4.1427547893, 4.6697691348), "k": (-0.0363741915, 0.0937665154, 1.3007390124)},
    "Au" :{ "eta": (0.1431189557, 0.3749570432, 1.4424785571), "k": (3.9831604247, 2.3857207478, 1.6032152899)},
    "Be" :{ "eta": (4.1850592788, 3.1850604423, 2.7840913457), "k": (3.8354398268, 3.0101260162, 2.8690088743)},
    "Cr" :{ "eta": (4.3696828663, 2.9167024892, 1.6547005413), "k": (5.2064337956, 4.2313645277, 3.7549467933)},
    "CsI" :{ "eta": (2.1449030413, 1.7023164587, 1.6624194173), "k": (0.0000000000, 0.0000000000, 0.0000000000)},
    "Cu" :{ "eta": (0.2004376970, 0.9240334304, 1.1022119527), "k": (3.9129485033, 2.4528477015, 2.1421879552)},
    "Cu2O" :{ "eta": (3.5492833755, 2.9520622449, 2.7369202137), "k": (0.1132179294, 0.1946659670, 0.6001681264)},
    "CuO" :{ "eta": (3.2453822204, 2.4496293965, 2.1974114493), "k": (0.5202739621, 0.5707372756, 0.7172250613)},
    "d-C" :{ "eta": (2.7112524747, 2.3185812849, 2.2288565009), "k": (0.0000000000, 0.0000000000, 0.0000000000)},
    "Hg" :{ "eta": (2.3989314904, 1.4400254917, 0.9095512090), "k": (6.3276269444, 4.3719414152, 3.4217899270)},
    "HgTe" :{ "eta": (4.7795267752, 3.2309984581, 2.6600252401), "k": (1.6319827058, 1.5808189339, 1.7295753852)},
    "Ir" :{ "eta": (3.0864098394, 2.0821938440, 1.6178866805), "k": (5.5921510077, 4.0671757150, 3.2672611269)},
    "K" :{ "eta": (0.0640493070, 0.0464100621, 0.0381842017), "k": (2.1042155920, 1.3489364357, 0.9132113889)},
    "Li" :{ "eta": (0.2657871942, 0.1956102432, 0.2209198538), "k": (3.5401743407, 2.3111306542, 1.6685930000)},
    "MgO" :{ "eta": (2.0895885542, 1.6507224525, 1.5948759692), "k": (0.0000000000, -0.0000000000, 0.0000000000)},
    "Mo" :{ "eta": (4.4837010280, 3.5254578255, 2.7760769438), "k": (4.1111307988, 3.4208716252, 3.1506031404)},
    "Na" :{ "eta": (0.0602665320, 0.0561412435, 0.0619909494), "k": (3.1792906496, 2.1124800781, 1.5790940266)},
    "Nb" :{ "eta": (3.4201353595, 2.7901921379, 2.3955856658), "k": (3.4413817900, 2.7376437930, 2.5799132708)},
    "Ni" :{ "eta": (2.3672753521, 1.6633583302, 1.4670554172), "k": (4.4988329911, 3.0501643957, 2.3454274399)},
    "Rh" :{ "eta": (2.5857954933, 1.8601866068, 1.5544279524), "k": (6.7822927110, 4.7029501026, 3.9760892461)},
    "Se-e" :{ "eta": (5.7242724833, 4.1653992967, 4.0816099264), "k": (0.8713747439, 1.1052845009, 1.5647788766)},
    "Se" :{ "eta": (4.0592611085, 2.8426947380, 2.8207582835), "k": (0.7543791750, 0.6385150558, 0.5215872029)},
    "SiC" :{ "eta": (3.1723450205, 2.5259677964, 2.4793623897), "k": (0.0000007284, -0.0000006859, 0.0000100150)},
    "SnTe" :{ "eta": (4.5251865890, 1.9811525984, 1.2816819226), "k": (0.0000000000, 0.0000000000, 0.0000000000)},
    "Ta" :{ "eta": (2.0625846607, 2.3930915569, 2.6280684948), "k": (2.4080467973, 1.7413705864, 1.9470377016)},
    "Te-e" :{ "eta": (7.5090397678, 4.2964603080, 2.3698732430), "k": (5.5842076830, 4.9476231084, 3.9975145063)},
    "Te" :{ "eta": (7.3908396088, 4.4821028985, 2.6370708478), "k": (3.2561412892, 3.5273908133, 3.2921683116)},
    "ThF4" :{ "eta": (1.8307187117, 1.4422274283, 1.3876488528), "k": (0.0000000000, 0.0000000000, 0.0000000000)},
    "TiC" :{ "eta": (3.7004673762, 2.8374356509, 2.5823030278), "k": (3.2656905818, 2.3515586388, 2.1727857800)},
    "TiN" :{ "eta": (1.6484691607, 1.1504482522, 1.3797795097), "k": (3.3684596226, 1.9434888540, 1.1020123347)},
    "TiO2-e" :{ "eta": (3.1065574823, 2.5131551146, 2.5823844157), "k": (0.0000289537, -0.0000251484, 0.0001775555)},
    "TiO2" :{ "eta": (3.4566203131, 2.8017076558, 2.9051485020), "k": (0.0001026662, -0.0000897534, 0.0006356902)},
    "VC" :{ "eta": (3.6575665991, 2.7527298065, 2.5326814570), "k": (3.0683516659, 2.1986687713, 1.9631816252)},
    "VN" :{ "eta": (2.8656011588, 2.1191817791, 1.9400767149), "k": (3.0323264950, 2.0561075580, 1.6162930914)},
    "V" :{ "eta": (4.2775126218, 3.5131538236, 2.7611257461), "k": (3.4911844504, 2.8893580874, 3.1116965117)},
    "W" :{ "eta": (4.3707029924, 3.3002972445, 2.9982666528), "k": (3.5006778591, 2.6048652781, 2.2731930614)},
}

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

def rotateZXY(R):
    return glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.z, (0, 0, 1)) 

def convert_srt(S, R, T):
    return glm.translate(T) * rotateZXY(R) * glm.scale(S)

def convert_shpe_transform(transform):
    T = glm.vec3(transform.get("position", [0,0,0]))
    R = glm.radians(glm.vec3(transform.get("rotation", [0,0,0])))
    S = glm.vec3(transform.get("scale", [1,1,1]))
    M = glm.scale(glm.vec3([-1,1,1])) * convert_srt(S, R, T)
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