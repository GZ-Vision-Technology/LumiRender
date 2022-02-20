# -*- coding:utf-8 -*-

import json
from pathlib import Path
from sys import argv
import os
import glm

def convert_material():
    pass

def main():
    fn = "LumiRender/res/render_scene/cornell-box/tungsten_scene.json"
    parent = os.path.dirname(fn)
    output_fn = os.path.join(parent, "lumi_scene.json")
    # print()
    with open(fn) as file:
        scene = json.load(file)
    
    help(glm.translate)



if __name__ == "__main__":
    main()