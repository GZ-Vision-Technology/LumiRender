# -*- coding:utf-8 -*-

import os
import numpy as np
import math


def transform(pos):
    mat = np.array(
        [
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ]
    )
    return np.matmul(mat, np.array([pos[0],pos[1],pos[2],1]))[:3]
    
p0 = np.array([0,0,0])
p1 = np.array([0,1,0])
p2 = np.array([1,1,0])

v1 = p1 - p0
v2 = p2 - p0

ng = np.cross(v1, v2)

print(ng)

