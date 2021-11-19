# -*- coding:utf-8 -*-

import os
import numpy as np
import math

m1 = np.array([
    [1,2],
    [3,4]
])


m2 = np.array([
    [2,1],
    [4,3]
])

print(np.matmul(m1, m2))
print(np.matmul(m2, m1))

# count = 0

# com = 0

# inCom = False

# num_file = 0

# for root,dirs,files in os.walk(os.path.join(os.getcwd(), "LumiRender/src")):
#     for file in files:
#         fn = os.path.join(root,file)
#         if "ext\\" in fn:
#             continue
#         if "gui" in fn:
#             continue
#         if "stats.py" in fn:
#             continue
#         if "jitify" in fn:
#             continue
#         if "sdk_pt" in fn:
#             continue
#         try:
#             f = open(fn, "r")
#             count += len(f.readlines())
#         except :
#             print(fn)

        
        
#         num_file += 1


# print(count, num_file)
        
# class C:
#     pass
# class A:
#     pass
# class B(A, C):
#     pass
    
# print(issubclass(B,C))  

def rotation_x(theta):
    theta = math.radians(theta)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    mat = [
        [1, 0,        0,         0],
        [0, cosTheta, sinTheta, 0],
        [0, -sinTheta, cosTheta,  0],
        [0, 0,        0,         1]
    ]
    return np.array(mat)

def rotation_y(theta):
    theta = math.radians(theta)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    mat = [
        [cosTheta,  0, -sinTheta, 0],
        [0,         1, 0,        0],
        [sinTheta, 0, cosTheta, 0],
        [0,         0, 0,        1]
    ]
    return np.array(mat)

def view_mat(pitch, yaw):
    return np.matmul(rotation_y(yaw), rotation_x(-pitch))

def sqr(n):
    return n * n

class Camera:
    def __init__(self, mat):
        self.pitch = 0
        self.yaw = 0
        self.update(mat)
    
    def update(self, m):
        sy = math.sqrt(sqr(m[2][1]) + sqr(m[2][2]))
        self.pitch = math.degrees(math.atan2(m[2][1], abs(m[2][2])))
        self.yaw = math.degrees(-math.atan2(-m[2][0], sy))
        
    def camera_to_world_rotation():
        horizontal = rotation_y(yaw);
        vertical = rotation_x(-pitch);
        return np.matmul(horizontal, vertical)
    
    def __repr__(self) -> str:
        return "yaw:%s, pitch: %s" % (self.yaw, self.pitch)


yaw = 95
pitch = 20

mat = view_mat(pitch, yaw)

# print(rotation_x(pitch))
# print(rotation_y(yaw))

cam = Camera(mat)

print(cam)

