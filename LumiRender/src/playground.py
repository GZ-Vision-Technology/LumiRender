# -*- coding:utf-8 -*-

import os
import numpy as np
import math


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
        # self.pitch = math.degrees(math.atan2(m[2][1], abs(m[2][2])))
        # self.yaw = math.degrees(-math.atan2(-m[2][0], sy))
        
        self.pitch = -math.degrees(math.atan2(m[1][2], (m[1][1])))
        self.yaw = math.degrees(math.atan2(m[2][0], m[0][0]))
        
        # print("-----------------------", (m[2][0], m[0][0]))
        
    def camera_to_world_rotation(self):
        horizontal = rotation_y(self.yaw);
        vertical = rotation_x(-self.pitch);
        return np.matmul(horizontal, vertical)
    
    def __repr__(self) -> str:
        return "yaw:%s, pitch: %s" % (self.yaw, self.pitch)


yaw = -150
pitch = 20

mat = view_mat(pitch, yaw)

# print(rotation_x(-pitch))
# print(rotation_y(yaw))
print(mat)

cam = Camera(mat)

print(cam)

print(cam.camera_to_world_rotation())

