#!/bin/python
# coding: utf-8
import math

# cos 输入是 弧度值
#     输出是 [-1, 1]的 角度相似度, [2pi, 0]
# acos 输入是 角度相似度
#     输出是

pi = 3.1416


print("du: ", [0, 60, 90, 120, 180, 240, 360])
arc = [0, pi/3, pi/2, pi*2/3, pi, pi*4/3, pi*2]
print("pi: ", arc)
cosList = list(map(math.cos, arc))
print("cos: ", cosList)
acosList = list(map(math.acos, cosList))
print("acos: ", acosList)
