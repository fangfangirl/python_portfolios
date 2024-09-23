# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:31:39 2023

@author: USER
"""

import numpy as np
from PIL import Image

x1 = 607
y1 = 209
x2 = 907
y2 = 239
x3 = 608
y3 = 690
x4 = 904
y4 = 677

p1 = 0
q1 = 0
p2 = 439
q2 = 0
p3 = 0
q3 = 589
p4 = 439
q4 = 589

A = np.zeros((8, 8))
b = np.zeros((8, 1))
A[0, :] = [x1, y1, 1, 0, 0, 0, -x1*p1, -y1*p1]
A[1, :] = [0, 0, 0, x1, y1, 1, -x1*q1, -y1*q1]
A[2, :] = [x2, y2, 1, 0, 0, 0, -x2*p2, -y2*p2]
A[3, :] = [0, 0, 0, x2, y2, 1, -x2*q2, -y2*q2]
A[4, :] = [x3, y3, 1, 0, 0, 0, -x3*p3, -y3*p3]
A[5, :] = [0, 0, 0, x3, y3, 1, -x3*q3, -y3*q3]
A[6, :] = [x4, y4, 1, 0, 0, 0, -x4*p4, -y4*p4]
A[7, :] = [0, 0, 0, x4, y4, 1, -x4*q4, -y4*q4]
b[0, 0] = p1
b[1, 0] = q1
b[2, 0] = p2
b[3, 0] = q2
b[4, 0] = p3
b[5, 0] = q3
b[6, 0] = p4
b[7, 0] = q4
x = np.linalg.lstsq(A, b)[0]
H = np.zeros((3, 3))
H[0, 0] = x[0]
H[0, 1] = x[1]
H[0, 2] = x[2]
H[1, 0] = x[3]
H[1, 1] = x[4]
H[1, 2] = x[5]
H[2, 0] = x[6]
H[2, 1] = x[7]
H[2, 2] = 1

BG = Image.open('BG.jpg')
BG_data = np.asarray(BG).copy()
I = Image.open('sung.jpg')
data = np.asarray(I)
[H1, W1, t] = BG_data.shape
[H2, W2, t] = data.shape
xy1 = np.ones((3, 1))
pq1 = np.ones((3, 1))
for h in range(H1):
    for w in range(W1):
        xy1[0] = w
        xy1[1] = h
        pq1 = np.dot(H, xy1)
        p = int(pq1[0, 0] / pq1[2, 0])
        q = int(pq1[1, 0] / pq1[2, 0])
        if (p >= 0 and p < W2 and q >= 0 and q < H2):
            BG_data[h, w, :] = data[q, p, :]
I2 = Image.formarray(BG_data)
I2.show()
