#!/usr/bin/env python
# coding: utf-8

import os, cv2
import matplotlib
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.io import imread, imshow
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabaz_score

B1, B2, imageLim = [], [], 500
for i,e in enumerate(os.listdir('./img/')):
    I = imread('./' + e)
    IMG = resize(I, (800, 800, 3), mode='constant', preserve_range=True)
    IMG = IMG + [10, 15, 18]
    B1.append(IMG)
    if i% 50 == 0:
        print(i, e)

img_batch = np.array(B1)
img_batch = np.asarray(img_batch, dtype=np.float32)/img_batch.max()
img_batch.shape, img_batch.max()

model = Unet(input_shape=(800, 800, 3)) 
model.load_weights('new_test.h5')
preds_test = model.predict(img_batch , batch_size = 1, verbose = 1)

""""""""""""""""""""""""""""""""""""""""""""""""
def mid(line):
    x1, x2 = line[0], line[2]
    y1, y2 = line[1], line[3]
    x_mid, y_mid = int(np.round((x1+x2)/2)), int(np.round((y1+y2)/2))
    return (x_mid, y_mid)
def crs(a1,b1,a2,b2):
    return a1*b2 - b1*a2
def converse(line1, line2):
    x1, y1, x2, y2 = line1
    a1, b1, c1 = (y2 - y1), (x2- x1), (x2*y1 - x1*y2)
    x3, y3, x4, y4 = line2
    a2, b2, c2 = (y4 - y3), (x4- x3), (x4*y3 - x3*y4)
    r1, r2 = int(np.round(crs(b1,c1,b2,c2)/crs(a1,b1,a2,b2))), int(np.round(crs(a1,c1,a2,c2)/crs(a1,b1,a2,b2)))
    return (r1, r2)
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]
def seg(line1, line2):
    return [(line1[0], line1[1]), (line2[0], line2[1])], [(line1[2], line1[3]), (line2[2], line2[3])]
def length(line):
    x1, x2 = line[0], line[2]
    y1, y2 = line[1], line[3]
    return np.sqrt((y2-y1)**2 + (x2-x1)**2)
def grad(line):
    x1, x2 = line[0], line[2]
    y1, y2 = line[1], line[3]
    return np.float64((y2- y1)/(x2-x1))
def mid(line):
    x1, x2 = line[0], line[2]
    y1, y2 = line[1], line[3]
    return (int(np.round((x1+x2)/2)), int(np.round((y1+y2)/2)))
def vanishingp(line1, line2):
    x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
    x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
    
    x_diff = (x1 - x3, x2 - x4)
    y_diff = (y1 - y3, y2 - y4)

    div = det(x_diff, y_diff)
    if div == 0:
        return None
    d = (det([x1, y1],[x3, y3]), det([x2, y2],[x4, y4]))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return int(x), int(y)
def merge(line1, line2):
    x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
    x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
    xtl, xtr = min([x1, x2, x3, x4]), max([x1, x2, x3, x4])
    ytl, ytr = y1, y4
    return (xtl, ytl, xtr, ytr)
def detect(line1, line2):
    x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
    x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
    a1, b1, c1 = (y2 - y1), (x2- x1), (x2*y1 - x1*y2)
    a2, b2, c2 = (y4 - y3), (x4- x3), (x4*y3 - x3*y4)
    if crs(a1,b1,a2,b2) == 0:
        return crs(b1,c1,b2,c2)/(1+crs(a1,b1,a2,b2))
    else: return crs(b1,c1,b2,c2)/crs(a1,b1,a2,b2)
def conn(lineset):
    refine = []
    for line1 in lineset:
        for line2 in lineset:
            if line1 != line2 and 0 < detect(line1, line2) < 800:
                refine.append(merge(line1, line2))
                if line1 in lineset: lineset.remove(line1)
                if line2 in lineset: lineset.remove(line2)
    return lineset, refine
def ladder(lineset):
    copy = lineset
    SET, length = [], []
    for ind, ele in enumerate(lineset):
        set_i = []
        tmp = ele
        linset = copy
        tmp_set =  [x for x in lineset if x != ele]
#         print(tmp, len(tmp_set))
        for x1, y1, x2, y2 in tmp_set:
            if x1 > tmp[0] and x2 < tmp[2] and y1 < tmp[1] and y2 < tmp[3]:
                set_i.append([x1, y1, x2, y2])
                tmp = [x1,y1,x2,y2]
        SET.append(set_i)
        length.append(len(set_i))
    return SET[length.index(max(length))]
def clustering(lineset, n_clusters):
    check = []
    for line in lineset:
        check.append(mid(line))
    Xtr = np.array(check)
    if len(list(Xtr.shape)) == 1:
        Xtr = Xtr.reshape(-1,1)
    label = MiniBatchKMeans(n_clusters = n_clusters, 
                            batch_size = 7, 
                            random_state = 233).fit_predict(Xtr)
    score = metrics.calinski_harabaz_score(Xtr, label)
    labels = np.where(label == label[np.argmax(np.bincount(label))])
    refine = np.array(lineset)[labels].tolist()
    return refine
def vector(lineset):
    start_point, end_point = sorted(lineset)[0], sorted(lineset)[-1]
    if start_point[1] < end_point[1]:
        return end_point, start_point
    else:
        return start_point, end_point
""""""""""""""""""""""""""""""""""""""""""""""""

for i in range(500):
    T = 0.5
    kernelSize = (5, 5)
    threshold1, threshold2= 50, 100
    houghTransformThreshold = 120
    houghTransformMinLineLength = 110
    houghTransformMaxLineGap = 150
    preds_test_t2 = (preds_test > T).astype(np.uint8)
    img = np.squeeze(img_batch[i]).copy()
    img_color = np.array(img.copy()*255, dtype = np.uint8)
    img3 = np.squeeze(preds_test_t2[i]).copy()
    gray = cv2.GaussianBlur(img3.copy(), kernelSize, 0)
    lane = cv2.Canny(gray.copy()*125, 
                     threshold1 = threshold1, threshold2 = threshold2, 
                     apertureSize = 5, L2gradient = True)
    Lineset = cv2.HoughLinesP(lane.copy(), rho = 1.0, theta = np.pi/360, 
                                 threshold = houghTransformThreshold, 
                                 minLineLength = houghTransformMinLineLength, 
                                 maxLineGap = houghTransformMaxLineGap)  # lineset is the list of lines
    if Lineset != np.array([]):
        origin_set, midpoint = [], []
        for lines in Lineset:
            for x1, y1, x2, y2 in lines:
                if 1 > abs((y2-y1)/(x2-x1))> 0.1:
                    origin_set.append([x1, y1, x2, y2])
                    midpoint.append(mid([x1, y1, x2, y2]))
        if origin_set != []:
            LINE = ladder(origin_set)
            line_s, refine_s = conn(origin_set)
            Union = line_s + refine_s
            if len(LINE) >= 2:
                VP = vanishingp(sorted(LINE)[0],sorted(LINE)[-1])
                O = mid(LINE[0])
                Linel, Liner = seg(sorted(LINE)[0], sorted(LINE)[-1])
            CONV = []
            for x1, y1, x2, y2 in Union:
                Vline = [VP[0], VP[1], O[0], O[1]]
                cov = converse(Vline, [x1, y1, x2, y2])
                CONV.append(cov)
            if len(CONV) >= 2:
                start_p, end_p = vector(CONV)
                cv2.line(img_color, start_p, end_p, (255,255,0), 15)
                cv2.circle(img_color, start_p, 1,(255, 0, 0), 15)
                cv2.circle(img_color, end_p, 1,(0, 0, 255), 15)

""""""""""""""""""""""""""""""""""""""""""""""""

# cv2.imwrite('./vanishing_point/%d.jpg'%i, img_color*255.0,)
matplotlib.image.imsave('./vanishing_point/307/%d.jpg'%i, img_color)