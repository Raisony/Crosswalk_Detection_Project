import os
import cv2
import sys
import numpy as np

def onTrackbarChange(max_slider):
    global img
    global dst
    global gray
    dst = np.copy(img)
    th1 = max_slider 
    th2 = th1 * 0.4
    edges = cv2.Canny(img, th1, th2)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180.0, 50, minLineLength = 10, maxLineGap = 100)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(dst, (x1, y1), (x2, y2), (0,0,255), 1)
    cv2.imshow('Result', dst)
    cv2.imshow('Edges', edges)

List = os.listdir('./Folder 1/')
img = cv2.imread('./Folder 1/' + List[0])

dst = np.copy(img)
img.astype('uint8')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Edges')
cv2.namedWindow('Result')

initThresh, maxThresh = 500, 1000

cv2.createTrackbar('Threshold', 'Result', initThresh, maxThresh, onTrackbarChange)
onTrackbarChange(initThresh)

while True:
    key = cv2.waitKey(1)
    if key == 10:
        break
cv2.destroyAllWindows()