# -*- coding: utf-8 -*-
import cv2
import numpy as np
import ProcessTool as pt
import matplotlib.pyplot as plt

con = 0.99
eps = con * con

srcImgPath = "data/pic_0705/DSC03618.jpg"
#获取检测图片
originalImg = cv2.imread(srcImgPath)
#旋转矫正
rotateImg = pt.roteImg(originalImg)
cv2.imwrite("data/pic/rotateImg.jpg", rotateImg)
#计算和模板相似度获取二值图
threshImg = pt.simi(rotateImg, eps)
cv2.imwrite("data/pic/threshImg.jpg", threshImg)
#腐蚀膨胀
erodeImg = pt.erode_dilates(threshImg)
cv2.imwrite("data/pic/erodeImg.jpg", erodeImg)
#框出零件
rectImg, width, height = pt.find_rect(erodeImg, rotateImg)

cv2.imwrite("data/pic/rectImg.jpg", rectImg)
#获取零件中心点
h = int(rectImg.shape[0])
w = int(rectImg.shape[1])
midHeight = int(h / 2)
midWidth = int(w / 2)
offset = 50

# Hough transform
minLineLength = 200
maxLineGap = 200
edges = cv2.Canny(erodeImg, 50, 200)

lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 100, minLineLength=100, maxLineGap=10)
lines1 = lines[:,0,:]#提取为为二维
print lines1[:]
for x1,y1,x2,y2 in lines1[:]:
    cv2.line(rotateImg,(x1,y1),(x2,y2),(0,255,0),2)

print width, height

# pt.printImg('test', rotateImg)
# cv2.waitKey(0)


left_top_vertical = pt.computeRadis(rectImg, offset, midHeight, offset, midWidth)
left_top_horizon = pt.computeRadis(rectImg, offset, midWidth, offset, midHeight, flag=False)

left_bottom_vertical = pt.computeRadis(rectImg, midHeight, h - offset, offset, midWidth)
left_bottom_horizon = pt.computeRadis(rectImg, offset, midWidth, midHeight, h - offset, flag=False)

right_top_vertical = pt.computeRadis(rectImg, offset, midHeight, midWidth, w - offset)
right_top_horizon = pt.computeRadis(rectImg, midWidth, w - offset, offset, midHeight, flag=False)

right_bottom_vertical = pt.computeRadis(rectImg, midHeight, h - offset, midWidth, w - offset)
right_bottom_horizon = pt.computeRadis(rectImg, midWidth, w - offset, midHeight, h - offset, flag=False)

print("左上垂直直径：", left_top_vertical)
print("左上水平直径：", left_top_horizon)

print("左下垂直直径：", left_bottom_vertical)
print("左下水平直径：", left_bottom_horizon)

print("右上垂直直径：", right_top_vertical)
print("右上水平直径：", right_top_horizon)

print("右下垂直直径：", right_bottom_vertical)
print("右下水平直径：", right_bottom_horizon)