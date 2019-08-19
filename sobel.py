# -*- coding: utf-8 -*-
import cv2 as cv
#Sobel算子
def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  #x方向上的梯度
    cv.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    cv.imwrite('image/sobel.jpg', gradxy)
    cv.imshow("gradient", gradxy)

# src = cv.imread('image/1.jpg')
src = cv.imread('data/pic_0705/DSC03618.JPG')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.resizeWindow('input_image', 640, 640)
cv.imshow('input_image', src)
sobel_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
