# coding:utf8
import cv2 as cv
#拉普拉斯算子
def Laplace_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("Laplace_demo", lpls)
src = cv.imread('data/pic_0705/DSC03618.JPG')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
Laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()