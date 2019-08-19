# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
from scipy import misc, ndimage
import imutils
from scipy.spatial import distance as dist
import random
'''
roteImg(Img) : 将图片矫正至水平位置
                返回三通道图片
RGBtoHIS(Img) : 将图片颜色空间从rgb转换到his
                返回三通道图
erode_dilates(Img) ：将图片膨胀腐蚀（开运算）
                返回二值图
find_rect(Img) ：找到图片中轮廓最大的一块区域用矩形框住
                返回二值图
computeRadis(img, x1, x2, y1, y2, flag = True) ：计算直径，x1,x2表示高度区间，y1,y2表示宽度区间，
                                                 flag等于true表示计算竖直方向，反之计算水平方向
                                                 返回长度
simi(readyImg, eps) ：根据余弦相似度，结合阈值将图片二值化
                      返回二值图
'''
#旋转矫正
def roteImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if x1 == x2 or y1 == y2:
            continue
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        rotate_img = ndimage.rotate(img, rotate_angle)
        return rotate_img

#从RGB转HIS
def RGBtoHIS(image):
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
    # 通道0是色相、通道1是亮度、通道2是饱和度
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
  #  lsImg = np.zeros(image.shape, np.float32)
    hlsCopy = np.copy(hlsImg)
    h = 0.6
    l = 1.0
    s = 3.0#亮度饱和度
    hlsCopy[:, :, 0] = h * hlsCopy[:, :, 0]
    hlsCopy[:, :, 0][hlsCopy[:, :, 0] > 1] = 1
    # 1.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
    hlsCopy[:, :, 1] = l * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
    # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
    hlsCopy[:, :, 2] = s * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # 显示调整后的效果
    return hlsCopy * 255

# #图片二值化
# def cvtThresh(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     b, g, r = cv2.split(image)
#     tmp0 = b
#     tmp1 = b
#     tmp0 = tmp0 - r
#     tmp1 = tmp1 - g
#     #gray[:, :] = 255
#     gray[(b[:, :] > 200) & (r[:, :] > 200) & (g[:, :] < 200)] = 0
#     return gray

def erode_dilates(img):
    r = 5
    i = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=i)
    #cv.imshow('morphology', result)
    return result

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#获取外接矩形
def find_rect(processImg, srcImg):
    img_gray = cv2.Canny(processImg, 127, 255)
    ret, img_gray = cv2.threshold(img_gray, 127, 255, 0)
    _, contours, hie = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # imgp = cv2.imread(srcImg)
    imgp = img_gray.copy()
    maxSquare = -1
    cache = ()

    for c in contours[1:]:
        x, y, w, h = cv2.boundingRect(c)  # 外接矩形

        cv2.rectangle(srcImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.namedWindow("enhanced", 0)
        cv2.resizeWindow("enhanced", 640, 480)
        cv2.imshow("enhanced", srcImg)

        if w * h > maxSquare:
            maxSquare = w * h
            cache = (x, y, w, h)
    img10 = imgp[cache[1] : cache[1] + cache[3], cache[0] : cache[0] + cache[2]]
    return img10, cache[3], cache[2]

#计算直径
def computeRadis(img, x1, x2, y1, y2, flag = True):
    maxlen = -1
   # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if flag:#外循环是宽
        for width in range(y1, y2):
            tmplen = 0
            for heigh in range(x1, x2):
                if img[heigh, width, 0] == 0:
                    tmplen = tmplen + 1
                else:
                    if tmplen > maxlen:
                        maxlen = tmplen
                    tmplen = 0
            if tmplen > maxlen:
                maxlen = tmplen
    else:#外循环是高
        for heigh in range(y1, y2):
            tmplen = 0
            for width in range(x1, x2):
                if img[heigh, width, 0] == 0:
                    tmplen = tmplen + 1
                else:
                    if tmplen > maxlen:
                        maxlen = tmplen
                    tmplen = 0
            if tmplen > maxlen:
                maxlen = tmplen
    #return maxlen * pix
    return maxlen * 0.0525

#传入原始图片和相似度阈值（余弦相似度）
def simi(readyImg, eps):
    img = cv2.imread("data/pic/model.jpg")
    #图片白平衡
    b, g, r = cv2.split(img)#通道分离
    meanB = cv2.mean(b)[0]#获取RGB分量的均值
    meanG = cv2.mean(g)[0]
    meanR = cv2.mean(r)[0]

    grayImg = cv2.cvtColor(readyImg, cv2.COLOR_BGR2GRAY)
    _, threshImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    readyImg = readyImg * 1.0
    #print(meanR, meanG, meanB)

    threshImg = threshImg * 0
    #print(threshImg.shape)
    threshImg[:, :][((meanB * readyImg[:, :, 0] + meanG * readyImg[:, :, 1] + meanR * readyImg[:, :, 2])
        * (meanB * readyImg[:, :, 0] + meanG * readyImg[:, :,1] + meanR * readyImg[:, :, 2])
        / (meanB * meanB + meanG * meanG + meanR * meanR)
        /(readyImg[:, :, 0] * readyImg[:, :, 0] + readyImg[:, :, 1] * readyImg[:, :, 1] + readyImg[:, :, 2] * readyImg[:, :, 2])) > eps] = 255

    #cv2.imwrite("456.jpg", threshImg)
    # h = 1320
    # w = 2350
    # print((meanB * readyImg[h, w, 0] + meanG * readyImg[h, w, 1] + meanR * readyImg[h, w, 2])
    #     * (meanB * readyImg[h, w, 0] + meanG * readyImg[h, w,1] + meanR * readyImg[h, w, 2])
    #     / (meanB * meanB + meanG * meanG + meanR * meanR)
    #     /(readyImg[h, w, 0] * readyImg[h, w, 0] + readyImg[h, w, 1] * readyImg[h, w, 1] + readyImg[h, w, 2] * readyImg[h, w, 2]))
    return threshImg

def findCenter(srcImage):
    """ return center of the shape of the image

        Args:
            srcImage: source image
            desImage: final picture

        Returns:
            cX: x of center point's coordinate
            cY: y of center point's coordinate
    """
    cnts = cv2.findContours(srcImage.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return cX, cY

def randomColorGen():
    """ Generate random color

        Returns:
            random color tuple, example: (0, 255, 145)

    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def isHorizontalLine(linePointSet):
    """ Determine if it is a horizontal line

        Args:
            linePointSet: Point coordinates at both ends of the line segment
        Returns:
            [Bool] Is this line a horizontal line
    """
    return any(abs(y1-y2)<20 for x1, y1, x2, y2 in linePointSet)

def printImg(title, img):
    """ print image to screen

        Args:
            title: image's title
            img: matrix of image
    """
    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title, 840, 840)
    cv2.imshow(title, img)
