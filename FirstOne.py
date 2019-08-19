# -*- coding: utf-8 -*-
import cv2
import numpy as np

#从RGB转HIS
def s_and_b(image):
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
    # 通道0是色相、通道1是亮度、通道2是饱和度
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
  #  lsImg = np.zeros(image.shape, np.float32)
    hlsCopy = np.copy(hlsImg)
    l = 1.0
    s = 5.0#亮度饱和度
    # 1.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
    hlsCopy[:, :, 1] = l * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
    # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
    hlsCopy[:, :, 2] = s * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # 显示调整后的效果
    return hlsCopy * 255

#图片二值化
def cvtThresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    tmp0 = b
    tmp1 = b
    tmp0 = tmp0 - r
    tmp1 = tmp1 - g
    #gray[:, :] = 255
    gray[(b[:, :] > 200) & (r[:, :] > 200) & (g[:, :] < 200)] = 0
    return gray

#图片膨胀腐蚀
def erode_dilates(img):
    r = 5
    i = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=i)
    #cv.imshow('morphology', result)
    return result


#获取外接矩形
def find_rect(img):
    img = cv2.imread("pic/erode_img.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray = cv2.threshold(img_gray, 127, 255, 0)
    _, contours, hie = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img10 = cv2.imread("pic/erode_img.jpg")
    maxSquare = -1
    cache = ()
    for c in contours[1:]:
        x, y, w, h = cv2.boundingRect(c)  # 外接矩形
        #cv.rectangle(img10, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w * h > maxSquare:
            maxSquare = w * h
            cache = (x, y, w, h)
    img10 = img10[cache[1] : cache[1] + cache[3], cache[0] : cache[0] + cache[2]]
    return img10

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
    return maxlen * pix

fn = "image/1.jpg"#原始图片
ori_image = cv2.imread(fn, 1)
his_img = s_and_b(ori_image)#原图颜色空间变化
cv2.imwrite("pic/his_img.jpg", his_img)

#图片二值化
bin_img = cvtThresh(his_img)
cv2.imwrite("pic/bin_img.jpg", bin_img)

#膨胀腐蚀
erode_img = erode_dilates(bin_img)
cv2.imwrite("pic/erode_img.jpg", erode_img)

#计算半径
rect_img = find_rect(erode_img)
cv2.imwrite("pic/rect_img.jpg", rect_img)
h = int(rect_img.shape[0])
w = int(rect_img.shape[1])
midHeight = int(h / 2)
midWidth = int(w / 2)
pix = 0.0467
offset = 50

left_top_vertical = computeRadis(rect_img, offset, midHeight, offset, midWidth)
left_top_horizon = computeRadis(rect_img, offset, midWidth, offset, midHeight, flag=False)

left_bottom_vertical = computeRadis(rect_img, midHeight, h - offset, offset, midWidth)
left_bottom_horizon = computeRadis(rect_img, offset, midWidth, midHeight, h - offset, flag=False)

right_top_vertical = computeRadis(rect_img, offset, midHeight, midWidth, w - offset)
right_top_horizon = computeRadis(rect_img, midWidth, w - offset, offset, midHeight, flag=False)

right_bottom_vertical = computeRadis(rect_img, midHeight, h - offset, midWidth, w - offset)
right_bottom_horizon = computeRadis(rect_img, midWidth, w - offset, midHeight, h - offset, flag=False)

print("左上垂直直径：", left_top_vertical)
print("左上水平直径：", left_top_horizon)

print("左下垂直直径：", left_bottom_vertical)
print("左下水平直径：", left_bottom_horizon)

print("右上垂直直径：", right_top_vertical)
print("右上水平直径：", right_top_horizon)

print("右下垂直直径：", right_bottom_vertical)
print("右下水平直径：", right_bottom_horizon)
#img = cv2.imread("161718.jpg")
# #边缘检测
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# edges = cv2.Canny(gray, 100, 150)
# edges = cv2.GaussianBlur(edges, (7, 7), 0)
# edges = cv2.erode(edges, None, iterations=1)
# edges = cv2.dilate(edges, None, iterations=1)
# cv2.imwrite("101112.jpg", edges)