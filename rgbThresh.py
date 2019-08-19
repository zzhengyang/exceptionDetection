# coding:utf8
import cv2
import numpy as np
import random
import imutils
import matplotlib.pyplot as plt

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


def isHorizontalLineSimple(y1, y2):
    """ Determine if it is a horizontal line

        Args:
            linePointSet: Point coordinates at both ends of the line segment
        Returns:
            [Bool] Is this line a horizontal line
    """
    return y1 == y2


def lineTypeClassifier(line, specialAxis, centerPosition):
    # TODO
    pass


def getMeanOfCoordinate(lineSet):
    # TODO
    for line in lineSet:
        if isHorizontalLine(line):
            return


def printImg(title, img):
    """ print image to screen

        Args:
            title: image's title
            img: matrix of image
    """
    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title, 840, 840)
    cv2.imshow(title, img)


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

def cvtThresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    tmp0 = b
    tmp1 = b
    tmp0 = tmp0 - r
    tmp1 = tmp1 - g
    #gray[:, :] = 255
    rt = cv2.getTrackbarPos('r', 'image')
    gt = cv2.getTrackbarPos('g', 'image')
    bt = cv2.getTrackbarPos('b', 'image')
    gray[(b[:, :] > bt) & (r[:, :] > rt) & (g[:, :] < gt)] = 0
    return gray
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

#图片膨胀腐蚀
def erode_dilates(img):
    r = 5
    i = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=i)
    #cv.imshow('morphology', result)
    return result

def nothing(X):
    pass


def thresh(img):
    h, w = img.shape[:2]
    m = np.reshape(img, [1, w * h])
    mean = m.sum() / (w * h)
    mean = cv2.getTrackbarPos('mean', 'image')
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)

    return binary

cv2.namedWindow('image', 0)
cv2.resizeWindow('image', 840, 840)

cv2.createTrackbar('r', 'image', 200, 255, nothing)
cv2.createTrackbar('g', 'image', 174, 255, nothing)
cv2.createTrackbar('b', 'image', 131, 255, nothing)
cv2.createTrackbar('l', 'image', 1, 10, nothing)
cv2.createTrackbar('s', 'image', 3, 10, nothing)
cv2.createTrackbar('mean', 'image', 1, 255, nothing)

while(1):

    # img = cv2.imread("pic/DSC03608.JPG")
    img = cv2.imread("image/sobel.jpg")


    hisImg = s_and_b(img)

    #
    img = cvtThresh(hisImg)
    # img = thresh(gray)
    img = erode_dilates(img)
    # Read the original image
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


    cv2.imshow('image', hisImg)

cv2.destroyAllWindows()

