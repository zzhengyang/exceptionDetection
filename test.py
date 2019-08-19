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

#图片二值化
def cvtThresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    tmp0 = b
    tmp1 = b
    tmp0 = tmp0 - r
    tmp1 = tmp1 - g
    #gray[:, :] = 255
    gray[(b[:, :] > 62) & (r[:, :] > 251) & (g[:, :] < 205)] = 0
    return gray

#图片膨胀腐蚀
def erode_dilates(img):
    r = 5
    i = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=i)
    #cv.imshow('morphology', result)
    return result


#从RGB转HIS
def s_and_b(image):
    image = image.copy()
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


# Read the original image
img = cv2.imread("data/pic_0705/DSC03618.JPG")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Gaussian filter using 7*7 kernel
gray = cv2.GaussianBlur(img, (7, 7), 0)

# Canny edge detection
edges = cv2.Canny(gray, 407, 398)

# Gaussian filter again using 5*5 kernel
edges = cv2.GaussianBlur(edges, (5, 5), 0)

# Open operation
NpKernel = np.uint8(np.ones((1,1)))
edges = cv2.erode(edges, NpKernel, iterations=1)
edges = cv2.dilate(edges, NpKernel, iterations=1)

# Hough transform
minLineLength = 200
maxLineGap = 200
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 300, minLineLength, maxLineGap)

# Get the center of the shape on the image
centerX, centerY = findCenter(edges)
print "center: x[%d], y[%d]" % (centerX, centerY)

topLineSet = []
bottomLineSet = []
leftLineSet = []
rightLineSet = []
topAndBottomLinePixelDiffArr = []
leftAndRightLinePixelDiffArr = []

tyMean = 0
byMean = 0
lxMean = 0
rxMean = 0
i = 1
print lines
for line in lines:
    print i, line
    if isHorizontalLine(line):
        if line[0][1] < centerY:
            topLineSet.append(line)
        else:
            bottomLineSet.append(line)
    else:
        if line[0][0] < centerX:
            leftLineSet.append(line)
        else:
            rightLineSet.append(line)
    for x1, y1, x2, y2 in line:
        # draw lines on the picture
        cv2.line(img, (x1, y1), (x2, y2), randomColorGen(), 5)

        cv2.putText(img, "{:.0f}".format(i),
                    (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (220, 20, 60), 2)
    i += 1

# Get mean gap pixel of horizeontal lines
for topLine in topLineSet:
    topLeftX, topLeftY, topRightX, topRightY = topLine[0]
    for bottomLine in bottomLineSet:
        bottomLeftX, bottomLeftY, bottomRightX, bottomRightY = bottomLine[0]
        if not (topRightX < bottomLeftX or topLeftX > bottomRightX):
            topAndBottomLinePixelDiffArr.append(bottomRightY - topRightY)

meanGapPixelOfHorizontalLine = np.mean(topAndBottomLinePixelDiffArr)
print "Mean gap pixel of horizontal lines is " + str(meanGapPixelOfHorizontalLine)


cv2.circle(img, (centerX, centerY), 10, (220, 20, 60), -1)
cv2.putText(img, "center", (centerX - 20, centerY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (220, 20, 60), 2)

# show the final picture
printImg("edges", edges)
printImg("lines", img)


# # statistic analysis
#
# horizonGapDict = []
# for topLine in topLineSet:
#     topLeftX, topLeftY, topRightX, topRightY = topLine[0]
#     horizonGapDict += [abs(topRightY-centerY)*2]*abs(topRightX-topLeftX)
#
# plt.subplot(121)
# plt.hist(horizonGapDict)
# plt.title('horizontal')
#
# verticleGapDict = []
# for leftLine in leftLineSet:
#     leftTopX, leftTopY, leftBottomX, leftBottomY = leftLine[0]
#     verticleGapDict += [abs(leftTopX-centerX)*2]*abs(leftBottomY-leftTopY)
#     print abs(leftTopX-centerX)*2
# plt.subplot(122)
# plt.hist(verticleGapDict)
# plt.title('verticle')
#
# plt.show()

pixelsPerMetric = 564 / 70.02
print 1132 / pixelsPerMetric
print abs(140.34 - 1132 / pixelsPerMetric)



cv2.waitKey(0)
