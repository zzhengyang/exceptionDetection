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

def nothing(X):
    pass

cv2.namedWindow('image', 0)
cv2.resizeWindow('image', 840, 840)

cv2.createTrackbar('gauss1', 'image', 7, 10, nothing)
cv2.createTrackbar('gauss2', 'image', 5, 10, nothing)
cv2.createTrackbar('canny1', 'image', 400, 500, nothing)
cv2.createTrackbar('canny2', 'image', 390, 500, nothing)
cv2.createTrackbar('erode', 'image', 1, 10, nothing)
cv2.createTrackbar('dilate', 'image', 1, 10, nothing)
cv2.createTrackbar('denoise', 'image', 1, 10, nothing)
cv2.createTrackbar('mKernel', 'image', 1, 10, nothing)


while(1):

    img = cv2.imread("data/pic_0704/DSC03608.JPG")
    # img = cv2.imread("data/pic/erodeImg.jpg")

    # Read the original image
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    gauss1 = cv2.getTrackbarPos('gauss1', 'image')
    gauss1 = (gauss1 - 1) if gauss1 % 2 == 0 else gauss1

    gauss2 = cv2.getTrackbarPos('gauss2', 'image')
    gauss2 = (gauss2 - 1) if gauss2 % 2 == 0 else gauss2

    canny1 = cv2.getTrackbarPos('canny1', 'image')
    canny2 = cv2.getTrackbarPos('canny2', 'image')

    erodeKernel = cv2.getTrackbarPos('erode', 'image')
    erodeKernel = (erodeKernel - 1) if erodeKernel % 2 == 0 else erodeKernel
    dilateKernel = cv2.getTrackbarPos('dilate', 'image')
    dilateKernel = (dilateKernel - 1) if dilateKernel % 2 == 0 else dilateKernel
    denoiseH = cv2.getTrackbarPos('denoise', 'image')

    mKernel = cv2.getTrackbarPos('mKernel', 'image')
    mKernel = (mKernel - 1) if mKernel % 2 == 0 else mKernel


    # Gaussian filter using 7*7 kernel
    gray = cv2.GaussianBlur(img, (gauss1, gauss1), 0)

    # Canny edge detection
    edges = cv2.Canny(gray, canny1, canny2)

    # Gaussian filter again using 3*3 kernel
    edges = cv2.GaussianBlur(edges, (gauss2, gauss2), 0)

    # Open operation
    edgesKernel = np.uint8(np.ones((erodeKernel, erodeKernel)))
    dilatesKernel = np.uint8(np.ones((dilateKernel, dilateKernel)))
    edges = cv2.erode(edges, edgesKernel, iterations=1)
    edges = cv2.dilate(edges, dilatesKernel, iterations=1)

    cv2.imshow('image', edges)

cv2.destroyAllWindows()

