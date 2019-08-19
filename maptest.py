import numpy as np
a = [[329,672,1039,672],[265,109,1112,109],[99,55,104,259],[1223,21,1228,533]]


def isHorizontalLine(linePointSet):
    return any(y1 == y2 for x1, y1, x2, y2 in linePointSet)

def getMean():
    return np.mean()

def mapFunctionTest():
    for i in a:
        if isHorizontalLine(a):
            print i

mapFunctionTest()
