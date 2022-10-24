import sys
from pathlib import Path
from tools.perspective_transform import processingROI, convertBoundingBoxYOLO2COCO
import numpy as np
import imutils
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from plyer import notification

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def calculateAngleLessThan90DegreesRotation(image, point):  # [(x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)]
    for coordinate in point:
        pointTopLeft = coordinate[0]
        pointBottomLeft = coordinate[-1]
        distance = np.sqrt(
            np.power((pointBottomLeft[1] - pointTopLeft[1]), 2) + np.power((pointBottomLeft[0] - pointTopLeft[0]), 2))
        '''Compute the distance between top_left's points and the image's width'''
        y = image.shape[1]
        perpendicularDistances = np.abs(pointTopLeft[1] - y)
        alpha = np.arctan2(distance, perpendicularDistances)
        return alpha


class rotationBaseOn4CornersYOLO(processingROI):
    def __init__(self, image, boundingBoxYOLO):
        super(rotationBaseOn4CornersYOLO, self).__init__(image, boundingBoxYOLO)
        self.image = image
        # self.address = os.path.splitext(image.split('/')[-1])
        self.boundingBoxYOLO = boundingBoxYOLO
        self.vectorOx = (self.image.shape[1], 0)
        self.angle = 0

    def _computeAngle(self, vectorTopLeftTopRight):
        vectorTopLeftTopRight = vectorTopLeftTopRight / np.linalg.norm(vectorTopLeftTopRight)
        self.vectorOx = self.vectorOx / np.linalg.norm(self.vectorOx)
        dot_product = np.dot(vectorTopLeftTopRight, self.vectorOx)
        self.angle = np.rad2deg(np.arccos(dot_product))
        return self.angle

    def _processingCorner(self):
        coordinateCenter = self.get_center_point()  # [(x_tl,y_tl), (x_tr,y_tr)]
        vectorTopLeftTopRight = (
            coordinateCenter[1][0] - coordinateCenter[0][0], coordinateCenter[1][1] - coordinateCenter[0][1])
        self.angle = self._computeAngle(vectorTopLeftTopRight)
        if np.abs(coordinateCenter[1][1] - coordinateCenter[0][1]) > np.abs(
                coordinateCenter[1][0] - coordinateCenter[0][0]) and (
                coordinateCenter[1][0] - coordinateCenter[0][0] <= 24 and coordinateCenter[1][1] - coordinateCenter[0][1] < 0):
            return self.angle + 180
        else:
            return self.angle

    def __call__(self, *args, **kwargs):
        angle = self._processingCorner()
        rotated = imutils.rotate_bound(self.image, -angle)
        print("[INFO] ROTATION: {:.3f}".format(angle))
        notification.notify(
            title='[INFO] ROTATION',
            message=str(angle) + ' DEGREES',
            timeout=5
        )
        return rotated


if __name__ == '__main__':
    imageFile = "/home/long/Downloads/datasets/datasetsRotation/3case/270"
    folderLabel = '/home/long/Downloads/datasets/datasetsRotation/labelsBoundingBoxRotation'
    for file in os.listdir(imageFile):
        fileInfor = os.path.splitext(file)
        coordinateCOCO = convertBoundingBoxYOLO2COCO(os.path.join(imageFile, file),
                                                     os.path.join(folderLabel, fileInfor[0] + '.txt'))
        print(coordinateCOCO)
        coordinateCOCONew = list([coordinateCOCO[-1], coordinateCOCO[0]])
        print(coordinateCOCONew)
        rotate = rotationBaseOn4CornersYOLO(os.path.join(imageFile, file), coordinateCOCONew)
        rotate()
