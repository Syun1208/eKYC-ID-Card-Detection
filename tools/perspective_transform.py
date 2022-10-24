import os
import cv2
import numpy as np


def convertBoundingBoxYOLO2COCO(img, fileTXT):
    file = open(fileTXT, 'r')
    data = file.readlines()
    file.close()
    image = cv2.imread(img)
    dh, dw, _ = image.shape
    coordinateConvert = []
    for dt in data:
        text, x, y, w, h = dt.split(' ')
        xmin = int((float(x) - float(w) / 2) * dw)
        xmax = int((float(x) + float(w) / 2) * dw)
        ymin = int((float(y) - float(h) / 2) * dh)
        ymax = int((float(y) + float(h) / 2) * dh)
        if xmin < 0:
            xmin = 0
        if xmax > dw - 1:
            xmax = dw - 1
        if ymin < 0:
            ymin = 0
        if ymax > dh - 1:
            ymax = dh - 1
        coordinateConvert.append([xmin, ymin, xmax, ymax])
    return coordinateConvert


class processingROI:
    def __init__(self, image, boundingBoxYOLO):
        self.boundingBoxYOLO = np.array(boundingBoxYOLO)
        self.image = image
        self.coordinateCenter = []

    def get_center_point(self):
        for i in range(len(self.boundingBoxYOLO)):
            x_center = (self.boundingBoxYOLO[i][0] + self.boundingBoxYOLO[i][2]) / 2
            y_center = (self.boundingBoxYOLO[i][1] + self.boundingBoxYOLO[i][3]) / 2
            self.coordinateCenter.append((x_center, y_center))
        return np.array(self.coordinateCenter)

    def order_points(self):
        rect = np.zeros((4, 2), dtype="float32")
        # self.suitable_cutting()
        s = self.get_center_point().sum(axis=1)
        rect[0] = self.get_center_point()[np.argmin(s)]
        rect[2] = self.get_center_point()[np.argmax(s)]
        diff = np.diff(self.get_center_point(), axis=1)
        rect[1] = self.get_center_point()[np.argmin(diff)]
        rect[3] = self.get_center_point()[np.argmax(diff)]
        return rect

    def _perspective_transform(self):
        (tl, tr, br, bl) = self.order_points()
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(self.order_points(), dst)
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))
        return warped

    def suitable_cutting(self):
        for i in range(len(self.get_center_point())):
            if self.get_center_point()[i][0] < 2:
                self.get_center_point()[i][0] = 0
            elif (self.get_center_point()[i] > (self.image.shape[1] - 10, self.image.shape[0] - 10)).all():
                self.get_center_point()[i] = (self.image.shape[1], self.image.shape[0] - 10)
            elif self.get_center_point()[i][0] > self.image.shape[1] - 10 and self.get_center_point()[i][1] < 2:
                self.get_center_point()[i] = (self.image.shape[1], 0)
            elif self.get_center_point()[i][0] < 2 and self.get_center_point()[i][1] > self.image.shape[0] - 10:
                self.get_center_point()[i] = (0, self.image.shape[0])

    def __call__(self, *args, **kwargs):
        return self._perspective_transform()


if __name__ == "__main__":
    folder = '/home/long/Pictures/295925243_835662654456053_3879872674587273906_n.png'
    folderLabel = '/home/long/Downloads/datasets/datasetsYOLO/labels/train'
    for imageFile in os.listdir(folder):
        informationFile = os.path.splitext(imageFile)
        fileLabel = os.path.join(folderLabel, informationFile[0] + '.txt')
        print('Start reading ' + fileLabel)
        coordinateCOCO = convertBoundingBoxYOLO2COCO(os.path.join(folder, imageFile), fileLabel)
        processing = processingROI(os.path.join(folder, imageFile), coordinateCOCO)
        processing()
