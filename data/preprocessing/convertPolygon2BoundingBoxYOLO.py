import os
import shutil
import cv2
import numpy as np
import tqdm
import argparse
from pathlib import Path
import sys
import filterLabel as fl
from IPython.display import clear_output

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


class visualizePolygonYOLOv5:
    def __init__(self, image, fileTXT, save, fileName):
        self.image = image
        self.fileTXT = fileTXT
        self.fileName = fileName
        self.save = save
        self.dictLabel = {'0': 'top-cmnd', '1': 'back-cmnd', '2': 'top-cccd', '3': 'back-cccd', '4': 'top-chip',
                          '5': 'back-chip', '6': 'passport', '7': 'rotate'}

    def __call__(self, *args, **kwargs):
        # print('Start with ' + self.fileTXT)
        img = cv2.imread(self.image)
        dh, dw, _ = img.shape
        file = open(self.fileTXT, 'r')
        data = file.readlines()
        file.close()
        for dt in data:
            text, x1, y1, x2, y2, x3, y3, x4, y4 = dt.split(' ')
            points = np.array([[[float(x1) * dw, float(y1) * dh], [float(x2) * dw, float(y2) * dh],
                                [float(x3) * dw, float(y3) * dh], [float(x4) * dw, float(y4) * dh]]], np.int32)
            cv2.putText(img, self.dictLabel[text], (int(float(x1) * dw), int(float(y1) * dh)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), thickness=2)
            img = cv2.polylines(img, [points], True, (0, 255, 0), thickness=3)
        cv2.imwrite(os.path.join(self.save, self.fileName + '.jpg'), img)
        # print('Already saved in ' + os.path.join(self.save, self.fileName + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class visualizedBoundingBoxYOLOv5:
    def __init__(self, image, fileTXT, save, fileName):
        self.image = image
        self.fileTXT = fileTXT
        self.fileName = fileName
        self.save = save
        self.dictLabel = {'0': 'top_left', '1': 'top_right', '2': 'bottom_right', '3': 'bottom_left'}

    def __call__(self, *args, **kwargs):
        # print('Start with ' + self.fileTXT)
        img = cv2.imread(self.image)
        dh, dw, _ = img.shape
        file = open(self.fileTXT, 'r')
        data = file.readlines()
        file.close()
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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
            cv2.putText(img, self.dictLabel[text], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.imshow('Image Visualization', img)
        cv2.imwrite(os.path.join(self.save, self.fileName + '.jpg'), img)


def calculateBoundingBoxYOLOv5(point, width, height):
    xmin = (point[0] / 100) * width
    ymin = (point[1] / 100) * height
    xmax = (point[2] / 100) * width
    ymax = (point[3] / 100) * height
    xCentral = ((xmin + xmax) / 2) / width
    yCentral = ((ymin + ymax) / 2) / height
    widthScale = (xmax - xmin) / width
    heightScale = (ymax - ymin) / height
    return [float(xCentral), float(yCentral), float(widthScale), float(heightScale)]


def calculateBoundingBox(x_center, y_center, width, height):
    scale = 10
    x_center = x_center * 100
    y_center = y_center * 100
    xmin = x_center - scale
    # xmin = np.where(xmin < 0, 0, xmin)
    ymin = y_center - scale
    # ymin = np.where(ymin < 0, 0, ymin)
    xmax = x_center + scale
    # xmax = np.where(xmax < 0, 0, xmax)
    ymax = y_center + scale
    # ymax = np.where(ymax < 0, 0, ymax)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def main():
    folder_label = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/labels'
    folder_image = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/images'
    folder_padding_image = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/augment_padding_images'
    folder_save = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/labels_yolo'
    dictLabelBoundingBox = {'0': 'top_left', '1': 'top_right', '2': 'bottom_right', '3': 'bottom_left'}
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    for file_image in tqdm.tqdm(os.listdir(folder_image), total=len(os.listdir(folder_image)),
                                desc='Re-calculate bounding box YOLO'):
        file_label = os.path.join(folder_label, os.path.splitext(file_image)[0] + '.txt')
        image = cv2.imread(os.path.join(folder_image, file_image))
        file_label = open(file_label, 'r+')
        list_file_label = file_label.readlines()[0].split(' ')
        coordinate_polygon = {'0': [float(list_file_label[1]), float(list_file_label[2])],
                              '1': [float(list_file_label[3]), float(list_file_label[4])],
                              '2': [float(list_file_label[5]), float(list_file_label[6])],
                              '3': [float(list_file_label[7]), float(list_file_label[8])]}
        fileBoundingBox = open(os.path.join(folder_save, os.path.splitext(file_image)[0] + '.txt'), 'w')
        for key, item in dictLabelBoundingBox.items():
            fileBoundingBox.write(key)
            fileBoundingBox.write(' ')
            '''Calculate bounding box YOLOv5'''
            coordinate = calculateBoundingBox(coordinate_polygon[key][0], coordinate_polygon[key][1], image.shape[1],
                                              image.shape[0])
            coordinateYOLOv5 = calculateBoundingBoxYOLOv5(coordinate, image.shape[1], image.shape[0])
            for j, value in enumerate(coordinateYOLOv5):
                fileBoundingBox.write('{:.6f}'.format(float(value)))
                if j >= len(coordinateYOLOv5) - 1:
                    break
                fileBoundingBox.write(' ')
            fileBoundingBox.write('\n')


def createVisualizedImage():
    imageSaveBoundingBox = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/visualization'
    folderImage = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/augment_padding_images'
    folderLabelBoundingBox = '/root/card-transformation/IDCardDetectionAndRecognition/data/datasets/labels_yolo'
    if not os.path.exists(imageSaveBoundingBox):
        os.makedirs(imageSaveBoundingBox)
    for file in tqdm.tqdm(os.listdir(folderImage), desc='Creating visualized image'):
        fileInformation = os.path.splitext(file)
        fileImage = os.path.join(folderImage, file)
        fileLabelBoundingBox = os.path.join(folderLabelBoundingBox, fileInformation[0] + '.txt')
        visualizedYOLOv5BoundingBox = visualizedBoundingBoxYOLOv5(fileImage, fileLabelBoundingBox,
                                                                  imageSaveBoundingBox, fileInformation[0])
        visualizedYOLOv5BoundingBox()
    # print('Polygon visualization: ', len(os.listdir(args.imageSavePolygon)))
    print('Bounding box visualization: ', len(os.listdir(imageSaveBoundingBox)))


if __name__ == '__main__':
    main()
    createVisualizedImage()
