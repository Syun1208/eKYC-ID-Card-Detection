import json
import os
import shutil
import cv2
import numpy as np
import tqdm
import argparse
import filterLabel as fl
from IPython.display import clear_output


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


def calculateBoundingBox(x_center, y_center):
    scale = 11
    xmin = x_center - scale
    xmin = np.where(xmin < 0, 0, xmin)
    ymin = y_center - scale
    ymin = np.where(ymin < 0, 0, ymin)
    xmax = x_center + scale
    xmax = np.where(xmax < 0, 0, xmax)
    ymax = y_center + scale
    ymax = np.where(ymax < 0, 0, ymax)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def normalizedPolygon(points):
    coordinate = []
    for i in range(len(points)):
        x = (points[i][0] / 100)
        y = (points[i][1] / 100)
        coordinate.append([x, y])
    coordinate = np.array(coordinate)
    return coordinate.reshape((8,))


def filterImageRotate(data, des):
    trueDes = '/home/long/Downloads/datasets/standard_datasets_YOLO/dataRightRotation'
    if not os.path.exists(trueDes):
        os.mkdir(trueDes)
    if not os.path.exists(des):
        os.mkdir(des)
    if os.path.exists(data['image']):
        for i in range(len(data['label'])):
            if data['label'][i]['polygonlabels'][0] == 'rotate':
                shutil.move(data['image'], des)
                print('Already moved ' + data['image'])
            else:
                return
    else:
        return


def filterImage(args):
    folderRotate = '/home/long/Downloads/datasets/standard_datasets_YOLO/dataRotate'
    # jsonPath = '/home/long/Downloads/datasets/images_labels.json'
    with open(args.jsonPath, 'r+') as f:
        datas = json.load(f)
    for data in tqdm.tqdm(datas, desc='Filtering label image'):
        filterImageRotate(data, folderRotate)


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
        # print('Already saved in ' + os.path.join(self.save, self.fileName + '.jpg'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(img)
        # plt.show()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderBoundingBox', type=str, help='folder saves label bounding box YOLO')
    parser.add_argument('--folderPolygon', type=str, help='folder saves label polygon YOLO')
    parser.add_argument('--folderImage', type=str, help='folder contains images')
    parser.add_argument('--imageSavePolygon', type=int, help='folder saves polygon visualization')
    parser.add_argument('--imageSaveBoundingBox', default='', help='folder saves bounding box visualization')
    parser.add_argument('--jsonPath', type=str, help='address file json label')
    return parser.parse_args()


def main(args):
    # folderBoundingBox = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsBoundingBox'
    # folderPolygon = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsPolygon'
    dictLabelBoundingBox = {'0': 'top_left', '1': 'top_right', '2': 'bottom_right', '3': 'bottom_left'}
    # jsonPath = '/home/long/Downloads/datasets/images_labels.json'
    if not os.path.exists(args.folderBoundingBox):
        os.mkdir(args.folderBoundingBox)
    if not os.path.exists(args.folderPolygon):
        os.mkdir(args.folderPolygon)
    with open(args.jsonPath, 'r+') as f:
        data = json.load(f)
    for i in tqdm.tqdm(range(len(data)), desc='Creating label'):
        clear_output(wait=True)
        nameFile = data[i]['image'].split('/')
        nameFileInformation = os.path.splitext(nameFile[len(nameFile) - 1])
        fileBoundingBox = open(os.path.join(args.folderBoundingBox, nameFileInformation[0] + '.txt'), 'w')
        filePolygon = open(os.path.join(args.folderPolygon, nameFileInformation[0] + '.txt'), 'w')
        if not data[i].get('label'):
            if os.path.exists(data[i]['image']):
                os.remove(data[i]['image'])
                continue
            else:
                continue
        for l in range(0, len(data[i]['label'])):
            for key, item in dictLabelBoundingBox.items():
                fileBoundingBox.write(key)
                fileBoundingBox.write(' ')
                '''Calculate bounding box YOLOv5'''
                coordinate = calculateBoundingBox(data[i]['label'][l]['points'][int(key)][0],
                                                  data[i]['label'][l]['points'][int(key)][1])
                coordinateYOLOv5 = calculateBoundingBoxYOLOv5(coordinate, data[i]['label'][l]['original_width'],
                                                              data[i]['label'][l]['original_height'])
                for j, value in enumerate(coordinateYOLOv5):
                    fileBoundingBox.write('{:.6f}'.format(float(value)))
                    if j >= len(coordinateYOLOv5) - 1:
                        break
                    fileBoundingBox.write(' ')
                fileBoundingBox.write('\n')
            # print('Already saved in' + os.path.join(folderBoundingBox, nameFileInformation[0] + '.txt'))
            dictLabelPolygon = {'0': 'top-cmnd', '1': 'back-cmnd', '2': 'top-cccd', '3': 'back-cccd', '4': 'top-chip',
                                '5': 'back-chip', '6': 'passport', '7': 'rotate'}
            position = list(dictLabelPolygon.values()).index(data[i]['label'][l]['polygonlabels'][0])
            key = list(dictLabelPolygon.keys())[position]
            filePolygon.write(key)
            filePolygon.write(' ')
            coordinatePolygon = normalizedPolygon(data[i]['label'][l]['points'])
            for k, value in enumerate(coordinatePolygon):
                filePolygon.write('{:.6f}'.format(float(value)))
                if k >= len(coordinatePolygon) - 1:
                    break
                filePolygon.write(' ')
            filePolygon.write('\n')
            # print('Already saved in' + os.path.join(folderPolygon, nameFileInformation[0] + '.txt'))


# def main():
#     folderYOLOv5 = '/home/long/Downloads/datasets/datasetsYOLO/labels'
#     dictLabel = {'0': 'top_left', '1': 'top_right', '2': 'bottom_right', '3': 'bottom_left'}
#     jsonPath = '/home/long/Downloads/datasets/images_labels.json'
#     if not os.path.exists(folderYOLOv5):
#         os.mkdir(folderYOLOv5)
#     with open(jsonPath, 'r+') as f:
#         data = json.load(f)
#     for i in range(len(data)):
#         nameFile = data[i]['image'].split('/')
#         nameFileInformation = os.path.splitext(nameFile[len(nameFile) - 1])
#         with open(os.path.join(folderYOLOv5, nameFileInformation[0] + '.txt'), 'w') as f:
#             if not data[i].get('label'):
#                 if os.path.exists(data[i]['image']):
#                     os.remove(data[i]['image'])
#                     continue
#                 else:
#                     continue
#             for key, item in dictLabel.items():
#                 f.write(key)
#                 f.write(' ')
#                 '''Calculate bounding box YOLOv5'''
#                 coordinate = calculateBoundingBox(data[i]['label'][0]['points'][int(key)][0],
#                                                   data[i]['label'][0]['points'][int(key)][1])
#                 coordinateYOLOv5 = calculateBoundingBoxYOLOv5(coordinate, data[i]['label'][0]['original_width'],
#                                                               data[i]['label'][0]['original_height'])
#                 for j, value in enumerate(coordinateYOLOv5):
#                     f.write('{:.6f}'.format(float(value)))
#                     if j >= len(coordinateYOLOv5) - 1:
#                         break
#                     f.write(' ')
#                 f.write('\n')
#             print('Already saved in' + os.path.join(folderYOLOv5, nameFileInformation[0] + '.txt'))


def test():
    point = [
        [
            0.0,
            0.0
        ],
        [
            100.0,
            0.0
        ],
        [
            100.0,
            100.0
        ],
        [
            0.0,
            100.0
        ]
    ]
    fileLabelTest = '/home/long/Downloads/datasets/datasetsYOLOv54Corner/labelYOLOv54Corne' \
                    'r/20211030_033753580109_cmnd_png.rf.d83b92a4df9be43900eccfe9aa79566d.txt'
    for data in point:
        coordinate = calculateBoundingBox(data[0], data[1])
        coordinateYOLOv5 = calculateBoundingBoxYOLOv5(coordinate, 1528, 1005)
        print(coordinateYOLOv5)


def createVisualizedImage(args):
    # folderLabelBoundingBox = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsBoundingBox'
    # folderLabelPolygon = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsPolygon'
    # folderImage = '/home/long/Downloads/datasets/standard_datasets_YOLO/datasets'
    # imageSavePolygon = '//home/long/Downloads/datasets/standard_datasets_YOLO/visualizedPolygon'
    # imageSaveBoundingBox = '/home/long/Downloads/datasets/standard_datasets_YOLO/visualizedBoundingBox'
    if not os.path.exists(args.imageSavePolygon):
        os.mkdir(args.imageSavePolygon)
    if not os.path.exists(args.imageSaveBoundingBox):
        os.mkdir(args.imageSaveBoundingBox)
    for file in tqdm.tqdm(os.listdir(args.folderImage), desc='Creating visualized image'):
        fileInformation = os.path.splitext(file)
        fileImage = os.path.join(args.folderImage, fileInformation[0] + '.jpg')
        fileLabelBoundingBox = os.path.join(args.folderLabelBoundingBox, fileInformation[0] + '.txt')
        fileLabelPolygon = os.path.join(args.folderLabelPolygon, fileInformation[0] + '.txt')
        visualizedYOLOv5BoundingBox = visualizedBoundingBoxYOLOv5(fileImage, fileLabelBoundingBox,
                                                                  args.imageSaveBoundingBox, fileInformation[0])
        visualizedYOLOv5BoundingBox()
        visualizedYOLOv5Polygon = visualizePolygonYOLOv5(fileImage, fileLabelPolygon, args.imageSavePolygon,
                                                         fileInformation[0])
        visualizedYOLOv5Polygon()
    print('Polygon visualization: ', len(os.listdir(args.imageSavePolygon)))
    print('Bounding box visualization: ', len(os.listdir(args.imageSaveBoundingBox)))


if __name__ == '__main__':
    # filterImage()
    args = parse_arg()
    main(args)
    fl.checkImageFile(args)
    fl.checkLabelFile(args)
    fl.checkImageWithLabel(args)
    createVisualizedImage(args)
