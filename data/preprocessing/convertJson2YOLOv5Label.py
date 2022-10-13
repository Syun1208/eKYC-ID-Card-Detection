import json
import os
import shutil
import cv2
import numpy as np
import argparse
import filterLabel as fl
import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing


# Classes
# names:
#   0: top-cmnd
#   1: back-cmnd
#   2: top-cccd
#   3: back-cccd
#   4: top-chip
#   5: back-chip
#   6: passport
#   7: rotate
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderBoundingBox', type=str, help='folder saves label bounding box YOLO')
    parser.add_argument('--folderPolygon', type=str, help='folder saves label polygon YOLO')
    parser.add_argument('--folderImage', type=str, help='folder contains images')
    parser.add_argument('--imageSavePolygon', type=int, help='folder saves polygon visualization')
    parser.add_argument('--imageSaveBoundingBox', default='', help='folder saves bounding box visualization')
    parser.add_argument('--jsonPath', type=str, help='address file json label')
    return parser.parse_args()


def calculateBoundingBoxYOLOv5(point, width, height):
    firstPoint = (point[0][0] + point[3][0]) / 2
    secondPoint = (point[0][1] + point[1][1]) / 2
    thirdPoint = (point[1][0] + point[2][0]) / 2
    fourthPoint = (point[2][1] + point[3][1]) / 2
    xminAverage = (firstPoint / 100) * width
    yminAverage = (secondPoint / 100) * height
    xmaxAverage = (thirdPoint / 100) * width
    ymaxAverage = (fourthPoint / 100) * height
    xCentral = ((xminAverage + xmaxAverage) / 2) / width
    yCentral = ((yminAverage + ymaxAverage) / 2) / height
    widthScale = (xmaxAverage - xminAverage) / width
    heightScale = (ymaxAverage - yminAverage) / height
    return [abs(xCentral), abs(yCentral), abs(widthScale), abs(heightScale)]


class visualizedBoundingBoxYOLOv5:
    def __init__(self, image, fileTXT, save, fileName):
        self.image = image
        self.fileTXT = fileTXT
        self.fileName = fileName
        self.save = save
        self.dictLabel = {'0': 'top-cmnd', '1': 'back-cmnd', '2': 'top-cccd', '3': 'back-cccd', '4': 'top-chip',
                          '5': 'back-chip', '6': 'passport', '7': 'rotate'}

    def __call__(self, *args, **kwargs):
        img = cv2.imread(self.image)
        dh, dw, _ = img.shape
        fl = open(self.fileTXT, 'r')
        data = fl.readlines()
        fl.close()
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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, self.dictLabel[text], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.imshow('Image Visualization', img)
        cv2.imwrite(os.path.join(self.save, self.fileName + '.jpg'), img)
        # clear_output(wait=True)
        # print('Already saved in ' + os.path.join(self.save, self.fileName + '.jpg'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(img)
        # plt.show()


def main(args):
    dictLabel = {'0': 'top-cmnd', '1': 'back-cmnd', '2': 'top-cccd', '3': 'back-cccd', '4': 'top-chip',
                 '5': 'back-chip', '6': 'passport', '7': 'rotate'}
    with open(args.jsonPath, 'r+') as f:
        data = json.load(f)
    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        nameFile = data[i]['image'].split('/')
        nameFileInformation = os.path.splitext(nameFile[len(nameFile) - 1])
        with open(os.path.join(args.folderBoundingBox, nameFileInformation[0] + '.txt'), 'w') as f:
            if len(data[i]) != 8:
                if os.path.exists(data[i]['image']):
                    os.remove(data[i]['image'])
                    continue
                else:
                    continue
            position = list(dictLabel.values()).index(data[i]['label'][0]['polygonlabels'][0])
            data[i]['label'][0]['polygonlabels'] = list(dictLabel.keys())[position]
            f.write(str(data[i]['label'][0]['polygonlabels']))
            f.write(' ')
            '''Calculate Bounding Box YOLOv5'''
            coordinate = np.array(calculateBoundingBoxYOLOv5(data[i]['label'][0]['points'],
                                                             data[i]['label'][0]['original_width'],
                                                             data[i]['label'][0]['original_height']))
            # originalCoordinate = [data[i]['label'][0]['original_width'],
            #                       data[i]['label'][0]['original_height']]
            # listNormalize = np.array([coordinate[0], coordinate[1], data[i]['label'][0]['original_width'],
            #                           data[i]['label'][0]['original_height']])
            # scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.99))
            # normalizedList = scaler.fit_transform(listNormalize).reshape(4, )
            count = 0
            for value in coordinate:
                if value > 1:
                    value = 1
                f.write(str('{:.6f}'.format(float(value))))
                if count == len(coordinate) - 1:
                    break
                f.write(' ')
                count += 1
            # for c in coordinate:
            #     f.write(str('{:.6f}'.format(float(c))))
            #     f.write(' ')
            # for label in originalCoordinate:
            #     f.write(str('{:.6f}'.format(float(label))))
            #     f.write(' ')
            f.close()
            # clear_output(wait=True)
            # print('Already saved in' + os.path.join(args.folderBoundingBox, nameFileInformation[0] + '.txt'))


def createVisualizedImage(args):
    for folder in tqdm.tqdm(os.listdir(args.folderImage), total=len(args.folderImage)):
        fileInformation = os.path.splitext(folder)
        fileImage = os.path.join(args.folderImage, fileInformation[0] + '.jpg')
        fileLabel = os.path.join(args.folderBoundingBox, fileInformation[0] + '.txt')
        visualizedYOLOv5 = visualizedBoundingBoxYOLOv5(fileImage, fileLabel, args.imageSaveBoundingBox,
                                                       fileInformation[0])
        visualizedYOLOv5()
    print('Bounding box visualization: ', len(os.listdir(args.imageSaveBoundingBox)))


if __name__ == '__main__':
    args = parse_arg()
    main(args)
    fl.checkImageFile(args)
    fl.checkLabelFile(args)
    fl.checkImageWithLabel(args)
    createVisualizedImage(args)
    # for folder in os.listdir(args.folderImage):
    #     fileInformation = os.path.splitext(folder)
    #     fileImage = os.path.join(args.folderImage, fileInformation[0] + '.jpg')
    #     fileLabel = os.path.join(args.folderBoundingBox, fileInformation[0] + '.txt')
    #     visualizedYOLOv5 = visualizedBoundingBoxYOLOv5(fileImage, fileLabel, args.imageSaveBoundingBox,
    #                                                    fileInformation[0])
    #     visualizedYOLOv5()
