import os
import sys
import time
import numpy as np
import argparse
import cv2
import tqdm
import logging
import io
import matplotlib.pyplot as plt
from PIL import Image
from tabulate import tabulate
from plyer import notification
import matplotlib.image as mpimg
from weights.load_weights import weights
from deploy.get_coordinate_yolo import detection
from tools.non_max_suppression import nms
from tools.perspective_transform import processingROI
from deploy.rotation import rotationBaseOn4CornersYOLO
from deploy.reduce_blur import reduceBlur
from pathlib import Path

# ROOT = os.path.dirname(os.path.realpath('__file__'))
# ROOT = str(Path.home())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    i = 0
    j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1


def mergeSort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2

        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path', default='weights/yolov7x/yolov7x.pt')
    parser.add_argument('--cfg-detection', type=str, help='model configuration: yolov5, yolov7', default='yolov7')
    parser.add_argument('--img_path', type=str, help='Image path')
    parser.add_argument('--folder_path', type=str, help='Folder Image path')
    parser.add_argument('--folder_save_rotation', type=str,
                        default=str(ROOT / 'results/correct'),
                        required=False)
    parser.add_argument('--folder_save_detection', type=str,
                        default=str(ROOT / 'results/detect'),
                        required=False)
    parser.add_argument('--option', type=int, help='activate 1 to open camera or 0 to add image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


def convertBoundingBox2Polygon(coordinates):
    coordinatePolygon = {}
    coordinateVisualize = []
    classes = {'top_left': ['x_min', 'y_min'], 'top_right': ['x_max', 'y_min'], 'bottom_right': ['x_max', 'y_max'],
               'bottom_left': ['x_min', 'y_max']}
    # classes = {'top_left': 0, 'top_right': 1, 'bottom_right': 2, 'bottom_left': 3}
    for i in range(len(coordinates)):
        x_center = (coordinates[i][0] + coordinates[i][2]) / 2
        y_center = (coordinates[i][1] + coordinates[i][3]) / 2
        coordinateVisualize.append((x_center, y_center))
        coordinatePolygon[coordinates[i][-1]] = {classes[coordinates[i][-1]][0]: x_center,
                                                 classes[coordinates[i][-1]][1]: y_center}
    coordinatePolygon = {'top_left': coordinatePolygon['top_left', 'top_right': coordinatePolygon['top_right'],
                                     'bottom_right': coordinatePolygon['bottom_right'],
                                     'bottom_left': coordinatePolygon['bottom_left']]}
    return coordinatePolygon, np.array([coordinateVisualize], np.int32)


def convertBoundingBox2PolygonForLabelStudio(coordinates):
    coordinateVisualize = {}
    coordinatePolygon = []
    # classes = {'top_left': 0, 'top_right': 1, 'bottom_right': 2, 'bottom_left': 3}
    for i in range(len(coordinates)):
        x_center = (coordinates[i][0] + coordinates[i][2]) / 2
        y_center = (coordinates[i][1] + coordinates[i][3]) / 2
        coordinateVisualize[coordinates[i][-1]] = [x_center, y_center]
    coordinatePolygon = [coordinateVisualize['top_left'], coordinateVisualize['top_right'],
                         coordinateVisualize['bottom_right'], coordinateVisualize['bottom_left']]
    return coordinatePolygon


def predict_yolov5(image, filename, args):
    # args = parse_arg()
    pretrainedModel = weights()
    classes = ['top-cmnd', 'back-cmnd', 'top-cccd', 'back-cccd', 'top-chip', 'back-chip', 'passport', 'rotate']
    pretrainedYOLO = pretrainedModel.modelYOLOv7(args.weights)
    modelYOLO = detection(image)
    coordinate, score, name, image = modelYOLO.v7(pretrainedYOLO)
    coordinate, score = nms(coordinate, score, 0.4)
    if not os.path.exists(os.path.abspath(args.folder_save_detection)):
        os.makedirs(os.path.abspath(args.folder_save_detection))
    cv2.imwrite(os.path.join(os.path.abspath(args.folder_save_detection), filename), image)
    try:
        coordinatePolygon = convertBoundingBox2PolygonForLabelStudio(coordinate)
        return coordinatePolygon, image
    except Exception as e:
        logging.error(e)
        text1 = ["NOTICE"]
        text2 = [["PLEASE TRY AGAIN !"], ["SUGGESTION: PUT YOUR IMAGE INCLUDING BACKGROUND"]]
        print(tabulate(text2, text1, tablefmt="pretty"))
        return [], [0, 0]


def predict(image, filename, args):
    try:
        coordinateBoundingBox = []
        coordinateCompute = []
        classes = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        weightPath = 'weights/yolov7x/yolov7x.pt'
        # backSub = cv2.createBackgroundSubtractorKNN()
        # mask = backSub.apply(image)
        image = cv2.copyMakeBorder(image, int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                                   int(image.shape[1] * 0.1), int(image.shape[1] * 0.1), cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
        # Load weights
        pretrainedModel = weights()
        pretrainedYOLO = pretrainedModel.modelYOLOv7(weightPath)
        modelYOLO = detection(image)
        coordinate, score, name, predictImage = modelYOLO.v7(pretrainedYOLO)
        coordinate, score = nms(coordinate, score, 0.4)
        # Correcting Image Orientation
        for i in range(len(coordinate)):
            if coordinate[i][-1] == 'top_left' or coordinate[i][-1] == 'top_right':
                coordinateCompute.append(coordinate[i])
        if coordinateCompute[0][-1] != 'top_left':
            coordinateCompute.reverse()
        imageRotation = rotationBaseOn4CornersYOLO(image, coordinateCompute)
        imageRotated = imageRotation()
        # YOLO
        modelYOLO = detection(imageRotated)
        coordinate, score, name, predictImage = modelYOLO.v7(pretrainedYOLO)
        coordinate, score = nms(coordinate, score, 0.4)
        coordinatePolygon, coordinateVisualize = convertBoundingBox2Polygon(coordinate)
        imagePolygon = cv2.polylines(cv2.resize(imageRotated, (640, 640)), [coordinateVisualize], True, (0, 255, 0),
                                     thickness=3)
        if not os.path.exists(os.path.abspath(args.folder_save_detection)):
            os.makedirs(os.path.abspath(args.folder_save_detection))
        cv2.imwrite(os.path.join(os.path.abspath(args.folder_save_detection), filename), predictImage)
        # Image Alignment
        for i in range(len(coordinate)):
            coordinate[i].pop(-1)
        print(imageRotated.shape)
        imageAlignment = processingROI(cv2.resize(imageRotated, (640, 640)), coordinate)
        print(imageAlignment.suitable_cutting())
        alignedImage = cv2.resize(imageAlignment(), (1290, 740))
        # Noise image processing
        # clearingImage = reduceBlur(alignedImage)
        # clearedImage = clearingImage()
        if not os.path.exists(os.path.abspath(args.folder_save_rotation)):
            os.makedirs(os.path.abspath(args.folder_save_rotation))
        cv2.imwrite(os.path.join(os.path.abspath(args.folder_save_rotation), filename), alignedImage)
        for i in tqdm.tqdm(range(len(name)), total=len(name)):
            coordinate[i] = np.array(coordinate[i])
            results = {"class_id": classes.index(name[i]), "class_name": name[i],
                       "bbox_coordinates": coordinate[i].tolist(),
                       "confidence_score": score[i]}
            coordinateBoundingBox.append(results)
        return coordinateBoundingBox, coordinatePolygon, alignedImage

    except Exception as e:
        logging.error(e)
        text1 = ["NOTICE"]
        text2 = [["PLEASE TRY AGAIN !"], ["SUGGESTION: PUT YOUR IMAGE INCLUDING BACKGROUND"]]
        print(tabulate(text2, text1, tablefmt="pretty"))
        return {'status': 'try again'}, {'status': 'try again'}, image


def main():
    args = parse_arg()
    pretrainedModel = weights()
    coordinate = []
    score = []
    pretrainedYOLO = 0
    originalImage = cv2.imread(args.img_path)
    image = np.array(cv2.imread(args.img_path))
    coordinateCompute = []
    if args.cfg_detection == 'yolov5':
        pretrainedYOLO = pretrainedModel.modelYOLOv5(args.weights)
        modelYOLO = detection(cv2.imread(args.img_path))
        coordinate, score, name, image = modelYOLO.v5(pretrainedYOLO)
    elif args.cfg_detection == 'yolov7':
        pretrainedYOLO = pretrainedModel.modelYOLOv7(args.weights)
        modelYOLO = detection(cv2.imread(args.img_path))
        coordinate, score, name, image = modelYOLO.v7(pretrainedYOLO)
    else:
        print('[INFOR]: WARNING There are no suitable models')
    try:
        # Non-max suppression
        coordinate, score = nms(coordinate, score, 0.4)
        print('[INFO] NON MAX SUPPRESSION: ', coordinate)
        # Correcting Image Orientation
        for i in tqdm.tqdm(range(len(coordinate)), total=(len(coordinate))):
            if coordinate[i][-1] == 'top_left' or coordinate[i][-1] == 'top_right':
                coordinateCompute.append(coordinate[i])
        if coordinateCompute[0][-1] != 'top_left':
            coordinateCompute.reverse()
        imageRotation = rotationBaseOn4CornersYOLO(cv2.imread(args.img_path), coordinateCompute)
        imageRotated = imageRotation()
        # YOLO
        modelYOLO = detection(imageRotated)
        coordinate, score, name, image = modelYOLO.v7(pretrainedYOLO)
        if not os.path.exists(os.path.abspath(args.folder_save_detection)):
            os.makedirs(os.path.abspath(args.folder_save_detection))
        cv2.imwrite(os.path.join(os.path.abspath(args.folder_save_detection), args.img_path.split('/')[-1]), image)
        # Image Alignment
        for i in tqdm.tqdm(range(len(coordinate)), total=(len(coordinate))):
            coordinate[i].pop(-1)
        imageAlignment = processingROI(cv2.resize(imageRotated, (640, 640)), coordinate)
        alignedImage = imageAlignment()
        if not os.path.exists(os.path.abspath(args.folder_save_rotation)):
            os.makedirs(os.path.abspath(args.folder_save_rotation))
        cv2.imwrite(os.path.join(os.path.abspath(args.folder_save_rotation), args.img_path.split('/')[-1]),
                    alignedImage)

        # Recognition
    except Exception as error:
        logging.error(error)
        text1 = ["NOTICE"]
        text2 = [["PLEASE TRY AGAIN !"], ["SUGGESTION: PUT YOUR IMAGE INCLUDING BACKGROUND"]]
        print(tabulate(text2, text1, tablefmt="pretty"))


if __name__ == "__main__":
    args = parse_arg()
    # v = predict(cv2.imread(args.img_path))
    # print(v)
    main()
    notification.notify(
        title='DONE',
        message='CHECK YOUR RESULTS',
        timeout=10
    )
