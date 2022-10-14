import os
import json
import argparse
import time
import shutil


# folderLabelPolygon = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsPolygon'
# folderLabelBoundingBox = '/home/long/Downloads/datasets/standard_datasets_YOLO/labelsBoundingBox'
# folderImagePath = '/home/long/Downloads/datasets/standard_datasets_YOLO/datasets'
# jsonFile = '/home/long/Downloads/datasets/images_labels.json'
# listNameFile = []
# listNameFileJson = []
# with open(jsonFile, 'r') as f:
#     datas = json.load(f)
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderBoundingBox', type=str, help='folder saves label bounding box YOLO')
    parser.add_argument('--folderPolygon', type=str, help='folder saves label polygon YOLO')
    parser.add_argument('--folderImage', type=str, help='folder contains images')
    parser.add_argument('--imageSavePolygon', type=int, help='folder saves polygon visualization')
    parser.add_argument('--imageSaveBoundingBox', default='', help='folder saves bounding box visualization')
    parser.add_argument('--jsonPath', type=str, help='address file json label')
    return parser.parse_args()


'''============Check label file==========='''


def checkLabelFile(args):
    invalid = 0
    listNameFile = []
    with open(args.jsonPath, 'r') as f:
        datas = json.load(f)
    for file in os.listdir(args.folderImage):
        listNameFile.append(os.path.splitext(file)[0])
    for data in datas:
        addressImg = data['image'].split('/')
        nameLabel = os.path.splitext(addressImg[len(addressImg) - 1])
        if nameLabel[0] not in listNameFile:
            if args.folderPolygon:
                os.remove(
                    os.path.join(args.folderPolygon, os.path.splitext(addressImg[len(addressImg) - 1])[0] + ".txt"))
            elif args.folderPolygon:
                os.remove(
                    os.path.join(args.folderBoundingBox, os.path.splitext(addressImg[len(addressImg) - 1])[0] + ".txt"))
            else:
                os.remove(
                    os.path.join(args.folderPolygon, os.path.splitext(addressImg[len(addressImg) - 1])[0] + ".txt"))
                os.remove(
                    os.path.join(args.folderBoundingBox, os.path.splitext(addressImg[len(addressImg) - 1])[0] + ".txt"))
            invalid += 1
    print('Existed file image: ', len(listNameFile))
    print('Invalid file label: ', invalid)
    print('Label length: ', len(datas))
    if args.folderPolygon:
        print('Reality Label: ', len(os.listdir(args.folderPolygon)))
        print("============================================")
    elif args.folderBoundingBox:
        print('Reality Label: ', len(os.listdir(args.folderBoundingBox)))
        print("============================================")
    else:
        print('Reality Label: ', len(os.listdir(args.folderPolygon)))
        print('Reality Label: ', len(os.listdir(args.folderBoundingBox)))
        print("============================================")


'''=============Check image file============'''


def checkImageFile(args):
    invalid = 0
    listNameFileJson = []
    with open(args.jsonPath, 'r') as f:
        datas = json.load(f)
    for data in datas:
        nameImg = data['image'].split('/')
        listNameFileJson.append(nameImg[len(nameImg) - 1])
    for file in os.listdir(args.folderImage):
        if file not in listNameFileJson:
            os.remove(os.path.join(args.folderImage, file))
            invalid += 1
    print('Existed file: ', len(listNameFileJson))
    print('Invalid file: ', invalid)
    print('Reality file: ', len(os.listdir(args.folderImage)))
    print("============================================")


def checkImageWithLabel(args):
    listImage = []
    invalid = 0
    valid = 0
    for image in os.listdir(args.folderImage):
        listImage.append(os.path.splitext(image.split('/')[-1])[0])
    if args.folderBoundingBox:
        for label in os.listdir(args.folderBoundingBox):
            nameLabel = os.path.splitext(label.split('/')[-1])[0]
            if nameLabel not in listImage:
                os.remove(os.path.join(args.folderBoundingBox, label))
                invalid += 1
            valid += 1
        print('Invalid file: ', invalid)
        print('Reality image file: ', len(os.listdir(args.folderImage)))
        print('Reality label file: ', len(os.listdir(args.folderBoundingBox)))
    elif args.folderPolygon:
        for label in os.listdir(args.folderPolygon):
            nameLabel = os.path.splitext(label.split('/')[-1])[0]
            if nameLabel not in listImage:
                os.remove(os.path.join(args.folderPolygon, label))
                invalid += 1
            valid += 1
        print('Invalid file: ', invalid)
        print('Reality image file: ', len(os.listdir(args.folderImage)))
        print('Reality label file: ', len(os.listdir(args.folderPolygon)))
    else:
        for label in os.listdir(args.folderBoundingBox):
            nameLabel = os.path.splitext(label.split('/')[-1])[0]
            if nameLabel not in listImage:
                os.remove(os.path.join(args.folderBoundingBox, label))
                invalid += 1
            valid += 1
        print('Invalid file: ', invalid)
        print('Reality image file: ', len(os.listdir(args.folderImage)))
        print('Reality label file: ', len(os.listdir(args.folderBoundingBox)))
        for label in os.listdir(args.folderPolygon):
            nameLabel = os.path.splitext(label.split('/')[-1])[0]
            if nameLabel not in listImage:
                os.remove(os.path.join(args.folderPolygon, label))
                invalid += 1
            valid += 1
        print('Invalid file: ', invalid)
        print('Reality image file: ', len(os.listdir(args.folderImage)))
        print('Reality label file: ', len(os.listdir(args.folderPolygon)))
