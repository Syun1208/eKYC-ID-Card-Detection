import os
import json
import time
import shutil
import tqdm

folderLabel = '/home/long/Downloads/datasets/standard_datasets_YOLO/datasets_bounding_box_original_card/labels'
folderImagePath = "//home/long/Downloads/datasets/standard_datasets_YOLO/datasets_bounding_box_original_card/images/test"
folderLabelMove = '/home/long/Downloads/datasets/standard_datasets_YOLO/datasets_bounding_box_original_card/labels/test'
jsonFile = '/home/long/Downloads/datasets/images_labels.json'
listNameFile = []
i = 0
if not os.path.exists(folderLabelMove):
    os.mkdir(folderLabelMove)
for file in os.listdir(folderImagePath):
    listNameFile.append(file)
for fileLabel in tqdm.tqdm(os.listdir(folderLabel), desc='Mapping datasets and labels'):
    fileInformation = os.path.splitext(fileLabel)
    fileCompare = fileInformation[0] + '.jpg'
    if fileCompare in listNameFile:
        i += 1
        shutil.move(os.path.join(folderLabel, fileLabel), folderLabelMove)

print('There are ' + str(i) + ' files already moved')
