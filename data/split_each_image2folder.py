import os
import cv2
import shutil
import json


def main():
    folder_datasets = '/home/long/Downloads/datasets/augment_padding'
    j = 0
    name = 0
    # for root, dirs, file in os.walk(folder_datasets):
    #     for name in file:
    #         paths.append(os.path.join(root, name))
    for file in os.listdir(folder_datasets):
        if not os.path.exists(os.path.join(folder_datasets, str(name))):
            os.makedirs(os.path.join(folder_datasets, str(name)))
        shutil.move(os.path.join(folder_datasets, file), os.path.join(folder_datasets, str(name)))
        j += 1
        if j % 30 == 0:
            name += 1


def import2json():
    folder_datasets = '/home/long/Downloads/datatests'
    list_data = []
    for file in os.listdir(folder_datasets):
        data = {'data': {'image': os.path.join('/data/local-files/?d=' + folder_datasets, file)}}
        list_data.append(data)
    with open('/home/long/Downloads/datasets/label_studio.json', 'w') as fileSave:
        json.dump(list_data, fileSave, indent=4)


if __name__ == '__main__':
    import2json()
