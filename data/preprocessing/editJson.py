import json
import os
import tqdm

folderImagePath = '/home/long/Downloads/datasets/version1'
jsonPath = '/home/long/Downloads/datasets/images_labels.json'
listNameImg = []
for file in os.listdir(folderImagePath):
    fileInformation = os.path.splitext(file)
    listNameImg.append(fileInformation[0])
with open(jsonPath, 'r+') as f:
    datas = json.load(f)
    for i in tqdm.tqdm(range(len(datas)), desc='Editing data'):
        nameImgJson = datas[i]['image'].split('/')
        nameImgJsonInfor = os.path.splitext(nameImgJson[len(nameImgJson) - 1])
        if nameImgJsonInfor[0] in listNameImg:
            datas[i]['image'] = os.path.join(folderImagePath, nameImgJson[len(nameImgJson) - 1])
            f.seek(0)
            json.dump(datas, f, indent=4)
            f.truncate()
