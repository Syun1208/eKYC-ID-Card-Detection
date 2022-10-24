import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


class detection:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.imageSave = cv2.resize(image, (640, 640))
        # self.fileSave = os.path.splitext(image.split('/')[-1])
        # self.folderSave = os.path.join(list(image.split('/')).pop(-1))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        # self.names = ['top-cmnd', 'back-cmnd', 'top-cccd', 'back-cccd', 'top-chip', 'back-chip', 'passport', 'rotate']

    def v5(self, modelDetection):
        listCoordinate = []
        self.image = cv2.resize(self.image, (640, 640))
        resultsDetection = modelDetection(self.image)
        # resultsDetection.show()
        print('\n', resultsDetection.pandas().xyxy[0])
        names = list(resultsDetection.pandas().xyxy[0]['name'])
        '''Calculate predicted bounding box'''
        for i in range(len(resultsDetection.xyxy[0])):
            x_min = int(resultsDetection.xyxy[0][i][0])
            # x_min = np.where(x_min < 0, 0, x_min)
            y_min = int(resultsDetection.xyxy[0][i][1])
            # y_min = np.where(y_min < 0, 0, y_min)
            x_max = int(resultsDetection.xyxy[0][i][2])
            # x_max = np.where(x_max > self.image.shape[1], 0, x_max)
            y_max = int(resultsDetection.xyxy[0][i][3])
            # y_max = np.where(y_max > self.image.shape[0], 0, y_max)
            cv2.rectangle(self.imageSave, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(self.imageSave, names[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [225, 255, 255], thickness=3)
            listCoordinate.append([x_min, y_min, x_max, y_max, names[i]])
        return listCoordinate, list(resultsDetection.pandas().xyxy[0].confidence[:]), names, self.imageSave

    def v7(self, modelDetection):
        listCoordinate = []
        self.image = cv2.resize(self.image, (640, 640))
        resultsDetection = modelDetection(self.image)
        # resultsDetection.show()
        print('\n', resultsDetection.pandas().xyxy[0])
        names = list(resultsDetection.pandas().xyxy[0]['name'])
        '''Calculate predicted bounding box'''
        for i in range(len(resultsDetection.xyxy[0])):
            x_min = int(resultsDetection.xyxy[0][i][0])
            y_min = int(resultsDetection.xyxy[0][i][1])
            x_max = int(resultsDetection.xyxy[0][i][2])
            y_max = int(resultsDetection.xyxy[0][i][3])
            cv2.rectangle(self.imageSave, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(self.imageSave, names[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [225, 255, 255], thickness=3)
            listCoordinate.append([x_min, y_min, x_max, y_max, names[i]])
        return listCoordinate, list(resultsDetection.pandas().xyxy[0].confidence[:]), names, self.imageSave
    # def v7(self, session, colors):
    #     listCoordinate = []
    #     listScore = []
    #     listInfor = []
    #     listName = []
    #     self.image = cv2.resize(self.image, (640, 640))
    #     image = self.image.copy()
    #     image, ratio, dwdh = letterbox(image, auto=False)
    #     image = image.transpose((2, 0, 1))
    #     image = np.expand_dims(image, 0)
    #     image = np.ascontiguousarray(image)
    #     im = image.astype(np.float32)
    #     im /= 255
    #     ori_images = [image.copy()]
    #     outName = [i.name for i in session.get_outputs()]
    #     inName = [i.name for i in session.get_inputs()]
    #     inp = {inName[0]: im}
    #     outputs = session.run(outName, inp)[0]
    #     print('[INFO] PREDICTED: ')
    #     tables = ['CLASS', 'mAP']
    #     # if not os.path.exists(self.folderSave):
    #     #     os.mkdir(self.folderSave)
    #     for i, (batch_id, x0, y0, x1, y1, cls_id, score) in tqdm.tqdm(enumerate(outputs), total=(len(outputs))):
    #         image = ori_images[int(batch_id)]
    #         x0 = np.where(x0 < 0, 0, x0)
    #         x1 = np.where(x1 > self.image.shape[1] - 1, self.image.shape[1] - 1, x1)
    #         y0 = np.where(y0 < 0, 0, y0)
    #         y1 = np.where(y1 > self.image.shape[0] - 1, self.image.shape[0] - 1, y1)
    #         box = np.array([(x0 / 100) * self.image.shape[1], y0 / 100 * self.image.shape[0],
    #                         x1 / 100 * self.image.shape[1], y1 / 100 * self.image.shape[0]])
    #         box -= np.array(dwdh * 2)
    #         box /= ratio
    #         box = box.round().astype(np.int32).tolist()
    #         cls_id = int(cls_id)
    #         score = round(float(score), 3)
    #         name = self.names[cls_id]
    #         color = colors[name]
    #         listInfor.append([name, score])
    #         listName.append(name)
    #         listScore.append(score)
    #         cv2.rectangle(self.imageSave, box[:2], box[2:], color, 2)
    #         cv2.putText(self.imageSave, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
    #                     thickness=3)
    #         listCoordinate.append([box[0], box[1], box[2], box[3], name])
    #     print(listCoordinate)
    #     headers = [listInfor[i] for i in range(len(listInfor))]
    #     print(tabulate(headers, tables, tablefmt="psql"))
    #     notification.notify(
    #         title='[INFO] PREDICTED',
    #         message=str(listInfor),
    #         timeout=5
    #     )
    #     return listCoordinate, listScore, listName, self.imageSave
