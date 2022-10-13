import sys

sys.path.insert(0, '/home/long/Desktop/IDCardDetectionandRecognition/yolov7')
import torch
from yolov7.models.yolo import Model
# from yolov5.utils.downloads import attempt_download
# from yolov5.utils.general import intersect_dicts
# from yolov5.models.common import DetectMultiBackend


class weights:
    def __init__(self):
        self.cuda = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.cuda else ['CPUExecutionProvider']
        self.names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

    def modelYOLOv5(self, path):
        predictedYOLOv5 = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        predictedYOLOv5.to(self.device)
        predictedYOLOv5.eval()
        return predictedYOLOv5
    # def modelYOLOv5(self, path, classes=4):
    #     model = DetectMultiBackend(path, device=self.device, fuse=True)
    #     ckpt = torch.load(attempt_download(path), map_location=self.device)  # load
    #     csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    #     csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
    #     model.load_state_dict(csd, strict=False)
    #     if len(ckpt['model'].names) == classes:
    #         model.names = ckpt['model'].names
    #     return model.to(self.device)

    # def modelYOLOv7(self, path):
    #     session = ort.InferenceSession(path, providers=self.providers)
    #     colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}
    #     return session, colors
    def modelYOLOv7(self, path, autoShape=True):
        model = torch.load(path, map_location=torch.device('cpu')) if isinstance(path, str) else path  # load checkpoint
        if isinstance(model, dict):
            model = model['ema' if model.get('ema') else 'model']  # load model
        hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
        hub_model.load_state_dict(model.float().state_dict())  # load state_dict
        hub_model.names = model.names  # class names
        if autoShape:
            hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return hub_model.to(self.device)
