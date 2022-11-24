# import sys
#
# sys.path.insert(0, '/home/long/Desktop/IDCardDetectionandRecognition')
# import os
# import cv2
# import numpy as np
# import uvicorn
# from typing import Optional
# from io import BytesIO
# from PIL import Image
# from PIL.Image import Image
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi import APIRouter, UploadFile, HTTPException
# from main import predict
#
# app = FastAPI()
# # yolo_router = APIRouter()
# DETECTION_URL = '/id-card-yolo/detect'
#
#
# class predictYOLO(BaseModel):
#     results: dict
#
#
# # @app.post(DETECTION_URL)
# # async def detect(image: UploadFile, response_model: predictYOLO):
# #     if image.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
# #         pass
# #     else:
# #         raise HTTPException(status_code=415, detail="Item not found")
# #     response_model = predict(image)
# #     return response_model
#
#
# @app.get('/hello-world')
# def hello():
#     return {"hello": 'Long'}
#
import uvicorn
import uuid
import cv2
import io
import base64
import imagezmq
from fastapi import FastAPI, HTTPException, UploadFile, APIRouter, File, Form, Body, Query, Depends
from pydantic import BaseModel
import os
from PIL import Image
import argparse
from starlette.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from typing import List, Union, Optional
from main import predict_yolov7, predict_yolov5
import numpy as np
from pathlib import Path
import sys
import socket
from starlette.responses import RedirectResponse

app_desc = """<h2>Made by`Pham Minh Long`</h2>"""
app = FastAPI(title="Chúa tể phát hiện cccd/cmnd", description=app_desc)
DETECTION_URL = '/id-card-yolo/detect/'
LABEL_STUDIO = '/id-card-yolo/detect_label_studio/'
hostname = socket.gethostname()
IP_ADDRESS = socket.gethostbyname(hostname)
# ROOT = str(Path.home())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_host', type=str, help='your local host connection', default=IP_ADDRESS)
    parser.add_argument('--weights', type=str, help='initial weights path', default='weights/yolov7x/yolov7x.pt')
    parser.add_argument('--port', type=int, help='your port connection', default=8000)
    parser.add_argument('--folder_save_rotation', type=str,
                        default=str(ROOT) + '/results/correct',
                        required=False)
    parser.add_argument('--folder_save_detection', type=str,
                        default=str(ROOT) + '/results/detect',
                        required=False)
    parser.add_argument('--folder_save_polygon', type=str,
                        default=str(ROOT / 'results/polygon'),
                        required=False)
    return parser.parse_args()


class choice(BaseModel):
    option: str


class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: List[str] = []


# class option(BaseModel):
#     option: str


def base64str_to_PILImage(predictedImage):
    base64_img_bytes = base64.b64encode(predictedImage).decode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


# @app.get("/", include_in_schema=False)
# async def index():
#     return RedirectResponse(url="/docs")


def image_to_base64(image):
    """
    It takes an image, encodes it as a jpg, and then encodes it as base64
    :param image: The image to be converted to base64
    :return: A base64 encoded string of the image.
    """
    _, buffer = cv2.imencode('.jpg', image)
    img_data = base64.b64encode(buffer)
    return img_data


@app.post(DETECTION_URL)
async def detect(image: UploadFile = File(...), option: str = None):
    if image.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    # with open(fileName, 'rb') as f:
    #     encoded_image = base64.b64encode(f.read())
    # decoded_image = encoded_image.decode('utf-8')
    args = parse_arg()
    contents = await image.read()
    array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    coordinateBoundingBox, coordinatePolygon, predictedImage = predict_yolov7(img, image.filename, args)
    # res, im_png = cv2.imencode(".png", predictedImage)
    # predictedImage = base64str_to_PILImage(predictedImage)
    # buffered = BytesIO()
    # Image.fromarray(predictedImage).save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue())
    encoded_string = image_to_base64(predictedImage)
    # img_decode = BytesIO(base64.b64decode(img_str))
    # print(predictedImage)
    # dataStr = json.dumps(predictedImage)
    #
    # base64EncodedStr = base64.b64encode(dataStr.encode('utf-8'))
    # print(base64EncodedStr)
    #
    # print('decoded', base64.b64decode(base64EncodedStr))
    if option == 'Y' or option == 'y':
        return {'image_name': image.filename}, coordinateBoundingBox, {'polygon_coordinates': coordinatePolygon}, {
            "encoded_image": encoded_string}
    else:
        return {'image_name': image.filename}, coordinateBoundingBox, {'polygon_coordinates': coordinatePolygon}


@app.post(LABEL_STUDIO)
async def detect_label_studio(image: UploadFile = File(...), option: str = None):
    if image.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    # with open(fileName, 'rb') as f:
    #     encoded_image = base64.b64encode(f.read())
    # decoded_image = encoded_image.decode('utf-8')
    args = parse_arg()
    contents = await image.read()
    array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    coordinatePolygon, predictedImage = predict_yolov5(img, image.filename, args)
    # res, im_png = cv2.imencode(".png", predictedImage)
    # predictedImage = base64str_to_PILImage(predictedImage)
    # buffered = BytesIO()
    # Image.fromarray(predictedImage).save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue())
    # encoded_string = image_to_base64(predictedImage)
    # img_decode = BytesIO(base64.b64decode(img_str))
    # print(predictedImage)
    # dataStr = json.dumps(predictedImage)
    #
    # base64EncodedStr = base64.b64encode(dataStr.encode('utf-8'))
    # print(base64EncodedStr)
    #
    # print('decoded', base64.b64decode(base64EncodedStr))
    return {
        'data': {'image': '/data/local-files/?d=/home/long/Downloads/datasets/augment_padding/' + image.filename},
        "annotations": [
            {
                "result": [
                    {
                        "original_width": img.shape[1],
                        "original_height": img.shape[0],
                        "image_rotation": 0,
                        "value": {
                            "points": coordinatePolygon,
                            "polygonlabels": [
                                "top-cmnd"
                            ]
                        },
                        "id": "796e373c3a",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "origin": "manual"
                    }
                ]
            }
        ],
        "predictions": []}


# @app.post('/yolo-id-card/request/')
# async def detect(data: str):
#     image = base64.b64decode(data)
#     coordinateBoundingBox, coordinatePolygon, predictedImage = predict_yolov7(image)
#     encoded_string = image_to_base64(predictedImage)
#     return coordinateBoundingBox, {'polygon_coordinates': coordinatePolygon}, {"encoded_image": encoded_string}


# @app.post("/files/")
# async def create_files(files: List[bytes] = File(description="Multiple files as bytes"),
#                        ):
#     return {"file_sizes": [len(file) for file in files]}

# @app.post("/uploadFiles/")
# async def create_upload_files(
#         files: List[UploadFile] = File(description="Multiple files as UploadFile"),
# ):
#     return {"filenames": [file.filename for file in files]}

# @app.get("/")
# async def main():
#     content = """
# <body>
# <form action="/files/" enctype="multipart/form-data" method="post">
# <input name="files" type="file" multiple>
# <input type="submit">
# </form>
# <form action="/uploadFiles/" enctype="multipart/form-data" method="post">
# <input name="files" type="file" multiple>
# <input type="submit">
# </form>
# </body>
#     """
#     return HTMLResponse(content=content)

@app.get('/')
async def read():
    return {"message": 'chào mừng đến với bình nguyên vô tận'}


# @app.get("/myapp/v1/filter/a")
# async def style_transfer(data: dict):
#     image_byte = data.get('image').encode()
#     image_shape = tuple(data.get('shape'))
#     image_array = np.frombuffer(base64.b64decode(image_byte)).reshape(image_shape)


if __name__ == '__main__':
    args = parse_arg()
    app_str = 'fast_api:app'
    uvicorn.run(app_str, host=args.local_host, port=args.port, reload=True, workers=1)
