import requests
import argparse
import tqdm
import json
import os
from IPython.display import clear_output
from pathlib import Path

ROOT = str(Path.home())


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_host', type=str, help='your local host connection', default='0.0.0.0')
    parser.add_argument('--port', type=str, help='your port connection', default='8000')
    parser.add_argument('--folder', type=str, help='folder image path', default=os.path.join(ROOT, 'Downloads'))
    parser.add_argument('--image', type=str, help='image path', default=os.path.join(ROOT, 'Downloads/long.jpg'))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    response = {}
    listResults = []
    url = 'http://' + args.local_host + ':' + args.port + '/id-card-yolo/detect/'
    # with open(args.image, "rb") as image_file:
    #     data = base64.b64encode(image_file.read()).decode('utf-8')
    option = input('Do you want to show encoded image[Y/N]: ')
    if args.folder:
        for image in tqdm.tqdm(os.listdir(args.folder), total=len(args.folder)):
            file = {'image': open(os.path.join(args.folder, image), 'rb')}
            data = {'option': option}
            response = requests.post(url=url, files=file, params=data)
            # response_dict = json.loads(response.text)
            clear_output(wait=True)
            listResults.append(response.json())
            print(response.json(), '\n')
            # print(response_dict)
            # print(os.path.dirname(os.path.realpath('__file__')))
            with open(os.path.join(ROOT, 'Desktop/base64.json'), 'w') as fileSave:
                json.dump(listResults, fileSave, indent=4)
    else:
        file = {'image': open(args.image, 'rb')}
        data = {'option': option}
        response = requests.post(url=url, files=file, params=data)
        with open(os.path.join(ROOT, 'Desktop/base64.json'), 'w') as fileSave:
            fileSave.seek(0)
            json.dump(response.json(), fileSave, indent=4)
            fileSave.truncate()
    # fileSave.write(json_object)
    # response = requests.post(
    #     url='http://10.40.2.223:8000/yolo-id-card/request/',
    #     data=str(data)
    # )
    # image_name = os.path.basename(args.image)
    # response = requests.post(
    #     url='http://10.40.2.223:8000/id-card-yolo/detect/',
    #     data={
    #         "image": data,
    #         "image_name": image_name
    #     }
    # )
    #
    # # Print output
    # print(requests.status_codes, response.json())
