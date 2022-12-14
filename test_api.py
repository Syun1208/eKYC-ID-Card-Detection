import logging

import requests
import argparse
import tqdm
import json
import os
import socket
import sys
from IPython.display import clear_output
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT / Path.cwd()))  # relative

hostname = socket.gethostname()
IP_ADDRESS = socket.gethostbyname(hostname)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_host', type=str, help='your local host connection', default=IP_ADDRESS)
    parser.add_argument('--port', type=str, help='your port connection', default='8000')
    parser.add_argument('--json_results', type=str, help='your json path results',
                        default=ROOT / 'results/results.json')
    parser.add_argument('--url', type=str, help='link API', default='http://10.40.2.215:8000/id-card-yolo/detect/')
    parser.add_argument('--source', type=str, help='folder image test', default=ROOT / 'image')
    # parser.add_argument('--folder', type=str, help='folder image test', default=ROOT / 'image')
    # parser.add_argument('--image', type=str, help='image path', default=os.path.join(ROOT, 'image/long.jpg'))
    return parser.parse_args()


def main():
    args = parse_arg()
    response = {}
    listResults = []
    url = 'http://' + args.local_host + ':' + args.port + '/id-card-yolo/detect/'
    # with open(args.image, "rb") as image_file:
    #     data = base64.b64encode(image_file.read()).decode('utf-8')
    option = input('Do you want to show encoded image[Y/N]: ')
    if os.path.isdir(args.source):
        for image in tqdm.tqdm(os.listdir(args.source), total=len(args.source)):
            file = {'image': open(os.path.join(args.source, image), 'rb')}
            data = {'option': option}
            response = requests.post(url=url, files=file, params=data)
            if args.url:
                response = requests.post(url=args.url, files=file, params=data)
            clear_output(wait=True)
            listResults.append(response.json())
            print(response.json(), '\n')
            with open(args.json_results, 'w') as fileSave:
                json.dump(listResults, fileSave, indent=4)
    elif os.path.isfile(args.source):
        file = {'image': open(args.source, 'rb')}
        data = {'option': option}
        response = requests.post(url=url, files=file, params=data)
        with open(args.json_results, 'w') as fileSave:
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(e)
