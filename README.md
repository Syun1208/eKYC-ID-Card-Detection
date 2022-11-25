[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)][1]
[![Messenger](https://img.shields.io/badge/Messenger-00B2FF?style=for-the-badge&logo=messenger&logoColor=white)][2]
[![Weights and Bias](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)][3]
[![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)][4]
[![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)][5]
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![PostMan](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=Postman&logoColor=white)
[![Skype](https://img.shields.io/badge/Skype-00AFF0?style=for-the-badge&logo=skype&logoColor=white)][6]
![GitLab](https://img.shields.io/badge/GitLab-330F63?style=for-the-badge&logo=gitlab&logoColor=white)
[![LinkIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)][8]
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Conda](    https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)


[1]: https://pytorch.org

[2]: https://www.facebook.com/syun128

[3]: https://wandb.ai/site

[4]: https://fastapi.tiangolo.com/

[5]: https://github.com/Syun1208

[6]: https://join.skype.com/invite/x3bJIhveDnae

[7]: https://git.sunshinetech.vn/dev/ai/icr/idc-transformation.git

[8]: https://www.linkedin.com/in/syun-cet


---
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <h3>Hi, I'm Long, author of this repository 🚀.</h3>
<a>
    <img src="image/315110985_534092584937812_6201874043567503082_n.png" alt="Logo" width="" height="">
</a>
<h1 align="center">VIETNAMESE ID CARD DETECTION BASED ON YOLOV7</h1>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#frame-works-and-environments">Frameworks and Environments</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#implementation">Implementation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project
* In this day and age, we have many model detection such as Faster-RCNN, SDD, YOLO, and so on.
* More specifically, we will apply the lastest version of YOLO, namely YOLOv7.
  In order to take ROI in ID Card, we additionally use Perspective Transform based on
  4 orientations of image, namely top-left, top-right, bottom-left, bottom-right.
* However, when we cut the ROI in image completely, the orientation of image is not correct. Moreover, many applications
  have used
  classification model to category the corners such as CNN, ResNet50, AlexNet, and so on. But this method will
  be low inference.
* Therefore, we decide to apply mathematics so as to calculate the corner replied on the orientated
  vector of top-left and top-right that we will describe in this repository.

### Frameworks and Environments

* [![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)][1]
* [![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)][4]
* ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
* ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

<a>
    <img src="image/Anaconda-entrada-hd.jpg" alt="Logo" width="500" height="250">
</a>

First of all, we need to install anaconda environment.

* conda
    ```sh
    conda create your_conda_environment
    conda activate your_conda_environment
    ```

Then, we install our frameworks and libraries by using pip command line.

* pip
  ```shell
  pip install -r path/to/requirements.txt
  ```

We suggest that you should use python version 3.8.12 to implement this repository.

### Installation

1. Check CUDA and install Pytorch with conda
    ```sh
    nvidia-smi
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
2. Clone the repository
   ```sh
   git clone https://github.com/Syun1208/IDCardDetectionAndRecognition.git
   ```

### Implementation

1. Preprocessing Data

* Dataset size
  ```shell
      Total: 21777 images (100%)
      Train: 10888 images (50%)
      Val: 4355 images (20%)
      Test: 6534 (30%)
  ```
* Data's label structure.
  ```shell
  [    
    {
        "image": "/home/long/Downloads/datasets/version1/top_132045101_13680_jpg.rf.6d2adba419f676ee9bbab8c5a277a1b2.jpg",
        "id": 13946,
        "label": [
            {
                "points": [
                    [
                        8.88888888888889,
                        36.796875
                    ],
                    [
                        86.25,
                        37.1875
                    ],
                    [
                        85.83333333333333,
                        64.765625
                    ],
                    [
                        9.305555555555555,
                        64.609375
                    ]
                ],
                "polygonlabels": [
                    "top-cmnd"
                ],
                "original_width": 720,
                "original_height": 1280
            }
        ],
        "annotator": 9,
        "annotation_id": 16871,
        "created_at": "2022-09-27T11:06:56.424119Z",
        "updated_at": "2022-09-27T11:06:58.197087Z",
        "lead_time": 15.073
    }, 
    ......................
  ]
  ```
* Folder structure trained on YOLOv7
  ```shell
    ├── test
    │   ├── images
    │   └── labels
    ├── train
    │   ├── images
    │   └── labels
    └── val
        ├── images
        └── labels
  ```
  
* If you want custom datasets(json) to yolo's bounding box, please run this command line.
    ```sh
    python path/to/data/preprocessing/convertJson2YOLOv5Label.py --folderBoundingBox path/to/labels --folderImage path/to/images --imageSaveBoundingBox path/to/save/visualization --jsonPath path/to/json/label 
    ```
* If you want custom datasets(json) to yolo's polygon and 4 corners of images, please run this command line.
    ```sh
    python path/to/data/preprocessing/convertJson2YOLOv54Corners.py --folderBoundingBox path/to/save/labels --folderPolygon path/to/save/labels --folderImage path/to/images --imageSaveBoundingBox path/to/save/visualization --imageSavePolygon path/to/save/visualization --jsonPath path/to/json/label
    ```
* Padding your dataset containing image.
  ```shell
  python path/to/data/preprocessing/augment_padding_datasets.py --folder path/to/folder/images --folder_save --path/to/save/result
  ```

2. Testing on local computer

* Put your image's option and run to see the result
  ```sh
  python path/to/main.py --weights path/to/weight.pt --cfg-detection yolov7 --img_path path/to/image 
  ```

3. Testing on API

* You need change your local host and port which you want to configure
  ```sh
  python path/to/fast_api.py --local_host your/local/host --port your/port
  ```
4. Request to API

* If you are down for requesting a huge of image to API, run this command
  ```shell
  python path/to/test_api.py --url link/to/api --source path/to/folder/images
  ```

<!-- ROADMAP -->

## Roadmap

- [x] Data Cleaning
- [x] Preprocessing Data
- [x] Model Survey and Selection
- [x] Do research on paper
- [x] Configuration and Training Model
- [x] Testing and Evaluation
- [x] Implement Correcting Image Orientation
- [x] Build Docker and API using FastAPI
- [x] Write Report and Conclusion

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (
and known issues).

<!-- CORRECTING IMAGE ORIENTATION -->

## Correcting Image Orientation

Based on the predicted bounding box,
we will flip the image with 3 cases 90, 180, 270
degrees by calculating the angle between vector Ox
and vector containing coordinates top left and top right
for vector AB with A as top left, B is the top right of the image as shown below.

<a>
    <img src="image/316366762_884260579409765_8308834300733637758_n.png" alt="Logo" width="" height="">
</a>

Let's assume that vector AB(xB - xA, yB-yA) is the combination between top_left(tl) and top_right(tr) coordination.
Therefore, we will have the equation to rotate.

<a>
    <img src="image/312217271_697159021943134_5572773548216541792_n.png" alt="Logo" width="" height="">
</a>

On the other hand, if the image has the angle which is different with zero and greater than 180 degrees, the image will
be considered with the condition below to rotate suitably.
Otherwise, the angle will be rotated following the figure above.

<a>
      <img src="image/316087765_588800566339135_7958354505803597411_n.png" alt="Logo" width="" height="">
</a>

<a>
      <img src="image/312512997_615147703719323_8867823975404687079_n.png" alt="Logo" width="" height="">
</a>

Finally we will flip in an anti-clockwise angle.

<!-- RESULTS -->

## Results

1. Polygon Detection

<a>
      <img src="image/20211102_080019211115_1_jpg.rf.fcf9ddb1141a4a907e80954a81bef6be.jpg" alt="Logo" width="" height="">
</a>

2. Correcting Image Rotation

<a>
      <img src="image/312022710_3267220280262027_4254670571883733612_n.png" alt="Logo" width="" height="">
</a>

3. Image Alignment

<a>
      <img src="image/312706888_1494662461041784_7768521949859409984_n.png" alt="Logo" width="" height="">
</a>

4. Results in API
    ```shell
    [
        {
            "image_name": "back_sang1_jpg.rf.405e033a9ecb2fb3593541e6ae20d056.jpg"
        },
        [
            {
                "class_id": 0,
                "class_name": "top_left",
                "bbox_coordinates": [
                    11,
                    120,
                    111,
                    287
                ],
                "confidence_score": 0.76953125
            },
            {
                "class_id": 1,
                "class_name": "top_right",
                "bbox_coordinates": [
                    519,
                    136,
                    636,
                    295
                ],
                "confidence_score": 0.85498046875
            },
            {
                "class_id": 2,
                "class_name": "bottom_right",
                "bbox_coordinates": [
                    524,
                    383,
                    636,
                    564
                ],
                "confidence_score": 0.89697265625
            },
            {
                "class_id": 3,
                "class_name": "bottom_left",
                "bbox_coordinates": [
                    41,
                    404,
                    104,
                    560
                ],
                "confidence_score": 0.7001953125
            }
        ]
    ```

<!-- CONTRIBUTING -->

## Contributing

1. Fork the Project
2. Create your Feature Branch

* `git checkout -b exist/folder`

3. Commit your Changes

* `git commit -m 'Initial Commit'`

4. Push to the Branch

* `git remote add origin https://git.sunshinetech.vn/dev/ai/icr/idc-transformation.git`
* `git branch -M main`
* `git push -uf origin main`

5. Open a Pull Request

<!-- CONTACT -->

## Contact

My Information - [LinkedIn](https://www.linkedin.com/in/syun-cet/) - longpm@unicloud.com.vn

Project
Link: [https://github.com/Syun1208/IDCardDetectionAndRecognition.git](https://github.com/Syun1208/IDCardDetectionAndRecognition.git)




<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [Unicloud Group](https://unicloud.com.vn/)
* [Leader: Kieu-Anh Nguyen](https://www.linkedin.com/in/kieefuanh/)