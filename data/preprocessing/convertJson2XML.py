import json as JS
import time
import xml.etree.ElementTree as ET
import os
import cv2
from PIL import Image


def calculateBoundingBox(x_center, y_center):
    scale = 10
    xmin = x_center - scale
    ymin = y_center - scale
    xmax = x_center + scale
    ymax = y_center + scale
    return [str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax))]


class visualizeImageByXML:
    def __init__(self, image, fileXML, index):
        self.image = image
        self.tree = ET.parse(fileXML)
        self.listCoordinate = []
        self.labels = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        self.index = index

    def readFileXML(self):
        root = self.tree.getroot()
        for element in root:
            for subElement in element:
                for subSubElement in subElement:
                    self.listCoordinate.append(subSubElement.text)

    def visualizationImage(self):
        img = cv2.imread(self.image)
        self.readFileXML()
        for i in range(0, len(self.listCoordinate) - 1, 4):
            cv2.rectangle(img, (int(self.listCoordinate[i]), int(self.listCoordinate[i + 1])),
                          (int(self.listCoordinate[i + 2]), int(self.listCoordinate[i + 3])), (255, 0, 0), 2)
            for j in range(len(self.labels)):
                cv2.putText(img, self.labels[j], (int(self.listCoordinate[i]), int(self.listCoordinate[i + 1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imwrite("/home/long/Downloads/datasets/visualizedImage/my_" + str(self.index) + ".png", img)
        # cv2.imshow("Visual Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    listNameFile = []
    folderDatasets = "/home/long/Downloads/datasets/images"
    # listNameFolder = folderDatasets.split('/')
    # for file in os.listdir(folderDatasets):
    #     listNameFile.append(file.split('.')[0:3])
    with open('/home/long/Downloads/datasets/images_labels.json', 'r') as json_file:
        data = JS.load(json_file)
    for i in range(len(data)):
        if len(data[i]) != 8:
            os.remove(data[i]['image'])
            continue
        else:
            root = ET.Element('annotation')
            fileJson = data[i]['image'].split('/')
            fileInformation = os.path.splitext(fileJson[len(fileJson) - 1])
            ET.SubElement(root, 'folder').text = fileJson[len(fileJson) - 2]
            ET.SubElement(root, 'filename').text = fileJson[len(fileJson) - 1]
            ET.SubElement(root, 'path').text = data[i]['image']
            Source = ET.SubElement(root, 'source')
            ET.SubElement(Source, 'database').text = 'Unknown'
            Size = ET.SubElement(root, 'size')
            ET.SubElement(Size, 'width').text = str(data[i]['label'][0]['original_width'])
            ET.SubElement(Size, 'height').text = str(data[i]['label'][0]['original_height'])
            ET.SubElement(Size, 'depth').text = str(3)
            ET.SubElement(root, 'segmented').text = str(0)
            '''Bounding box 1'''
            Object = ET.SubElement(root, 'object')
            ET.SubElement(Object, 'name').text = 'top_left'
            ET.SubElement(Object, 'pose').text = 'Unspecified'
            ET.SubElement(Object, 'truncated').text = '1'
            ET.SubElement(Object, 'difficult').text = '0'
            boundingBox = ET.SubElement(Object, 'bndbox')
            coordinate = calculateBoundingBox(
                (data[i]['label'][0]['points'][0][0] / 100) * data[i]['label'][0]['original_width'],
                (data[i]['label'][0]['points'][0][1] / 100) * data[i]['label'][0]['original_height'])
            ET.SubElement(boundingBox, 'xmin').text = coordinate[0]
            ET.SubElement(boundingBox, 'ymin').text = coordinate[1]
            ET.SubElement(boundingBox, 'xmax').text = coordinate[2]
            ET.SubElement(boundingBox, 'ymax').text = coordinate[3]
            '''Bounding box 2'''
            Object2 = ET.SubElement(root, 'object')
            ET.SubElement(Object2, 'name').text = 'top_right'
            ET.SubElement(Object2, 'pose').text = 'Unspecified'
            ET.SubElement(Object2, 'truncated').text = '1'
            ET.SubElement(Object2, 'difficult').text = '0'
            boundingBox2 = ET.SubElement(Object2, 'bndbox')
            coordinate2 = calculateBoundingBox(
                (data[i]['label'][0]['points'][1][0] / 100) * data[i]['label'][0]['original_width'],
                (data[i]['label'][0]['points'][1][1] / 100) * data[i]['label'][0]['original_height'])
            ET.SubElement(boundingBox2, 'xmin').text = coordinate2[0]
            ET.SubElement(boundingBox2, 'ymin').text = coordinate2[1]
            ET.SubElement(boundingBox2, 'xmax').text = coordinate2[2]
            ET.SubElement(boundingBox2, 'ymax').text = coordinate2[3]
            '''Bounding box 3'''
            Object3 = ET.SubElement(root, 'object')
            ET.SubElement(Object3, 'name').text = 'bottom_right'
            ET.SubElement(Object3, 'pose').text = 'Unspecified'
            ET.SubElement(Object3, 'truncated').text = '1'
            ET.SubElement(Object3, 'difficult').text = '0'
            boundingBox3 = ET.SubElement(Object3, 'bndbox')
            coordinate3 = calculateBoundingBox(
                (data[i]['label'][0]['points'][2][0] / 100) * data[i]['label'][0]['original_width'],
                (data[i]['label'][0]['points'][2][1] / 100) * data[i]['label'][0]['original_height'])
            ET.SubElement(boundingBox3, 'xmin').text = coordinate3[0]
            ET.SubElement(boundingBox3, 'ymin').text = coordinate3[1]
            ET.SubElement(boundingBox3, 'xmax').text = coordinate3[2]
            ET.SubElement(boundingBox3, 'ymax').text = coordinate3[3]
            '''Bounding box 4'''
            Object4 = ET.SubElement(root, 'object')
            ET.SubElement(Object4, 'name').text = 'bottom_left'
            ET.SubElement(Object4, 'pose').text = 'Unspecified'
            ET.SubElement(Object4, 'truncated').text = '1'
            ET.SubElement(Object4, 'difficult').text = '0'
            boundingBox4 = ET.SubElement(Object4, 'bndbox')
            coordinate4 = calculateBoundingBox(
                (data[i]['label'][0]['points'][3][0] / 100) * data[i]['label'][0]['original_width'],
                (data[i]['label'][0]['points'][3][1] / 100) * data[i]['label'][0]['original_height'])
            ET.SubElement(boundingBox4, 'xmin').text = coordinate4[0]
            ET.SubElement(boundingBox4, 'ymin').text = coordinate4[1]
            ET.SubElement(boundingBox4, 'xmax').text = coordinate4[2]
            ET.SubElement(boundingBox4, 'ymax').text = coordinate4[3]
            tree = ET.ElementTree(root)
            tree.write('/home/long/Downloads/datasets/labels/' + fileInformation[0] + '.xml')
            print('Saved in: /home/long/Downloads/datasets/labels/' + fileInformation[0] + '.xml')


if __name__ == '__main__':
    # imagePath = '/home/long/Downloads/datasets/images/' \
    #             '20210829_042129_image_jpeg.rf.73902a1ea862c9a8407bd3af3382e62b.jpg'
    # XMLPath = '/home/long/Downloads/datasets/sample/' \
    #           '20210829_042129_image_jpeg.rf.73902a1ea862c9a8407bd3af3382e62b.xml'
    i = 0
    folderDatasets = '/home/long/Downloads/datasets/sample'
    for file in os.listdir(folderDatasets):
        fileSplit = os.path.splitext(file)
        imagePath = fileSplit[0] + '.jpg'
        XMLPath = fileSplit[0] + '.xml'
        VI = visualizeImageByXML(os.path.join(folderDatasets, imagePath), os.path.join(folderDatasets, XMLPath), i)
        VI.visualizationImage()
        i += 1
    # main()
