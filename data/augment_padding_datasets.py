import os
import cv2
import tqdm
import logging
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import argparse
from IPython.display import clear_output


def scale(im):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    desired_size = 368
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return top, bottom, left, right


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    # for img in images:
    h, w = images.shape[:2]
    width_max = max(width_max, w)
    height_max = max(height_max, h)

    images_padded = []
    # for img in images:
    h, w = images.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert // 2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori // 2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(images, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
    assert img_padded.shape[:2] == (height_max, width_max)
    # images_padded.append(img_padded)

    return img_padded


def padding(img):
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = old_image_width + 100
    new_image_height = old_image_height + 100
    color = (255, 0, 255)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='folder datasets', default='./data/datasets/images')
    parser.add_argument('--args.folder_save', type=int, help='path to save result',
                        default='./data/datasets/augment_padding_images')
    return parser.parse_args()


def main():
    args = parse_arg()
    for folder in tqdm.tqdm(os.listdir(args.folder), total=len(os.listdir(args.folder))):
        if os.path.isdir(os.path.join(args.folder, folder)):
            for file in tqdm.tqdm(os.listdir(os.path.join(args.folder, folder)),
                                  total=len(os.listdir(os.path.join(args.folder, folder)))):
                clear_output(wait=True)
                if os.path.isfile(os.path.join(args.folder, folder, file)):
                    image = cv2.imread(os.path.join(args.folder, folder, file))
                    # image = Image.open(os.path.join(args.folder, folder))
                    # image = cv2.copyMakeBorder(image, int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                    #                            int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                    #                            cv2.BORDER_CONSTANT,
                    #                            value=[255, 255, 255])
                    # pad 100 to left/right and 50 to top/bottom
                    # transform = transforms.Pad((image.size[1] * 0.1, image.size[1] * 0.1))
                    # add padding to image
                    # image = transform(image)
                    image = padding(image)
                    if not os.path.exists(os.path.join(args.folder_save)):
                        os.makedirs(os.path.join(args.folder_save))
                    cv2.imwrite(os.path.join(args.folder_save, os.path.splitext(file)[0] + '.jpg'), image)
                    # image.save(os.path.join(args.folder_save, folder))
                else:
                    logging.error('It is not a file')
        elif os.path.isfile(os.path.join(args.folder, folder)):
            image = cv2.imread(os.path.join(args.folder, folder))
            # image = Image.open(os.path.join(args.folder, folder))
            # image = cv2.copyMakeBorder(image, int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
            #                            int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
            #                            cv2.BORDER_CONSTANT,
            #                            value=[255, 255, 255])
            # pad 100 to left/right and 50 to top/bottom
            # transform = transforms.Pad([int(image.size[1] * 0.1), int(image.size[1] * 0.1)])
            # add padding to image
            # image = transform(image)
            image = padding(image)
            if not os.path.exists(os.path.join(args.folder_save)):
                os.makedirs(os.path.join(args.folder_save))
            cv2.imwrite(os.path.join(args.folder_save, os.path.splitext(folder)[0] + '.jpg'), image)
            # image.save(os.path.join(args.folder_save, folder))
        else:
            print(folder)
            logging.error('It is not a image')


if __name__ == '__main__':
    main()
