import os
import cv2
import tqdm
import logging
from IPython.display import clear_output


def main():
    folders = '/home/long/Downloads/datasets/aws_files_22_0505_1517'
    folder_save = '/home/long/Downloads/datasets/augment_padding'
    for folder in tqdm.tqdm(os.listdir(folders), total=len(os.listdir(folders))):
        if os.path.isdir(os.path.join(folders, folder)):
            for file in tqdm.tqdm(os.listdir(os.path.join(folders, folder)),
                                  total=len(os.listdir(os.path.join(folders, folder)))):
                clear_output(wait=True)
                if os.path.isfile(os.path.join(folders, folder, file)):
                    image = cv2.imread(os.path.join(folders, folder, file))
                    image = cv2.copyMakeBorder(image, int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                                               int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                                               cv2.BORDER_CONSTANT,
                                               value=[255, 255, 255])
                    if not os.path.exists(os.path.join(folder_save)):
                        os.makedirs(os.path.join(folder_save))
                    cv2.imwrite(os.path.join(folder_save, file), image)
                else:
                    logging.error('It is not a file')
        elif os.path.isfile(os.path.join(folders, folder)):
            image = cv2.imread(os.path.join(folders, folder))
            image = cv2.copyMakeBorder(image, int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                                       int(image.shape[1] * 0.1), int(image.shape[1] * 0.1),
                                       cv2.BORDER_CONSTANT,
                                       value=[255, 255, 255])
            if not os.path.exists(os.path.join(folder_save)):
                os.makedirs(os.path.join(folder_save))
            cv2.imwrite(os.path.join(folder_save, folder), image)
        else:
            print(folder)
            logging.error('It is not a image')


if __name__ == '__main__':
    main()
