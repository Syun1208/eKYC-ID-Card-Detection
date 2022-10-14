import shutil
import os
import numpy as np
import argparse


def get_files_from_folder(path):
    files = os.listdir(path)
    return files


def main(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_path", required=False,
                        help="Path to data", default='/home/long/Downloads/datasets/train')
    parser.add_argument("--test_data_path_to_save", required=False,
                        help="Path to test data where to save", default='/home/long/Downloads/datasets/val')
    parser.add_argument("--train_ratio", required=False,
                        help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test", default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_path, args.test_data_path_to_save, float(args.train_ratio))
