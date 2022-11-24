import shutil
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def main():
    folder_datasets = '/root/card-transformation/uniarts/datasets/images'
    for root, folders, files in os.walk(folder_datasets, topdown=True):
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                # os.rename(os.path.join(root, file), os.path.join(root, root.split('/')[-1] + '_' + file))
                # if os.path.exists(os.path.join(folder_datasets, root.split('/')[-1] + '_' + file)):
                #     continue
                shutil.move(os.path.join(root, file), folder_datasets)
        # shutil.rmtree(root)
    for item in os.listdir(folder_datasets):
        if os.path.isdir(os.path.join(folder_datasets, item)):
            os.system('rm -rf %s' % os.path.join(folder_datasets, item))


if __name__ == '__main__':
    main()
