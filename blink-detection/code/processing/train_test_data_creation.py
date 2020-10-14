import os
from glob import glob
import numpy as np
import random
import shutil
from tqdm import tqdm
from ..utils.GenUtils import GenUtils
from collections import Counter

if __name__ == "__main__":

    genu = GenUtils()
    final_faces_data_path = "faces_data"

    ## for train images
    image_paths = glob(os.path.join(final_faces_data_path, "*.jpg"))
    annot_paths = glob(os.path.join(final_faces_data_path, "*.pkl"))
    print(len(image_paths), len(annot_paths))

    # testing randomly if both idxes has the image and corresponding annots
    # for i in tqdm(range(1000)):
    #     test_idx = np.random.choice(range(len(image_paths)), 1)[0]
    #     assert(image_paths[test_idx].split(".")[0] == \
    #                     annot_paths[test_idx].split(".")[0])

    shuffled_idxes = list(range(len(image_paths)))
    random.shuffle(shuffled_idxes)

    train_idxes = shuffled_idxes[:200000]
    val_idxes = shuffled_idxes[200000:-10000]
    test_idxes = shuffled_idxes[-10000:]

    # move train images
    print("Moving Train Files.....")
    genu.moveFiles(train_idxes, image_paths, os.path.join(final_faces_data_path, "train"))
    genu.moveFiles(train_idxes, annot_paths, os.path.join(final_faces_data_path, "train"))

    # move val images
    print("Moving Validation Files.....")
    genu.moveFiles(val_idxes, image_paths, os.path.join(final_faces_data_path, "val"))
    genu.moveFiles(val_idxes, annot_paths, os.path.join(final_faces_data_path, "val"))

    # move test images
    print("Moving Test Files.....")
    genu.moveFiles(test_idxes, image_paths, os.path.join(final_faces_data_path, "test"))
    genu.moveFiles(test_idxes, annot_paths, os.path.join(final_faces_data_path, "test"))

