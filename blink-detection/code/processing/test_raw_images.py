import os
import cv2
import glob
import numpy as np
from ..utils.ImageUtils import ImageUtils

if __name__ == "__main__":

    imu = ImageUtils()

    final_data_path = os.path.join(os.getcwd(), "final_data")
    DB_list = ["300VW", "300W", "afw", "lfpw", "ibug", "helen"]
    # print(os.path.exists(final_data_path))
    # print(len(glob.glob(os.path.join(final_data_path, "300W*.jpg"))))

    for db in DB_list:
        image_files = glob.glob(os.path.join(final_data_path, db+"*.jpg"))
        annot_files = glob.glob(os.path.join(final_data_path, db+"*.pts"))

        assert(len(image_files) == len(annot_files))

        test_idx = np.random.choice(range(len(image_files)))
        print(test_idx)
        print(image_files[test_idx])
        print(annot_files[test_idx])

        coords_ls = imu.extractAnnotations_68FPs(annot_files[test_idx]).reshape(-1,2).tolist()
        im = imu.drawAnnotationsOnImg(image_files[test_idx], coords_ls, window=db)
        print(im.shape)
