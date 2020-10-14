import os
import cv2
from ..utils.GenUtils import GenUtils
from ..utils.ImageUtils import ImageUtils
from glob import glob
import numpy as np


if __name__ == "__main__":
    faces_data_path = os.path.join("faces_data", "test")
    image_paths = glob(os.path.join(faces_data_path, "*.jpg"))
    annot_paths = glob(os.path.join(faces_data_path, "*.pkl"))

    TARGET_SIZE = 128

    genu = GenUtils()
    imu = ImageUtils()

    ims = []
    for _ in range(10):
        test_idx = np.random.choice(range(len(image_paths)), 1)[0]

        coords_np = genu.loadPKL(annot_paths[test_idx]).reshape(-1,2)
        # print(coords_np.shape)
        rescaled_coords = imu.getInvTransformCoords(coords_np, TARGET_SIZE, TARGET_SIZE)

        im = cv2.imread(image_paths[test_idx])
        im = imu.drawAnnotationsOnImg(im, rescaled_coords.tolist(), "Preview", display=False)
        ims.append(im)

    imu.display_multiple_img(ims, rows = 2, cols=5)
