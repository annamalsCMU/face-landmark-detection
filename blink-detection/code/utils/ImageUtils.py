import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

class ImageUtils:
    def __init__(self):
        pass

    def display(self, im, window="Preview"):
        cv2.imshow(window, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def extractAnnotations_68FPs(self, annot_file):
        # return coords of annotation in [x1, y1, x2, y2, ..., x68, y68] format
        with open(annot_file, "r") as f:
            coords = []
            start = False
            for line in f:
                line = line.strip()
                if "}" in line:
                    break # to stop the file

                if start:
                    # print(line.split())
                    coords.extend(map(np.float64, line.split()))
                if "{" in line:
                    start = True

            np_coords = np.array(coords)
            return np_coords
        pass

    def drawAnnotationsOnImg(self, img, coords_ls,
                             window="Preview", display=True):
        # image path or image array
        # coords as a 2-D list format
        if img is None:
            return None

        if isinstance(img, str): # if sent as a path
            im = cv2.imread(img)
        else: # if sent as numpy array
            im = img

        for points in coords_ls:
            # print(points)
            cv2.circle(im, # image
                       (int(points[0]), int(points[1])), # center
                       1, # thickness
                       (0, 255, 0), # color
                       lineType=cv2.LINE_AA)

        if display:
            self.display(im, window)

        return im

    def drawContours(self, img, coords_ls, window="Preview", display=True):
        if img is None:
            return None

        if isinstance(img, str): # if sent as a path
            im = cv2.imread(img)
        else: # if sent as numpy array
            im = img

        pts = np.array(coords_ls)
        cv2.drawContours(im, contours=np.int32([pts]),contourIdx=0,color=(0,255,0),
                             thickness=1, lineType=cv2.LINE_AA)

        return im

    def showBboxOnImg(self, img, bbox_coords_tup, window="Preview", display=True):
        if img is None:
            return None

        if isinstance(img, str): # if sent as a path
            im = cv2.imread(img)
        else: # if sent as numpy array
            im = img

        cv2.rectangle(im, bbox_coords_tup[0], bbox_coords_tup[1], (0,255,0), 2)

        if display:
            self.display(im, window)

        return im

    def cropImg(self, img, x1, y1, x2, y2):
        if img is None:
            return None

        if isinstance(img, str): # if sent as a path
            im = cv2.imread(img)
        else: # if sent as numpy array
            im = img

        return im[y1:y2, x1:x2]

    def writeTextOnImg(self, im, x, y, text, color=(0,0,255), thick=2, window="Preview", display = True):
        # write text at x, y coords
        xy_tup = (x, y - 10 if y - 10 > 10 else y + 10)
        cv2.putText(im, text, xy_tup, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color=color, thickness=thick)
        if display:
            self.display(im, window)
        return im

    def makeSquareBbox(self, x1, y1, x2, y2):
        # print(x1, y1, x2, y2)
        bbox_w, bbox_h = x2 - x1, y2 - y1
        diff = bbox_h - bbox_w
        delta = int(np.abs(diff) / 2)

        if diff == 0:
            return x1, y1, x2, y2
        if diff > 0:
            x1 -= delta
            x2 += delta
            if diff % 2 == 1:
                x2 += 1
        else:
            y1 -= delta
            y2 += delta
            if diff % 2 == 1:
                y2 += 1

        assert((x2 - x1) == (y2 - y1)), "Not Equal in length - check bbox."
        return x1, y1, x2, y2

    def moveBbox(self, x1, y1, x2, y2, by, dir = 'down'):
        if dir == 'up':
            x1, y1, x2, y2 = x1, y1 - by, x2, y2 - by
        elif dir == 'down':
            x1, y1, x2, y2 = x1, y1 + by, x2, y2 + by
        elif dir == 'right':
            x1, y1, x2, y2 = x1 + by, y1, x2 + by, y2
        elif dir == 'left':
            x1, y1, x2, y2 = x1 - by, y1, x2 - by, y2
        return x1, y1, x2, y2

    def getOptBbox(self, x1, y1, x2, y2, img_shape, fit=False):
        h, w = img_shape
        # print(f"Original - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, h: {h}, w: {w}")

        if y2 <= h and x2 <= w and x1 >= 0 and y1 >= 0:
            return (x1, y1, x2, y2)

        if fit:
            if y2 > h:
                while y2 > h and y1 >= 0:
                    # print("moving up.....")
                    x1, y1, x2, y2 = self.moveBbox(x1, y1, x2, y2, by=1, dir='up')
                    # print(f"Up - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, h: {h}, w: {w}")
            if x2 > w:
                while x2 > w and x1 >= 0:
                    # print("moving left.....")
                    x1, y1, x2, y2 = self.moveBbox(x1, y1, x2, y2, by=1, dir='left')
                    # print(f"Left - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, h: {h}, w: {w}")
            if y1 < 0:
                while y1 < 0 and y2 <= h:
                    # print("moving down.....")
                    x1, y1, x2, y2 = self.moveBbox(x1, y1, x2, y2, by=1, dir='down')
                    # print(f"Down - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, h: {h}, w: {w}")
            if x1 < 0:
                while x1 < 0 and x2 <= w:
                    # print("moving right.....")
                    x1, y1, x2, y2 = self.moveBbox(x1, y1, x2, y2, by=1, dir='right')
                    # print(f"Right - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, h: {h}, w: {w}")

        # makes sure that the bbox coords are not outside the image dims
        return (max(x1, 0), max(y1, 0), min(x2, w), min(y2, h))

    def getTransformCoords(self, x1, y1, x2, y2, coords_np):
        # make coord correction
        coords_np[:, 0] -= x1
        coords_np[:, 1] -= y1

        # normalize
        w = x2 - x1
        h = y2 - y1
        coords_np[:, 0] /= w
        coords_np[:, 1] /= h

        return coords_np

    def getInvTransformCoords(self, coords_np, w, h):
        # restore coords based on dims w,h
        coords_np[:, 0] *= w
        coords_np[:, 1] *= h
        return coords_np

    def saveImage(self, img, path, name, display=False):
        file = os.path.join(path, name)
        cv2.imwrite(file, img)
        if display:
            print(f"Image saved successfully in path: {file}")

        pass

    def display_multiple_img(self, images, rows=1, cols=1, window_size=(10,10),
                             print_title=False, titles=[]):
        plt.rcParams['figure.figsize'] = window_size
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        if print_title:
            images = zip(images, titles)
        else:
            images = zip(images, [""] * len(images))

        for ind, img_title in enumerate(images):
            img, title = img_title
            ax.ravel()[ind].imshow(img)
            if print_title:
                ax.ravel()[ind].set_title(title)
            ax.ravel()[ind].set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # check the util function to draw annotations in an image
    annot_file_path = os.path.join("..", "..", "data", "afw", "134212_1.pts")
    image_file_path = os.path.join("..", "..", "data", "afw", "134212_1.jpg")
    imu = ImageUtils()
    annot_coords = imu.extractAnnotations_68FPs(annot_file_path)
    annot_coords_ls = annot_coords.reshape(-1,2).tolist()

    imu.drawAnnotationsOnImg(image_file_path, annot_coords_ls)

