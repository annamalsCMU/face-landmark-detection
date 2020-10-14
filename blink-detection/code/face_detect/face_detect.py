import os
import cv2
import numpy as np
from ..utils.ImageUtils import ImageUtils

class Face_Detectors:
    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == "HAAR":
            pass
        elif self.model_name == "DNN":
            model_path = os.path.join("code", "face_detect", "face_detection_models",
                            "res10_300x300_ssd_iter_140000.caffemodel")
            config_path = os.path.join("code", "face_detect", "face_detection_models",
                            "deploy.prototxt.txt")
            self.model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.imu = ImageUtils()
        pass

    def getPredictions(self, im):
        blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)),
                                     1.0,  # scale factor
                                     (300, 300),  # shape
                                     -127,  # mean to be normalized
                                     False, False)
        self.model.setInput(blob)
        # predictions of the network
        detections = self.model.forward()
        return detections

    def showDetectionDNN(self, im, detections, display=True, color=(0,0,255), window="Preview"):
        (h, w) = im.shape[:2]
        for i in range(detections.shape[0]):
            box = detections[i, 1:] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

            text = f"Face: {np.round(detections[i,0] * 100, 4)}"
            im = self.imu.writeTextOnImg(im, x1, y1, text,color=color, display=False)
            # text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
            # cv2.putText(im, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if display:
            self.imu.display(im, window)

        return im

    def isFaceFound_DNN(self, detections, conf_thr=0.5):
        num_found = 0
        faces_bbox_arr = []

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > conf_thr:
                num_found += 1
                faces_bbox_arr.append(detections[0, 0, i, 2:7])
        faces_bbox_arr = np.array(faces_bbox_arr)

        return num_found > 0, num_found, faces_bbox_arr

    def bbox_IoU(self, box1, box2):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # separate areas
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        # compute iou
        iou = interArea / float(box1Area + box2Area - interArea)
        # print("iou", iou)

        return iou

    def removeOverlapBbox_IoU(self, face_bboxes, shape):
        (h,w) = shape
        removed_faceids = set()
        for i in range(face_bboxes.shape[0]):
            if i in removed_faceids:
                continue
            for j in range(i+1, face_bboxes.shape[0]):
                conf1, box1 = face_bboxes[i, 0], face_bboxes[i, 1:] * np.array([w,h,w,h])
                conf2, box2 = face_bboxes[j, 0], face_bboxes[j, 1:] * np.array([w,h,w,h])
                thisIoU = self.bbox_IoU(box1, box2)
                if thisIoU > 0.3:
                    removed_faceids.add(j) if conf1 > conf2 else removed_faceids.add(i)
                # print(removed_faceids)
                if i in removed_faceids:
                    break

        face_idxes = set(range(face_bboxes.shape[0])) - removed_faceids
        return face_bboxes[list(face_idxes)]


    def getFaces_DNN(self, im, detections, display=True):
        (h,w) = im.shape[:2]
        faces = []

        for i in range(detections.shape[0]):
            box = detections[i] * np.array([1, w, h, w, h])
            conf, x1, y1, x2, y2 = tuple(box.tolist())
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face = self.imu.cropImg(im, x1, y1, x2, y2)
            if display:
                self.imu.display(face, window="Face Preview")
            faces.append(face)

        return faces

    def getAdjustedFaceBbox_DNN(self, im_shape, face_bboxes):
        (h, w) = im_shape
        # print(h,w)
        adj_face_bboxes = []
        for i in range(face_bboxes.shape[0]):
            box = face_bboxes[i] * np.array([1, w, h, w, h]) # convert to orig scale to perform operations
            conf, x1, y1, x2, y2 = tuple(box.tolist())
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_h = (y2 - y1)
            bbox_w = (x2 - x1)

            # calc move down by (bbox_w + bbox_h / 3) pixels
            move_down = int(np.abs(bbox_w - bbox_h) / 3)
            if (bbox_w/bbox_h) >= 0.85:
                # print("tried bbox_w/bbox_h")
                # move_down = int(np.abs(bbox_w - bbox_h) / (bbox_w / bbox_h))
                move_down += int(np.abs(bbox_w - bbox_h) / (bbox_w / bbox_h))
                move_down = int(move_down/2)
            elif 0.75 <= (bbox_w/bbox_h) < 0.85:
                # print("tried bbox_h/bbox_w")
                # move_down = int(np.abs(bbox_w - bbox_h) / (bbox_h / bbox_w))
                move_down += int(np.abs(bbox_w - bbox_h) / (bbox_h / bbox_w))
                move_down = int(move_down / 2)
            # else:
            #     move_down = int(np.abs(bbox_w - bbox_h) / 3)

            # move_down_wh = int(np.abs(bbox_w - bbox_h) / (bbox_w / bbox_h))
            # move_down_hw = int(np.abs(bbox_w - bbox_h) / (bbox_h/bbox_w))
            # print("3", int(np.abs(bbox_w - bbox_h) / 3),
            #       "w/h", (bbox_w / bbox_h), move_down_wh,
            #       "h/w", (bbox_h/bbox_w), move_down_hw)

            # move the box down
            x1, y1, x2, y2 = self.imu.moveBbox(x1, y1, x2, y2,
                                               by=move_down, dir='down')

            # making the box a square
            x1, y1, x2, y2 = self.imu.makeSquareBbox(x1, y1, x2, y2)


            # make sure it doesnt go beyond image dims
            x1, y1, x2, y2 = self.imu.getOptBbox(x1, y1, x2, y2, (h,w), fit=True)
            # choosing to allow some pictures that are not boxes

            adj_bbox = np.array([conf, x1, y1, x2, y2])
            adj_bbox /= np.array([1, w, h, w, h]) # re-normalizing
            adj_face_bboxes.append(adj_bbox)

        adj_face_bboxes = np.array(adj_face_bboxes)

        return adj_face_bboxes

    ##### below methods are to be used only for processing
    def getPotentialFaceBbox_DNN(self, im_shape, face_bboxes, chk_annots):
        (h, w) = im_shape
        poten_face_bboxes = []

        for i in range(face_bboxes.shape[0]):
            box = face_bboxes[i] * np.array([1, w, h, w, h])  # convert to orig scale to perform operations
            conf, x1, y1, x2, y2 = tuple(box.tolist())
            potential_flag = True
            for [x, y] in chk_annots:
                # print(x1 , x , x2 , y1 , y , y2, x1 <= x <=x2, y1 <= y <= y2)
                if not (x1 <= x <=x2 and y1 <= y <= y2):
                    potential_flag = False
            if potential_flag:
                poten_face_bboxes.append(face_bboxes[i])

        poten_face_bboxes = np.array(poten_face_bboxes)

        return len(poten_face_bboxes), poten_face_bboxes

    def isValidFaceBbox_DNN(self, im_shape, face_bboxes, coords_np):
        num_valid_faces, valid_face_bboxes = \
            self.getPotentialFaceBbox_DNN(im_shape, face_bboxes, coords_np.tolist())

        return num_valid_faces, valid_face_bboxes

    def getTransformCoords(self, im_shape, face_bboxes, coords_ls):
        (h, w) = im_shape
        print(coords_ls)
        trns_coords = []
        for i in range(face_bboxes.shape[0]):
            box = face_bboxes[i] * np.array([1, w, h, w, h])  # convert to orig scale to perform operations
            conf, x1, y1, x2, y2 = tuple(box.tolist())
            coords_np = self.imu.getTransformCoords(x1, y1, x2, y2, coords_ls[i])
            trns_coords.append(coords_np)
        print(trns_coords)
        return trns_coords