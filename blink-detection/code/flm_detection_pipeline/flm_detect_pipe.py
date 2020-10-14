import os
import cv2
from ..utils.ImageUtils import ImageUtils
from ..flm_detection_model.flm_model import FaceLandmarkDetection
from ..face_detect.face_detect import Face_Detectors
from glob import glob
import numpy as np

class FLM_detect_pipe:
    def __init__(self, flm_model_path, flm_model_nm):
        self.face_model = Face_Detectors("DNN")
        self.flm = FaceLandmarkDetection()
        # self.flm.loadModel(model_path=flm_model_path,
        #                     arch_nm=flm_arch,
        #                     model_nm=flm_weights)
        self.flm.loadKerasModel(model_path=flm_model_path, model_nm=flm_model_nm)

        pass

    def getFaces(self, im, target_size=(128,128), conf_thr=0.3):
        (h, w) = im.shape[:2]
        face_preds = self.face_model.getPredictions(im)
        isFound, numFound, face_bboxes = self.face_model.isFaceFound_DNN(face_preds,
                                                               conf_thr=conf_thr)
        if isFound:
            # self.face_model.showDetectionDNN(im, face_bboxes, display=True)
            adj_face_bboxes = self.face_model.getAdjustedFaceBbox_DNN((h, w), face_bboxes)
            # self.face_model.showDetectionDNN(im, adj_face_bboxes, color=(255,0,0), display=True)
            # print("adj_face_bboxes", adj_face_bboxes)
            if numFound > 1:
                adj_face_bboxes = self.face_model.removeOverlapBbox_IoU(adj_face_bboxes,
                                                                        shape=(h,w))
            faces = self.face_model.getFaces_DNN(im, adj_face_bboxes, display=False)
            faces_resized = [cv2.resize(face, target_size)/255. for face in faces]

            return adj_face_bboxes, faces, faces_resized

        return None, None, None

    def getFLMPredictions(self, faces, faces_resized, annot_type=None, idxes_range=None, display=False):

        annot_faces = []
        for face, face_resz in zip(faces, faces_resized):
            thisPred = self.flm.predict_one(face_resz)

            (h, w) = face.shape[:2]
            imu = ImageUtils()
            thisPred = thisPred.reshape(-1,2)

            # to have only specific landmarks of interest to be shown
            if idxes_range is not None:
                for range in idxes_range:
                    pred = thisPred[range[0]:range[1]]

                    rescaled_coords = imu.getInvTransformCoords(pred, w, h)

                    if annot_type == "contour":
                        face = imu.drawContours(face, rescaled_coords.tolist(),
                                                             display=display)
                    else:
                        face = imu.drawAnnotationsOnImg(face, rescaled_coords.tolist(), display=display)
            else:
                rescaled_coords = imu.getInvTransformCoords(thisPred, w, h)
                face = imu.drawAnnotationsOnImg(face, rescaled_coords.tolist(), display=display)

            annot_faces.append(face)

        return annot_faces

    def replaceAnnotFace(self, im, face_bboxes, annot_faces):
        (h,w) = im.shape[:2]
        # assert len(face_bboxes) == len(annot_faces)
        for i in range(face_bboxes.shape[0]):
            conf, x1, y1, x2, y2 = face_bboxes[i] * np.array([1,w,h,w,h])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # replace the actual face with annotated face in the original image
            im[y1:y2, x1:x2, :] = annot_faces[i]

        return im


if __name__ == "__main__":
    # path variables
    test_path = os.path.join("code", "flm_detection_pipeline", "unseen_faces")
    model_path = os.path.join("code", "flm_detection_model", "model")
    # arch_nm = "Run_1__FLMmodel_Arch_3_32_0.001.txt"
    # model_wgts = "Run_1__FLMmodel_3_32_0.001.h5"
    model_nm = "CPU_Run_1_flm_model_3_32_0.001.h5"

    img_paths = glob(os.path.join(test_path,"*.jpg"))
    # flm_detect = FLM_detect_pipe(flm_model_path=model_path,
    #                              flm_arch=arch_nm,
    #                              flm_weights=model_wgts)
    flm_detect = FLM_detect_pipe(flm_model_path=model_path, flm_model_nm=model_nm)
    imu = ImageUtils()
    for path in img_paths:
        im = cv2.imread(path)
        face_bboxes, faces, faces_resized = flm_detect.getFaces(im, target_size=(128,128))
        if faces:
            annot_faces = flm_detect.getFLMPredictions(faces, faces_resized, display=False)
            im = flm_detect.replaceAnnotFace(im, face_bboxes, annot_faces)
            imu.display(im)
        # break

    pass
