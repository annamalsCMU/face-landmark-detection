import os
import cv2
from ..flm_detection_pipeline.flm_detect_pipe import FLM_detect_pipe
from ..utils.ImageUtils import ImageUtils
from ..utils.GenUtils import GenUtils
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


class iBlink:
    def __init__(self, flm_model_path, flm_model_nm, ear_thresh, frames_check):
        self.imu = ImageUtils()
        self.genu = GenUtils()
        self.flm_detect = FLM_detect_pipe(flm_model_path, flm_model_nm)
        self.EAR_THRESHOLD = ear_thresh
        self.FRAME_COUNT_CHECK = frames_check
        self.blinked_frame_count = 0
        self.BLINK_COUNTS = 0

    def calcEAR(self, eye_coords):
        p1, p2, p3, p4, p5, p6 = eye_coords[0], eye_coords[1], eye_coords[2], \
                                 eye_coords[3], eye_coords[4], eye_coords[5],
        p1_p4 = dist.euclidean(p1, p4)
        p2_p5 = dist.euclidean(p2, p5)
        p3_p6 = dist.euclidean(p3, p6)

        ear = (p2_p5 + p3_p6)/(p1_p4 * 2.)

        return ear

    def iBlinkPredictions(self, faces, faces_resized,
                          idxes_range=([36,42], [42,48]),
                          annot_type="points", display=False):
        annot_faces = []
        avg_EAR = None

        for face, face_resz in zip(faces, faces_resized):
            thisPred = self.flm_detect.flm.predict_one(face_resz)

            (h, w) = face.shape[:2]
            imu = ImageUtils()
            thisPred = thisPred.reshape(-1,2)

            # to have only specific landmarks of interest to be shown
            EAR = 0
            for range in list(idxes_range):

                pred = thisPred[range[0]:range[1]]
                rescaled_coords = imu.getInvTransformCoords(pred, w, h)
                EAR += self.calcEAR(rescaled_coords)

                if annot_type == "contour":
                    face = imu.drawContours(face, rescaled_coords.tolist(),
                                                         display=display)
                else:
                    face = imu.drawAnnotationsOnImg(face, rescaled_coords.tolist(), display=display)

            avg_EAR = EAR/2.

            annot_faces.append(face)

        return avg_EAR, annot_faces

    def countBlinks(self, EAR):
        if EAR < self.EAR_THRESHOLD:
            self.blinked_frame_count += 1
        else:
            if self.blinked_frame_count >= self.FRAME_COUNT_CHECK:
                self.BLINK_COUNTS += 1
            self.blinked_frame_count = 0

    def getHighestConfFace(self, face_bboxes):
        high_conf_idx = 0
        high_conf = -np.inf
        for i in range(len(face_bboxes)):
            if face_bboxes[i][0] > high_conf:
                high_conf = face_bboxes[i][0]
                high_conf_idx = i
        return high_conf_idx

    def start_iBlink(self, video_path=None, test_thres=False):
        # define a video capture object
        if video_path:
            vid = cv2.VideoCapture(video_path)
        else:
            vid = cv2.VideoCapture(0)

        avg_EARs = []
        # time = [datetime.now()]
        while True:

            # Capture the video frame-by-frame
            ret, frame = vid.read()

            if not ret:
                break

            # detect facial land marks
            face_bboxes, faces, faces_resized = \
                self.flm_detect.getFaces(frame, target_size=(128, 128))

            if faces:
                high_conf_idx = self.getHighestConfFace(face_bboxes)

                face, face_reszd = faces[high_conf_idx], faces_resized[high_conf_idx]

                EAR, annot_faces = self.iBlinkPredictions([face], [face_reszd],
                                                                annot_type="points",
                                                                display=False)
                frame = self.flm_detect.replaceAnnotFace(frame, face_bboxes, annot_faces)

                if test_thres:
                    avg_EARs.append(EAR)
                else:
                    self.countBlinks(EAR)
                    frame = self.imu.writeTextOnImg(frame, x=10, y=10,
                                                    text=f"Blinks: {self.BLINK_COUNTS}", display=False)
                    frame = self.imu.writeTextOnImg(frame, x=10, y=50,
                                                    text=f"Blink Frames: {self.blinked_frame_count}", display=False)

                    # frame = self.imu.writeTextOnImg(frame, x=10, y=90,
                    #                                 text=f"Face Conf: {conf}", display=False)
                    # frame = self.imu.writeTextOnImg(frame, x=10, y=130,
                    #                                 text=f"facebbox size: {(faces[0].shape[0]*faces[0].shape[1])/10000.}", display=False)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # 'q' is quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()


        # moving_average_EARs = pd.Series(avg_EARs).rolling(10).mean().tolist()[10 - 1:]
        # plt.plot(range(len(moving_average_EARs)), moving_average_EARs)
        if test_thres:
            plt.plot(range(len(avg_EARs)), avg_EARs)
            plt.title("Moving Average Eye Aspect Ratio - window size 10")
            plt.show()


if __name__ == "__main__":
    model_path = os.path.join("code", "flm_detection_model", "model")
    # arch_nm = "Run_1__FLMmodel_Arch_3_32_0.001.txt"
    # model_wgts = "Run_1__FLMmodel_3_32_0.001.h5"
    # model_nm = "CPU_Run_1_flm_model_3_32_0.001.h5"

    model_nm = "GPU_RUN_3__FLMmodel_10_32_0.001.h5"

    ib = iBlink(flm_model_path=model_path, flm_model_nm=model_nm,
                ear_thresh=0.42, frames_check=2)

    ib.start_iBlink(test_thres=False)