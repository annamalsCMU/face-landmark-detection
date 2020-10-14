import os
import cv2
from .flm_detect_pipe import FLM_detect_pipe
from ..utils.ImageUtils import ImageUtils
from ..utils.GenUtils import GenUtils

class VideoFLM:
    def __init__(self, flm_model_path, flm_model_nm):
        self.imu = ImageUtils()
        self.genu = GenUtils()
        self.flm_detect = FLM_detect_pipe(flm_model_path, flm_model_nm)


    def stream_FLM(self, video_path=None):
        # define a video capture object
        if video_path:
            vid = cv2.VideoCapture(video_path)
        else:
            vid = cv2.VideoCapture(0)

        while True:

            # Capture the video frame-by-frame
            ret, frame = vid.read()

            if not ret:
                break

            # detect facial land marks
            face_bboxes, faces, faces_resized = \
                self.flm_detect.getFaces(frame, target_size=(128, 128))

            if faces:
                annot_faces = self.flm_detect.getFLMPredictions(faces, faces_resized,
                                                                display=False)
                frame = self.flm_detect.replaceAnnotFace(frame, face_bboxes, annot_faces)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # the 'q' quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    # def video_FLM(self, video_path):


if __name__ == "__main__":
    model_path = os.path.join("code", "flm_detection_model", "model")
    # arch_nm = "Run_1__FLMmodel_Arch_3_32_0.001.txt"
    # model_wgts = "Run_1__FLMmodel_3_32_0.001.h5"
    model_nm = "GPU_RUN_3__FLMmodel_10_32_0.001.h5"

    vflm = VideoFLM(flm_model_path=model_path, flm_model_nm=model_nm)

    vflm.stream_FLM()

    # video_path = os.path.join("code", "flm_detection_pipeline",
    #                           "videos", "Inspiring the next generation of female engineers  Debbie Sterling  TEDxPSU.mp4")

    # vflm.stream_FLM(video_path)