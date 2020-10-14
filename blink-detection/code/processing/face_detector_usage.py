import os
import cv2
import numpy as np
import glob
from ..utils.ImageUtils import ImageUtils

def showDetectionHAAR(faceCascade, image_path, annot_path, window_nm="Preview"):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(50,50)
    )

    print(f"Found {len(faces)} faces")

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow(window_nm, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

def showDetectionDNN(model, image_path, annot_path, conf_thr=0.5,
                     window_nm="Preview"):

    im = cv2.imread(image_path)
    (h,w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(im, (300,300)),
                                 1.0, # scale factor
                                 (300,300), # shape
                                 -127, # mean to be normalized
                                 False, False)
    model.setInput(blob)

    # predictions of the network
    detections = model.forward()
    # print(detections.shape)

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > conf_thr:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype("int")

            cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,255), 2)

            text=f"Face: {np.round(conf*100,4)}"
            text_pos = (x1, y1-10 if y1-10 > 10 else y1+10)
            cv2.putText(im, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # put annotations in place
    imu = ImageUtils()
    coords_ls = imu.extractAnnotations_68FPs(annot_path).reshape(-1, 2).tolist()
    im = imu.drawAnnotationsOnImg(im, coords_ls, display=False)

    cv2.imshow(window_nm, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return im


if __name__ == "__main__":

    final_data_path = os.path.join("final_data")

    image_files = glob.glob(os.path.join(final_data_path, "*.jpg"))
    annot_files = glob.glob(os.path.join(final_data_path, "*.pts"))

    idxes = np.random.choice(range(len(image_files)), 20)
    print(idxes)

    # HAAR
    #### has a lot of falss-positives
    # haar_model = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')
    ### eye-detector not so good
    # eye_detector = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_eye.xml')

    # DNN
    mdl_file = os.path.join(os.getcwd(), "code", "face_detect", "face_detection_models",
                            "res10_300x300_ssd_iter_140000.caffemodel")
    cfg_file = os.path.join(os.getcwd(), "code", "face_detect", "face_detection_models",
                            "deploy.prototxt.txt")
    if os.path.exists(mdl_file) & os.path.exists(cfg_file):
        dnn_model = cv2.dnn.readNetFromCaffe(cfg_file, mdl_file)
    else:
        print(mdl_file)
        print(cfg_file)
        print(os.path.exists(mdl_file), os.path.exists(cfg_file))

    for idx in idxes:
        image_path = image_files[idx]
        annot_path = annot_files[idx]

        # HAAR
        # showDetectionHAAR(haar_model, image_path, annot_path, str(idx)

        # DNN - chosen model
        showDetectionDNN(dnn_model, image_path, annot_path, window_nm=str(idx))