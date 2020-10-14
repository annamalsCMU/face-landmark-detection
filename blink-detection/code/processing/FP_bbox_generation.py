import os
import cv2
import glob
import numpy as np
import pandas as pd
from ..utils.ImageUtils import ImageUtils
from ..utils.GenUtils import GenUtils
from ..face_detect.face_detect import Face_Detectors
from tqdm import tqdm
import pandas as pd

def getBboxCoordsFromFPs(annot_coords, adj_fact=1.2):
    x1, x2 = int(np.min(annot_coords[:, 0])), int(np.max(annot_coords[:, 0]))
    y1, y2 = int(np.min(annot_coords[:, 1])), int(np.max(annot_coords[:, 1]))

    if adj_fact > 1.0:
        cent = (int((x2+x1)/2), int((y2+y1)/2))
        adj_len = adj_fact * (y2-y1) if (y2-y1) > (x2-x1) else adj_fact * (x2-x1)
        adj_len = int(adj_len/2.)
        x1, x2 = cent[0]-adj_len, cent[0]+adj_len
        y1, y2 = cent[1]-adj_len, cent[1]+adj_len

    # print(x1, y1, x2, y2)
    return ((x1, y1, x2, y2))

def getFacesFromImage_FPs(img_path, annot_path, target_size, adj_fact=1.2):
    imu = ImageUtils()

    im = cv2.imread(img_path)
    (h, w) = im.shape[:2]

    coords_np = imu.extractAnnotations_68FPs(annot_path).reshape(-1, 2)
    bbox_coords_tup = getBboxCoordsFromFPs(coords_np, adj_fact=adj_fact)
    opt_bbox_coords_tup = imu.getOptBbox(bbox_coords_tup[0],
                                         bbox_coords_tup[1],
                                         bbox_coords_tup[2],
                                         bbox_coords_tup[3],
                                         (h, w), fit=False)
    # print("opt", opt_bbox_coords_tup)
    # print(opt_bbox_coords_tup)
    x1, y1, x2, y2 = opt_bbox_coords_tup

    face = imu.cropImg(im, x1, y1, x2, y2)
    face = cv2.resize(face, (target_size, target_size))
    normalized_coords = imu.getTransformCoords(x1, y1, x2, y2, coords_np)

    return face, normalized_coords

def checkFaces_DNN(model, im, coords_np, conf_thr=0.5):
    (h, w) = im.shape[:2]

    below_nose_annots = coords_np.tolist()[31:36]

    predictions = model.getPredictions(im)
    isFound, _, face_bboxes = model.isFaceFound_DNN(predictions, conf_thr=conf_thr)

    num_pot_faces, pot_face_bboxes = model.getPotentialFaceBbox_DNN(
                                                (h, w),
                                                face_bboxes,
                                                below_nose_annots
                                            )

    # print(isFound, f"{num_pot_faces} Potential Face(s) Found!")
    return num_pot_faces > 0, num_pot_faces, pot_face_bboxes

def getFacesFromImage_DNN(model, im, coords_np, potential_faces):
    (h, w) = im.shape[:2]

    adj_face_bboxes = model.getAdjustedFaceBbox_DNN((h, w), potential_faces)
    _, valid_face_bboxes = model.isValidFaceBbox_DNN((h, w),
                                                   adj_face_bboxes,
                                                   coords_np)
    faces = model.getFaces_DNN(im, valid_face_bboxes, display=False)
    normalized_coords = model.getTransformCoords((h, w), adj_face_bboxes,
                                                 [coords_np]*len(adj_face_bboxes))

    return faces, normalized_coords

def getFacesFromImage(model, img_path, annot_path, target_size,
                      savePath, conf_thr=0.5, display=False, ):
    img_name = img_path.split("\\")[-1]
    annot_name = annot_path.split("\\")[-1]

    face = None
    norm_coords = None
    errors = []

    imu = ImageUtils()
    genu = GenUtils()

    im = cv2.imread(img_path)

    coords_np = imu.extractAnnotations_68FPs(annot_path).reshape(-1, 2)

    isFound, numFound, potential_faces = checkFaces_DNN(model, im, coords_np, conf_thr=conf_thr)

    if isFound:
            faces, norm_coords_ls = getFacesFromImage_DNN(model, im, coords_np, potential_faces)

            # since only one face from one image,
            # so even if 0 or > 1 valid faces comes up,
            # we skip and go to extract from FP strategy
            if len(faces) == 1:
                # print("Getting face based on DNN")
                face = cv2.resize(faces[0], (target_size, target_size))
                norm_coords = norm_coords_ls[0] # since only one is there
            else:
                if len(faces) == 0:
                    # print("No Valid face(s) found!!, pushing to FP strategy to extract face")
                    errors.append(
                        (img_path, annot_path, "No Valid Faces with all FPs in it!! - Extracted based on FP strategy!!"))
                else:
                    # print("More than 1 face found!!, pushing to FP strategy to extract face")
                    errors.append(
                        (img_path, annot_path, "Multiple Potential Faces!! - Extracted based on FP strategy!!"))

                # print("Found faces using detection, but still getting face based on FPs")
                face, norm_coords = getFacesFromImage_FPs(
                                                    img_path,
                                                    annot_path,
                                                    target_size=target_size,
                                                    adj_fact=1.2
                                                )
    else:
        # print("Getting face based on FPs")
        face, norm_coords = getFacesFromImage_FPs(
                                img_path,
                                annot_path,
                                target_size=target_size,
                                adj_fact=1.2
                            )
    # save the image and normalized coordinates
    assert(face.shape[0] == target_size and face.shape[1] == target_size), "Result face image not in shape of target size!"
    imu.saveImage(img=face, path=savePath, name=img_name)

    norm_coords_flat = norm_coords.flatten()
    norm_coords_name = annot_name.split(".")[0]
    genu.dumpPKL(obj=norm_coords_flat, path=savePath, name=norm_coords_name)

    # if we wanna see the cropped image and annotations respectively
    if display:
        rescaled_coords = imu.getInvTransformCoords(norm_coords, target_size, target_size)
        imu.drawAnnotationsOnImg(face, rescaled_coords.tolist(), img_name, display=True)

    return errors

def testImages(model, img_path, annot_path, conf_thr=0.3):
    imu = ImageUtils()

    coords_np = imu.extractAnnotations_68FPs(annot_path).reshape(-1, 2)
    im = cv2.imread(img_path)
    (h, w) = im.shape[:2]

    isFound, numFound, potential_faces = checkFaces_DNN(model, im, coords_np, conf_thr=conf_thr)

    adj_face_bboxes = model.getAdjustedFaceBbox_DNN((h, w), potential_faces)
    im = model.showDetectionDNN(im, adj_face_bboxes, display=False, color=(255,0,0))
    im = imu.drawAnnotationsOnImg(im, coords_np.tolist(), display=True, window="Testing")

    print(adj_face_bboxes)
    num_valid, valid_face_bboxes = model.isValidFaceBbox_DNN((h, w),
                                                     adj_face_bboxes,
                                                     coords_np)
    print(num_valid, len(adj_face_bboxes))

    face, norm_coords = getFacesFromImage_FPs(img_path, annot_path, target_size=128)
    rescaled_coords = imu.getInvTransformCoords(norm_coords, 128, 128)
    imu.drawAnnotationsOnImg(face, rescaled_coords.tolist(), window="Testing FPs", display=True)

    pass

if __name__ == "__main__":
    # imu = ImageUtils()

    image_paths = glob.glob(os.path.join("final_data", "*.jpg"))
    annot_paths = glob.glob(os.path.join("final_data", "*.pts"))
    # assert(len(image_paths) == len(annot_paths)), "#Images and #Annots are different"

    TARGET_SIZE = 128

    # idxes = np.random.choice(range(len(image_paths)), 2)
    idxes = range(len(image_paths))

    invalid_files = []

    for idx in tqdm(idxes):
        im_path = image_paths[idx]
        ann_path = annot_paths[idx]
        try:
            errs = getFacesFromImage(
                Face_Detectors("DNN"),
                im_path,
                ann_path,
                target_size=TARGET_SIZE,
                savePath="faces_data",
                conf_thr=0.3,
                display=False
            )

            testImages(
                Face_Detectors("DNN"),
                im_path,
                ann_path,
                )
            # collect errors
            invalid_files.extend(errs)

            # ip = input("Continue? (y)/any other char")
            # if ip.lower() == 'y':
            #     continue
            # else:
            #     break
        except Exception as e:
            invalid_files.append((im_path, ann_path, e))

    if len(invalid_files) > 0:
        # save error files names
        invalid_files_df = pd.DataFrame(invalid_files, columns=["Img_path", "Annot_path", "Comments"])
        invalid_files_df.to_csv("invalid_files.csv", index=False)

    print("Run Complete !!!")


