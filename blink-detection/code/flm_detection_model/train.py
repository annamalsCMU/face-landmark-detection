import os
import numpy as np
from .flm_model import FaceLandmarkDetection
from ..utils.GenUtils import GenUtils
from ..utils.ImageUtils import ImageUtils


def train(data_path, model_path, model_prefix,
          h, w, c, l, epochs, lr, batch_size):
    flm = FaceLandmarkDetection()

    train_files = flm.getDataPaths(data_path, "train")
    val_files = flm.getDataPaths(data_path, "val")

    # train and save best model
    flm.train(X_dims=(h, w, c),
              y_dim=l * 2,
              train_files=train_files,
              val_files=val_files,
              epochs=epochs,
              batch_size=batch_size,
              lr=lr,
              model_path=model_path,
              model_prefix=model_prefix,
              display=True)

    # save the keras model
    flm.saveKerasModel(model_path=model_path,
                       model_weights=f"{model_prefix}_FLMmodel_Wgts_{epochs}_{batch_size}_{lr}.h5",
                       model_arch=f"{model_prefix}_FLMmodel_Arch_{epochs}_{batch_size}_{lr}.txt",
                       model_nm=f"{model_prefix.upper()}_FLMmodel_{epochs}_{batch_size}_{lr}.h5")


# load saved model
def getPredictionsAndScore(data_path, model_path, model_prefix,
                           model_nm):
    flm = FaceLandmarkDetection()
    #     flm.loadModel(model_path=model_path,
    #                   arch_nm=arch_nm,
    #                   model_nm=model_nm)
    flm.loadKerasModel(model_path=model_path, model_nm=model_nm)

    test_files = flm.getDataPaths(data_path, "test")
    print("Predicting test_files ....")
    preds, score = flm.predict_datagen(data=test_files,
                                       model_path=model_path,
                                       prefix=model_prefix)
    print("-" * 30)
    print("average RMSE:")
    print(score)
    print("-" * 30)
    return preds, score


def showPredicitons(data_path, model_path, pred_name, w, h):
    flm = FaceLandmarkDetection()
    test_files = flm.getDataPaths(data_path, "test")
    genu = GenUtils()
    imu = ImageUtils()
    preds = genu.loadPKL(os.path.join(model_path, pred_name))
    print(preds)

    while True:
        X = [(
            imu.drawAnnotationsOnImg(
                test_files[idx][0],
                imu.getInvTransformCoords(preds[idx].reshape(-1, 2), w, h),
                window="Preview", display=False
            ),
            str(idx) + " || " + test_files[idx][0].split("\\")[-1]
        )
            for idx in np.random.choice(preds.shape[0], 1)]

        imu.display_multiple_img([x[0] for x in X],
                                 titles=[x[1] for x in X],
                                 print_title=True,
                                 rows=4, cols=5, window_size=(15, 8))
        ip = input("Continue? [Q/q to quit] / [other keys to continue]")
        if ip.lower() == 'q':
            break


if __name__ == "__main__":
    data_path = os.path.join("faces_data")

    model_path = os.path.join("code", "flm_detection_model", "model")
    model_prefix = "GPU_Run_3_"

    HEIGHT = 128
    WIDTH = 128
    CHANNELS = 3
    LANDMARKS = 68

    epochs = 10
    lr = 0.001
    batch_size = 32

######################### TRAINING IN GPU ##############################

    train(data_path, model_path, model_prefix,
          HEIGHT, WIDTH, CHANNELS, LANDMARKS, epochs, lr, batch_size)

#####################################################################

#   ##  see predictions
    preds, score = getPredictionsAndScore(data_path=data_path,
                                          model_path=model_path,
                                          model_prefix=model_prefix,
                                          model_nm=f"{model_prefix.upper()}_FLMmodel_{epochs}_{batch_size}_{lr}.h5",
                                          )

#     showPredicitons(data_path=data_path, model_path=model_path,
#                     pred_name=f"{model_prefix}_pred_matrix.pkl",
#                     w=WIDTH, h=HEIGHT)

    pass
