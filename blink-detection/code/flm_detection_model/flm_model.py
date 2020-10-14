import cv2
import os
from glob import glob
from ..utils.GenUtils import GenUtils
from ..utils.ImageUtils import ImageUtils
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import random

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import *
from tensorflow import keras

class FaceLandmarkDetection:
    def __init__(self):
        self.genu = GenUtils()
        # print(self.genu)
        self.imu = ImageUtils()
        pass

    def getDataPaths(self, path, dir):
        img_files = glob(os.path.join(path,dir, "*.jpg"))
        ann_files = glob(os.path.join(path,dir, "*.pkl"))
        assert(len(img_files) == len(ann_files)), "#Images != #Annotations"
        return list(zip(img_files, ann_files))

    def data_generator(self, data, batch_size=32, shuffle_data=True):
        tot_len = len(data)

        while True:
            if shuffle_data:
                random.shuffle(data)

            for offset in range(0, tot_len, batch_size):
                thisBatch = data[offset:offset+batch_size]
                X, y = [], []
                for img_path, annot_path in thisBatch:
                    im = cv2.imread(img_path)
                    annot = self.genu.loadPKL(annot_path)
                    X.append(im / 255.) # normalize
                    y.append(annot)
                X = np.stack(X)
                y = np.stack(y)

                yield X, y

    def FLMmodel(self, ip_dims, op_units):
        # Approaching Human level facial landmark localization paper
        # based architecture

        i = Input(shape=ip_dims)

        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(i)
#         x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
#         x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dense(units=op_units, activation='sigmoid')(x)

        model = Model(i, x)
        print(model.summary())

        return model

    def train(self, X_dims, y_dim, train_files, val_files,
              epochs, batch_size, lr,
              model_path, model_prefix,
              display=True):

        self.model = self.FLMmodel(X_dims, y_dim)

        self.model.compile(optimizer=Adam(lr=lr),
                           loss=keras.losses.mean_squared_error,
                           metrics=[keras.metrics.mean_squared_error])

        model_nm = "_".join([model_prefix, "FLMmodel_Wgts",
                    str(epochs), str(batch_size), str(lr)])+".h5"
        arch_nm = "_".join([model_prefix, "FLMmodel_Arch",
                    str(epochs), str(batch_size), str(lr)])+".txt"

        checkpoints = ModelCheckpoint(os.path.join(model_path, model_nm), monitor='val_loss',
                                      verbose=1, save_best_only=True, mode='min')
        callback_ls = [checkpoints]

        # data generators and steps
        train_batch_size = len(train_files) if len(train_files) < batch_size else batch_size
        train_gen = self.data_generator(train_files, batch_size=train_batch_size, shuffle_data=True)

        val_batch_size = len(val_files) if len(val_files) < batch_size else batch_size
        val_gen = self.data_generator(val_files, batch_size=val_batch_size, shuffle_data=True)

        steps_per_epoch = len(train_files) // batch_size
        val_steps = len(val_files) // batch_size

        # train
        r = self.model.fit(train_gen,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data=val_gen,
                           validation_steps=val_steps,
                           callbacks=callback_ls,
                           verbose=1)

        # display loss accuracy
        if display:
            plt.plot(r.history["loss"], label="loss")
            plt.plot(r.history["val_loss"], label="val_loss")
            plt.legend()
            plt.show()

        # save model architecture
        with open(os.path.join(model_path, arch_nm), 'w',
                  encoding='utf-8') as arch:
            json.dump(self.model.to_json(), arch)

        pass

    def loadModel(self, model_path, arch_nm, model_nm):
        with open(os.path.join(model_path, arch_nm), 'r',
                  encoding='utf-8') as json_file:
            arch = json.load(json_file)

        # load architecture
        self.model = model_from_json(arch)
        # load weights
        self.model.load_weights(os.path.join(model_path, model_nm))

        # print(self.model.summary())
        pass

    def saveKerasModel(self, model_path, model_arch, model_weights, model_nm):
        with open(os.path.join(model_path, model_arch), 'r',
                  encoding='utf-8') as json_file:
            arch = json.load(json_file)

        # load architecture
        model = model_from_json(arch)
        # load weights
        model.load_weights(os.path.join(model_path, model_weights))

        model.save(os.path.join(model_path, model_nm))
        print(f"Saved successfully in {model_path} as {model_nm}")

    def loadKerasModel(self, model_path, model_nm):
        self.model = load_model(os.path.join(model_path, model_nm))
        pass

    def predict_one(self, x):
        x = np.expand_dims(x, axis=0)
        y_pred = self.model.predict(x)
        return y_pred

    def predict_datagen(self, data, model_path="", prefix="", savePreds=True):
        if len(data) == 0:
            print("No data to feed test generator")
            return None, None

        test_gen = self.data_generator(data, batch_size=1, shuffle_data=False)
        preds = self.model.predict(test_gen, steps=len(data))

        if savePreds:
            print("saving the predictions matrix....")
            pred_filename = f"{prefix}_pred_matrix"
            self.genu.dumpPKL(preds, model_path, pred_filename)
            print(f"predictions saved in {model_path} with the name {pred_filename}")

        return preds, self.score(data, preds)


    def score(self, actuals_path, preds):
        y = self.genu.getActualsFromPKLs(actuals_path)
        rmse = (np.sqrt(np.square(y - preds))).mean(axis=0)
        avg_rmse = rmse.mean()
        return avg_rmse

    # def testImages(self, X, y, name, w, h):
    #     idx = np.random.choice(y.shape[0])
    #     y = y[idx].reshape(-1, 2)
    #     y = self.imu.getInvTransformCoords(y, w, h)
    #     self.imu.drawAnnotationsOnImg(X[idx], y, window=name, display=True)
    #     pass
