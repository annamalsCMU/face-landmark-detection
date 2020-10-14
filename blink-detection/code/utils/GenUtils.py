import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import h5py
import json
from tensorflow.keras.models import Model, model_from_json

class GenUtils:
    def __init__(self):
        pass

    def loadPKL(self, path):
        with open(path, "rb") as pkl:
            obj = pickle.load(pkl)
        return obj

    def dumpPKL(self, obj, path, name, display=False):
        with open(os.path.join(path, name+".pkl"), 'wb') as pkl:
            pickle.dump(obj, pkl)
        if display:
            print(f"Save Successful filename: {name}.pkl in path: {path}")
        pass

    def getActualsFromPKLs(self, data_paths):
        y = []
        for _,path in data_paths:
            y.append(self.loadPKL(path))
        return np.array(y)

    def moveFiles(self, indices, src_file_list, dest):
        for i in tqdm(indices):
            src = src_file_list[i]
            shutil.move(src, dest)
        print("Move Successful !!")
        pass

    def saveAsHDF5(self, path, name, data_ls, display=False):
        hf = h5py.File(os.path.join(path, name+".h5"), 'w')
        for data_nm, data in data_ls:
            hf.create_dataset(data_nm, data=data)
        hf.close()
        if display:
            print(f"Save Successful filename: {name}.h5 in path: {path}")
        pass

    def loadFromHDF5(self, path, datasets):
        hf = h5py.File(path, 'r')
        data = []
        for name in datasets:
            data.append(hf.get(name))
        hf.close()
        return data


if __name__ == "__main__":
    pass