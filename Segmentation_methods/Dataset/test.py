import pickle
import numpy as np
import os

workdir = os.getenv("WORKDIR_Segmentation")
print("Mon répertoire :", workdir)

with open("/home/rchaki/Bureau/segmentation of plant diseases/Segmentation_methods/Dataset/SyntheticFluoDataset_DiseasedPlants/Y_PlantDiseaseLesionsLabels.pickle", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Shape:", data.shape)
print("Dtype:", data.dtype)