
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os
#file = open("table.txt")

model_dim = 32
image_dim=512

dim = image_dim/model_dim
matrix_no_fire = np.zeros((dim,dim))
matrix_fire = np.zeros((dim,dim))
temp= []

print("Wczytywanie modelu")
model = load_model(config.MODEL_PATH)

print("Predykcja obrazów")
firePaths = list(paths.list_images(config.FIRE_PATH))
nonFirePaths = list(paths.list_images(config.NON_FIRE_PATH))


imagePaths = firePaths + nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[0]
image = cv2.imread(imagePaths)

image = cv2.resize(image, (image_dim, image_dim))
image = image.astype("float32") / 255.0

for a in range(dim):
    for b in range (dim):

        w=model_dim*a
        w2=model_dim*a+model_dim
        h=model_dim*b
        h2=model_dim*b+model_dim
        temp = image[w:w2, h:h2]

        preds = model.predict(np.expand_dims(temp, axis=0))[0]
        j = np.argmax(preds)
        matrix_no_fire[a][b]=preds[0]
        matrix_fire[a][b] = preds[1]
        if preds[1]> 0.8:
            image[w:w2,h:h2] = [255,0,0]

cv2.imwrite("Result.jpg", image)

filename1 = "0.txt"
np.savetxt(filename1, matrix_no_fire)
filename2 = "1.txt"
np.savetxt(filename2, matrix_fire)
