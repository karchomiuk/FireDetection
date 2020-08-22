
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

print("Wczytywanie modelu")
model = load_model(config.MODEL_PATH)

print("Predykcja obrazów")
imagePaths = list(paths.list_images(config.FLICKR_PATH))
file = open("flickr_result.txt", "w+")


for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    #print(type(imagePath))

    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0


    preds = model.predict(np.expand_dims(image, axis=0))[0]
    j = np.argmax(preds)
    label = config.CLASSES[j]
    if preds[1] > 0.9:
        file.write(imagePath)
        file.write(label)


file.close()