# USAGE
# python predict_fire.py

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os
import glob

img_array = []

model_dim = 32
image_dim=512

dim = image_dim/model_dim


print("Wczytywanie modelu")
model = load_model(config.MODEL_PATH)


print("Predykcja video")



vidcap = cv2.VideoCapture('video2.mp4')
success ,image = vidcap.read()
count = 0

video = cv2.VideoCapture('video2.mp4')
windowName = "Detekcja ognia"

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_time = round(1000 / fps)

keepProcessing = True

while keepProcessing:
    #cv2.imwrite("frame%d.jpg" % count, image)
    success,image = vidcap.read()
    print ('Read a new frame: ', success)
    count += 1
    if not success:
        print("... end of video file reached");
        break;
    output = image.copy()
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


    filename = "{}.png".format(count)
    p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
    cv2.imwrite(p, image)


for filename in glob.glob('output\examples\*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

