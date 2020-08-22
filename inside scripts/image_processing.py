import argparse
import cv2
import os
import glob

os.chdir("more_fajer")
filenames=glob.glob("**")
print(filenames)
coordinates = []
cropping = False

def drawing_rect(event, x, y, flags, parameters):
    global coordinates, cropping

    if event ==  cv2.EVENT_LBUTTONDOWN:
        coordinates = [(x,y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        coordinates.append((x,y))
        cropping = False

        cv2.rectangle(image, coordinates[0], coordinates[1], (0,255,0), 2)
        cv2.imshow("image", image)

i=0
for file in filenames:
    i=i+1
    print(i)
    image = cv2.imread(file)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", drawing_rect)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
        elif key == ord("c"):
            break

    if len(coordinates) == 2:
        roi = clone[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]
        #cv2.imshow("ROI", roi)
        rotate1roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        rotate2roi = cv2.rotate(rotate1roi, cv2.ROTATE_90_CLOCKWISE)
        rotate3roi = cv2.rotate(rotate2roi, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f"{i}.jpg", roi)
        cv2.imwrite(f"{i}a.jpg", rotate1roi)
        cv2.imwrite(f"{i}ab.jpg", rotate2roi)
        cv2.imwrite(f"{i}abc.jpg", rotate3roi)

    cv2.destroyAllWindows