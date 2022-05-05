import argparse
import glob
import cv2
import numpy as np
import os


class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        segments = [(0, cX, 0, cY), (cX, w, 0, cY),
                    (cX, w, cY, h), (0, cX, cY, h)]

        (axesX, axesY) = (int((w * 0.75) / 2), int((h * 0.75) / 2))
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return features

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

        cv2.normalize(hist, hist)
        hist = hist.flatten()

        return hist


cd = ColorDescriptor((8, 12, 3))
# # # open the output index file for writing
output = open("file.csv", "w")
for inputFolder in glob.glob("dataset_2/*"):
    index = 1
    for imagePath in glob.glob(inputFolder+"/*.jpg"):
        path = os.path.dirname(imagePath)
        imageID = os.path.basename(path) + "_" + str(index)
        image = cv2.imread(imagePath)
        features = cd.describe(image)
        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))
        index += 1
    # close the index file
output.close()
