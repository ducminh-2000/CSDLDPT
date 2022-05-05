import cv2
import glob
import os

from cv2 import INTER_AREA
inputFolder = 'sago'
os.mkdir('Sago_palm')
i=0
for img in glob.glob(inputFolder+"/*.jpg"):
    image = cv2.imread(img)
    imgrs = cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
    cv2.imwrite("Sago_palm/image%i.jpg" %i, imgrs)

    i+=1

cv2.destroyAllWindows()