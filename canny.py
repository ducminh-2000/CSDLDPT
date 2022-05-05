import cv2

img = cv2.imread('download.jpeg',0)

img2 = cv2.Canny(img,0,10)

cv2.imwrite('bien.jpg', img2)