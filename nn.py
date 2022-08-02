import torch
import yolov5.detect as yolo
import torchvision.models as models
import cv2 as cv

yol = yolo.run(weights='bestNano.pt',source='images/piano15.jpg',view_img=True)

cv.imshow("tee",yol)
cv.waitKey()

