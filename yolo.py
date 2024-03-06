from ultralytics import YOLO as _YOLO
import numpy as np
import cv2 as cv
import torch


model = _YOLO('yolov8n-seg.pt')
# model = _YOLO('yolov8n-seg.onnx')

# model.export(format='onnx')

i = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

while True:
    results = model('https://ultralytics.com/images/bus.jpg').to(device)
    result = results[0]
    masks = result.masks
    mask = masks[0] if i else masks[1]
    i = not i
    img = mask.data[0].numpy() * np.uint8(255)
    cv.imshow('img', img)
    cv.waitKey(20)
    