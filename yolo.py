import cv2 as cv
import numpy as np
import time
import torch

from ultralytics import YOLO


class Yolo():

    def __init__(self, use_onnx, onnx_path=None):
        self.model = YOLO('yolov8n-seg.pt') if not use_onnx else YOLO('./yolov8n-seg.onnx' if not onnx_path else onnx_path)
        self.is_onnx = use_onnx
        self.classes = [0]

    def ort_export(self):
        if self.is_onnx:
            print('model is onnx: cannot export')
            exit(1)
        self.model.export(format='onnx')

    def run(self, tensor : torch.Tensor):
        return self.model.predict(source=tensor, classes=self.classes, save=True)[0]
    
    def visualize(self, result, frame):
        masks = result.masks
        main_mask = masks[0].data[0].numpy() * np.uint8(255)
        for mask in masks:
            cv_mask = mask.data[0].numpy() * np.uint8(255)
            main_mask = cv.bitwise_or(main_mask, cv_mask)
            # cv.imshow('img', cv_mask)
            # cv.imshow('main', main_mask)
            # cv.waitKey(500)
        alpha = 0.8
        green_mask = cv.cvtColor(main_mask, cv.COLOR_GRAY2BGR)
        green_mask[:,:,0] = 0
        green_mask[:,:,2] = 0
        masked_frame = np.uint8(alpha * frame + (1.0 - alpha) * green_mask)
        cv.imshow('frame', frame)
        cv.imshow('main_mask', main_mask)
        cv.imshow('masked_frame', masked_frame)
        cv.waitKey(1)


if __name__ == '__main__':
    model = Yolo(False)
    # model.ort_export()

    while True:
        st = time.time()
        result = model.run('https://ultralytics.com/images/bus.jpg')
        print(result)
        masks = result.masks
        cv_masks = []
        print(masks)
        main_mask = masks[0].data[0].numpy() * np.uint8(255)
        for mask in masks:
            cv_mask = mask.data[0].numpy() * np.uint8(255)
            main_mask = cv.bitwise_or(main_mask, cv_mask)
            # cv.imshow('img', cv_mask)
            # cv.imshow('main', main_mask)
            # cv.waitKey(500)
        ft = time.time()
        print(1 / (ft - st))
        cv.imshow('main', main_mask)
        cv.waitKey(20)
