import numpy as np
import cv2 as cv
import time
import torch

import yolo


def main():
    torch.set_num_threads(4)
    print('threads:', torch.get_num_threads())

    onnx_path = './yolov8n-seg.onnx'
    video_path = './tum_static.mp4'

    cap = cv.VideoCapture(video_path)
    
    model = yolo.Yolo(use_onnx=False, onnx_path=onnx_path)
    # model.ort_export()
    
    t = time.time()
    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        st = time.time()
        result = model.run(frame)
        model.visualize(result, frame)
        ft = time.time()
        print(1 / (ft - st))

    print('eof')


if __name__ == '__main__':
    main()
