import time
import torch
import cv2 as cv

import resnet


def main():
    torch.set_num_threads(4)
    print('threads:', torch.get_num_threads())

    onnx_fp32_path = './resnet_fp32.onnx'
    video_path = './tum_static.mp4'

    cap = cv.VideoCapture(video_path)
    

    model = resnet.Resnet()
    model.eval()

    model.export_to_onnx_fp32(onnx_fp32_path)

    model.ort_check(onnx_fp32_path)

    model.ort_load(onnx_fp32_path)
    
    t = time.time()
    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        t = time.time()
        tensor = model.cv2torch(frame)
        output = model.run(tensor)
        # output = model.ort_run(tensor)
        model.visualize(frame, tensor, output)
        nt = time.time()
        dt = nt - t
        t = nt
        cnt += 1
        print(cnt, 'dt =', dt)

    print('eof')


if __name__ == '__main__':
    main()
