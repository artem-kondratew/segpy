import time
import torch

import resnet
import node


def main():
    torch.set_num_threads(4)
    print('threads:', torch.get_num_threads())

    onnx_fp32_path = './resnet_fp32.onnx'
    video_path = './tum_static.mp4'

    model = resnet.Resnet()

    # model.export_to_onnx_fp32(onnx_fp32_path)

    model.ort_load(onnx_fp32_path)

    n = node.Node(video_path, model, use_onnx=True)

    t = time.time()
    cnt = 0

    while n.spin():
        dt = time.time() - t
        t = time.time()
        cnt += 1
        print(cnt, 'dt =', dt)

    print('eof')


if __name__ == '__main__':
    main()
