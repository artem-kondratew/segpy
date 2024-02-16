import cv2 as cv
from PIL import Image

import torch
import numpy as np

import torchvision.transforms

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

# from torchvision.io import read_image
# from pathlib import Path

from torchvision.utils import draw_segmentation_masks


model = None
weights = None
transforms = None

cv_to_tensor = torchvision.transforms.ToTensor()
pil_to_tensor = torchvision.transforms.PILToTensor()


def proc(frame: cv.Mat):
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as prof:
    pil = Image.fromarray(frame)
    tensor_frame = pil_to_tensor(pil)
    batch = torch.stack([transforms(tensor_frame)])
    output = model(batch)['out']
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    person_mask = [normalized_masks[0, sem_class_to_idx['person']]]
    class_dim = 1
    boolean_person_mask = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
    person_frame = draw_segmentation_masks(tensor_frame, masks=boolean_person_mask, alpha=0.7)
    
    return person_frame.numpy().transpose(1, 2, 0)


def printInfo():
    print('Threads:', torch.get_num_threads())


def main():
    printInfo()

    video_path = './tum_static.mp4'
    local_weights_path = './fcn_resnet50_coco-1167a1af.pth'

    video = cv.VideoCapture(video_path)

    global model, weights, transforms

    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model = model.eval()
    transforms = weights.transforms(resize_size=None)

    print(weights.meta["categories"])

    cnt = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        result = proc(frame)
        cv.imshow('result', result)

        cv.imshow('frame', frame)
        cnt += 1
        print(cnt)
        
        cv.waitKey(20)

    print('eof')


if __name__ == '__main__':
    main()
