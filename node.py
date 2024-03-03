import torch

import torchvision.transforms
from torchvision.utils import draw_segmentation_masks

import cv2 as cv
import numpy as np
from PIL import Image

import resnet


class Node:

    def __init__(self, video_path, model, use_onnx=True):
        self.pil_to_tensor = torchvision.transforms.PILToTensor()
        self.model = model
        self.model.eval()
        self.run = self.model.ort_run if use_onnx else self.model.run
        self.cap = cv.VideoCapture(video_path)
        
    def cv2torch(self, frame : cv.Mat) -> torch.Tensor:
        return self.pil_to_tensor(Image.fromarray(frame))

    def run(self, tensor : torch.Tensor) -> torch.Tensor:
        return self.run(tensor)
    
    def visualize(self, frame : cv.Mat, tensor : torch.Tensor, output : torch.Tensor):
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.model.get_classes())}
        person_mask = [normalized_masks[0, sem_class_to_idx['person']]]
        class_dim = 1
        boolean_person_mask = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
        cv_mask = boolean_person_mask.numpy().transpose(1, 2, 0) * np.uint8(255)
        cv.imshow('mask', cv_mask)
        person_frame = draw_segmentation_masks(tensor, masks=boolean_person_mask, alpha=0.7)
        result = person_frame.numpy().transpose(1, 2, 0)
        cv.imshow('result', result)
        cv.imshow('frame', frame)
        cv.waitKey(20)
        return result
    
    def spin(self) -> bool:
        ret, frame = self.cap.read()
        if not ret:
            return
        tensor = self.cv2torch(frame)
        output = self.run(tensor)
        self.visualize(frame, tensor, output)
        return self.cap.isOpened()
