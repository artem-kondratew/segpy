import torch
import onnx
import onnxruntime
import cv2 as cv
import numpy as np
from PIL import Image

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import PILToTensor
from torchvision.utils import draw_segmentation_masks


class Resnet():    

    def __init__(self, weights_path=None):
        self.weights = FCN_ResNet50_Weights.DEFAULT if weights_path == None else exec("print('TODO: load weights from path'); exit(1)")
        self.transforms = self.weights.transforms(resize_size=None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fcn_resnet50(weights=self.weights).to(self.device)
        self.ort_session = None
        self.classes = self.weights.meta["categories"]
        print('device:', self.device)
        self.pil_to_tensor = PILToTensor()

    def get_classes(self) -> list:
        return self.classes

    def print_classes(self):
        print(c for c in self.classes)

    def eval(self):
        self.model.eval()

    def cv2torch(self, frame : cv.Mat) -> torch.Tensor:
        return self.pil_to_tensor(Image.fromarray(frame))

    def create_batch(self, tensor : torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transforms(tensor)]).to(self.device)
    
    def run(self, tensor : torch.Tensor) -> torch.Tensor:
        output = self.model(self.create_batch(tensor))['out'].cpu()
        return output
        
    def export_to_onnx_fp32(self, path : str):
        print('exporting to onnx...')
        dummy_input = torch.randn(1, 3, 480, 640).to(self.device)
        torch.onnx.export(self.model,
                          dummy_input,
                          path,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}})
        print('successfully exported')

    def export_to_onnx_int8(self, path : str):
        pass

    def ort_check(self, path : str):
        try:
            print('checking onnx model...')
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            print('successsfully checked')
        except FileNotFoundError:
            print('no such file')

    def ort_load(self, path : str) -> bool:
        try:
            self.ort_session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())
            print('onnxruntime device:', self.ort_session.get_providers())
            print('onnxruntime inputs:', [o.name for o in self.ort_session.get_inputs()])
            print('onnxruntime outputs:', [o.name for o in self.ort_session.get_outputs()])
            return True
        except FileNotFoundError:
            print('no such file')
            return False

    def to_numpy(self, tensor) -> np.array:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def ort_run(self, tensor : torch.Tensor) -> torch.Tensor:
        if self.ort_session == None:
            print('no ort session loaded')
            return None
        output = self.ort_session.run(['output'], {'input': self.to_numpy(self.create_batch(tensor))})
        return torch.tensor(output[0]).cpu()

    def visualize(self, frame : cv.Mat, tensor : torch.Tensor, output : torch.Tensor):
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.get_classes())}
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
