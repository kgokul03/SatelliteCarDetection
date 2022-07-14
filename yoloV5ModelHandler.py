from pathlib import Path
import numpy as np

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_img_size, colorstr, cv2,
                           increment_path, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
# from ts.torch_handler.base_handler import BaseHandler

class YOLOV5ModelHandler():
    imgsz=(640, 640) # inference size (height, width)
    conf_thres=0.3  # confidence threshold
    iou_thres=0.45 # NMS IOU threshold
    max_det=1000  # maximum detections per image
    line_thickness=3  # bounding box thickness (pixels)
    half=False  # use FP16 half-precision inference
    dnn=False
    bs = 1
    name='exp'
    project = 'runs/detect'
        
    def preprocess(self, input_image):
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=False)  # increment run
        (self.save_dir / 'heatMap').mkdir(parents=True, exist_ok=True)
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        self.dataset = LoadImages(input_image, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        path, im, im0s, vid_cap, s = list(enumerate(self.dataset))[0][1]
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None] 
        return im

    def inference(self, model_input):
        return self.model(model_input, augment=False, visualize=False)

    def postprocess(self,image,data):
            # NMS
            path, im, im0s, vid_cap, s = list(enumerate(self.dataset))[0][1]
            im = image
            data = non_max_suppression(data, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)

            # Process predictions
                
            det = list(enumerate(data))[0][1]
            p, im0, im1, frame = path, im0s.copy(),np.full((im0s.shape[0], im0s.shape[1]), 0, dtype=np.uint8), getattr(self.dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            hm_path = str(self.save_dir / 'heatMap' / p.name)
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            annotatorHeatmap = Annotator(im1, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label=None
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    annotatorHeatmap.point_label(xywh, label, (255,255,255))

            # Stream results
            im0 = annotator.result()
            im1 = annotatorHeatmap.result()

            cv2.imwrite(save_path, im0)
            cv2.imwrite(hm_path, im1)

            LOGGER.info(f'{s}Done.')# ({t3 - t2:.3f}s)')
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return len(det),im0,im1

    def initialize(self):
        self.weights = 'best.pt'
        self.data = 'data/cowc.yaml'
        self.device = torch.device('cpu')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

    def __init__(self):
        self.weights = None
        self.data = None
        self.device = None
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.dataset = None

    def handle(self,input_image):
        self.initialize()
        img = self.preprocess(input_image)
        data = self.inference(img)
        car_count,box_img_cv2_array,peak_img_cv2_array = self.postprocess(img,data)
        return car_count,box_img_cv2_array,peak_img_cv2_array

obj1 = YOLOV5ModelHandler()
obj1.handle("../images/input2.jpg");