import os

try:
    from mmdet.apis import init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from mmpose.utils import adapt_mmdet_pipeline

class PersonDetector():
    def __init__(self, device, base_dir):
        self.BASE_DIR = base_dir
        self.device_name = device

        ckpt_path = 'model/detector/weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
        config_path = 'model/detector/configs/rtmdet_nano_320-8xb32_coco-person.py'
        
        self.abs_ckpt_path = os.path.join(self.BASE_DIR, ckpt_path)
        self.abs_config_path = os.path.join(self.BASE_DIR, config_path)

        # build detector
        self.detector = init_detector(
            self.abs_config_path,
            self.abs_ckpt_path,
            device=self.device_name
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
