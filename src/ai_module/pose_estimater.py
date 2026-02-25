import os

from mmpose.apis import init_model as init_pose_estimator

class PoseEstimator():
    def __init__(self, device, base_dir):
        self.BASE_DIR = base_dir
        self.device_name = device

        # ckpt_path = 'model/pose/weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        # config_path = 'model/pose/configs/rtmpose-m_8xb256-420e_coco-256x192.py'

        # ckpt_path = 'model/pose/weights/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
        # config_path = 'model/pose/configs/rtmpose-s_8xb256-420e_coco-256x192.py'

        # ckpt_path = 'model/pose/weights/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
        # config_path = 'model/pose/configs/rtmpose-t_8xb256-420e_coco-256x192.py'

        # RTMPose Wholebody
        # ckpt_path = 'model/pose/weights/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'
        # config_path = 'model/pose/configs/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'

        # ckpt_path = 'model/pose/weights/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth'
        # config_path = 'model/pose/configs/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'

        # ckpt_path = 'model/pose/weights/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth'
        # config_path = 'model/pose/configs/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py'

        # RTMW Wholebody
        # M (256)
        ckpt_path = 'model/pose/weights/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth'
        config_path = 'model/pose/configs/rtmw-m_8xb1024-270e_cocktail14-256x192.py'

        # L (256)

        # X (256)
        # ckpt_path = 'model/pose/weights/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth'
        # config_path = 'model/pose/configs/rtmw-x_8xb704-270e_cocktail14-256x192.py'

        # L (384)

        # X (384)
        # ckpt_path = 'model/pose/weights/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
        # config_path = 'model/pose/configs/rtmw-x_8xb320-270e_cocktail14-384x288.py'

        self.abs_ckpt_path = os.path.join(self.BASE_DIR, ckpt_path)
        self.abs_config_path = os.path.join(self.BASE_DIR, config_path)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            self.abs_config_path,
            self.abs_ckpt_path,
            device=self.device_name
        )