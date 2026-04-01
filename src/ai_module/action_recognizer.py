import os
import sys

import mmengine
sys.path.append('/mnt/hdd1/003_SOLUTION/industrial_safety/mmaction2')
sys.path.append('/mnt/hdd1/003_SOLUTION/industrial_safety/ctr-gcn')
sys.path.append('/mnt/hdd1/001_AI/project/industrial_safety/solution/model/action/test')
# sys.path.append('/mnt/hdd1/003_SOLUTION/industrial_safety/model/action/ctrgcn')
from mmaction.apis import init_recognizer


class ActionRecognizer():
    def __init__(self, device, base_dir):
        self.BASE_DIR = base_dir
        self.device_name = device

        # ------------------------------------------------------------------------
        # STGCN++
        # -------
        # joint
        # ckpt_path = 'model/action/stgcnpp/joint/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth'
        # config_path = 'model/action/stgcnpp/joint/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'
        # test 1
        ckpt_path = 'model/action/test/best_acc_top1_epoch_12.pth'
        config_path = 'model/action/test/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_jh_test.py'
        # test 2
        # ckpt_path = 'model/action/test/best_acc_top1_epoch_5.pth'
        # config_path = 'model/action/test/stgcnpp_2d-joint-xsub_S002.py'
        
        # bone
        # ckpt_path = 'model/action/stgcnpp/bone/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth'
        # config_path = 'model/action/stgcnpp/bone/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py'

        # joint-motion
        # ckpt_path = 'model/action/stgcnpp/joint_motion/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth'
        # config_path = 'model/action/stgcnpp/joint_motion/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py'

        # bone-motion
        # ckpt_path = 'model/action/stgcnpp/bone_motion/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'
        # config_path = 'model/action/stgcnpp/bone_motion/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'

        # ------------------------------------------------------------------------
        # CTR-GCN
        # ------
        # ckpt_path = 'model/action/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth'
        # config_path = 'model/action/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'
        # ------------------------------------------------------------------------
        # PoseC3D
        # ------
        # keypoint
        # ckpt_path = 'model/action/posec3d/keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth'
        # config_path = 'model/action/posec3d/keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'

        # limb
        # ckpt_path = 'model/action/posec3d/limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb_20220815-af2f119a.pth'
        # config_path = 'model/action/posec3d/limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.py'
        
        # ------------------------------------------------------------------------

        label_map_path = 'model/action/label_map_ntu60.txt'
        # label_map_path = 'model/action/label_map_industry_safety3.txt'

        abs_ckpt_path = os.path.join(self.BASE_DIR, ckpt_path)
        abs_config_path = os.path.join(self.BASE_DIR, config_path)
        abs_label_map_path = os.path.join(self.BASE_DIR, label_map_path)

        # config = mmengine.Config.fromfile(abs_config_path)

        # build action recognizer
        self.action_recognizer = init_recognizer(abs_config_path,
                                                 abs_ckpt_path,
                                                 device='cuda:0')  # or device='cpu'
        
        self.label_map = abs_label_map_path