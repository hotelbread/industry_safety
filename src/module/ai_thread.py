import cv2
import torch
import numpy as np
import traceback
from time import sleep, perf_counter

import sys
from pathlib import Path

from src.ai_module.pose_estimater import PoseEstimator
from src.ai_module.person_detector import PersonDetector
from src.ai_module.action_recognizer import ActionRecognizer

from src.misc.duration import Duration

from src.trackers.ocsort import OCSort

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
from mmpose.evaluation.functional import nms
from mmpose.apis import inference_topdown
from mmaction.apis import inference_skeleton, inference_recognizer

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

class AiThread(QThread):
    signalSetImage = Signal(QImage)
    
    def __init__(self, RESOURCE_DIR, parent=None):
        super().__init__()
        self.parent = parent
        self.RESOURCE_DIR = RESOURCE_DIR

        # print()
        print('=================================================')
        
        # Resource relative path
        # self.RESOURCE_DIR = get_resource_dir()
        
        self.parent.log(f'[Thread][AI] Initialize AI Thread')
        """set up the AI model"""
        # self.device_name =  cpu'        # for cpu
        self.device_name = 'cuda:0'     # for GPU Desktop
        self.person_detector = PersonDetector(self.device_name, self.RESOURCE_DIR)
        self.pose_estimator = PoseEstimator(self.device_name, self.RESOURCE_DIR)
        self.action_recognizer = ActionRecognizer(self.device_name, self.RESOURCE_DIR)


        # self.frame_loader = frame_loader
        # self.person_detector = person_detector
        # self.pose_estimator = pose_estimator
        # self.action_recognizer = action_recognizer

        self._running = True
        self.prevTime = 0

        self.det_cat_id = 0
        self.bbox_thr = 0.5     # origin(mmpose src) = 0.3
        self.nms_thr = 0.3      # origin(mmpose src) = 0.3
        self.kpt_thr = 0.5

        self.bbox_alpha = 0.1
        self.line_alpha = 0.5
        self.skeleton_alpha = 0.6
        self.text_alpha = 0.3

        # -------------------
        # action
        # self.window_size = 32
        # self.window_size = 48
        self.window_size = 64
        # self.window_size = 100
        self.action_thr = 0.1
        self.K = 3
        self.label_map = [x.strip() for x in open(self.action_recognizer.label_map).readlines()]

        self.alpha = 0.7
        # -------------------

        self.line_width = 3
        self.circle_radius = 5
        # self.line_width = self.frame_loader.line_width
        # self.circle_radius = self.frame_loader.circle_radius

        self.det_timer = Duration('[AI][Detection]')
        self.pose_timer = Duration('[AI][Pose estimation]')
        self.multi_pose_timer = Duration('[AI][Multi-Person Pose estimation]')
        self.action_timer = Duration('[AI][Action Recognition]')

        self.det_duration = []
        self.pose_duration = []
        self.multi_pose_duration = []
        self.action_duration = []

        print()
        self.parent.log(f'[Thread][AI][Detection] categories id : {self.det_cat_id}')
        self.parent.log(f'[Thread][AI][Detection] bbox thr : {self.bbox_thr}')
        self.parent.log(f'[Thread][AI][Detection] nms thr : {self.nms_thr}')
        print()
        self.parent.log(f'[Thread][AI][Pose Estimation] kpt thr : {self.kpt_thr}')
        print()
        self.parent.log(f'[Thread][AI][Visualization] line width : {self.line_width} px')
        self.parent.log(f'[Thread][AI][Visualization] cicle radius : {self.circle_radius} px')
        print()

        # -------------------
        # for tracking
        self.thConfTrack = 0.5
        self.thIouTrack = 0.5

        self.tracker_list = OCSort(
            det_thresh=self.thConfTrack,
            iou_threshold=self.thIouTrack,
            min_hits=0,
            max_age=3,
            use_byte=False
        )

        if hasattr(self.tracker_list, 'model'):
            if hasattr(self.tracker_list.model, 'warmup'):
                self.tracker_list.model.warmup()
            else:
                pass
        else:
            pass
        # -------------------

    def run(self):
        pose_results = dict()
        while self._running:
            try:
                frame = self.frame_loader.get_frame()
                if frame is None:
                    # print('skip !!!!!!!!')
                    continue
                print()
                print('=================================================')
                rgb_img = frame[:, :, ::-1]

                h, w, _ = rgb_img.shape

                self.det_timer.set_prev()

                det_result = inference_detector(self.person_detector.detector, frame)
                
                self.det_timer.calc_elapsed()
                self.det_duration.append(self.det_timer.get_elapsed())
                # self.det_timer.print_fps()
                # self.det_timer.print_sec()
                # print()

                pred_instance = det_result.pred_instances.cpu().numpy()

                output_bboxes = pred_instance.bboxes # (N, 4)
                output_labels = pred_instance.labels # (N,)
                output_scores = pred_instance.scores # (N,)
                
                person_idx_lst = np.logical_and(output_labels == self.det_cat_id,
                                                output_scores > self.bbox_thr)
                if not True in person_idx_lst:
                    result_img = np.ascontiguousarray(rgb_img)
                    qimg = self.convert_qimage(result_img)
                    self.signalSetImage.emit(qimg)
                    continue

                people_bboxes = output_bboxes[person_idx_lst] # (M, 4)
                people_labels = output_labels[person_idx_lst] # (M,)
                people_scores = output_scores[person_idx_lst] # (M,)
                
                # (M, 5)
                pose_bboxes = np.concatenate((people_bboxes,
                                            people_scores[:, None]),
                                            axis=1)
                
                # (M, 6) -> [ [x1, y1, x2, y2, conf, cls], ... ]
                track_bboxes = np.concatenate((pose_bboxes,
                                            people_labels[:, None]),
                                            axis=1)

                bboxes_info = pose_bboxes[nms(pose_bboxes, self.nms_thr), :]
                # pose_input_bbox = bboxes_info[:,:4]

                # -------------------------------------------------------
                # OC-SORT
                try:
                    if len(track_bboxes) > 0:
                        # (M, 7) -> [ [x1, y1, x2, y2, idx, cls, conf], ... ]
                        tracked_bboxes = self.tracker_list.update(torch.tensor(track_bboxes), rgb_img)
                    else:
                        pass
                except:
                    print('[ERROR][AI] {0}'.format(traceback.format_exc()))

                # -------------------------------------------------------

                multi_pose_results = []
                now_frame_id = []

                self.multi_pose_timer.set_prev()

                for single_bbox in tracked_bboxes:
                    identification = single_bbox[4]
                    now_frame_id.append(identification)
                    if not identification in pose_results.keys():
                        pose_results[identification] = []

                    # original (coco 17key)
                    # id = np.expand_dims(np.array([identification]*17), 0)
                    # id = np.expand_dims(id, 2)
                    # for whole body
                    id = np.expand_dims(np.array([identification]*133), 0)
                    id = np.expand_dims(id, 2)

                    self.pose_timer.set_prev()

                    single_pose_result = inference_topdown(self.pose_estimator.pose_estimator,
                                                           rgb_img,
                                                           single_bbox[None, :4])
                    
                    self.pose_timer.calc_elapsed()
                    self.pose_duration.append(self.pose_timer.get_elapsed())
                    # self.pose_timer.print_fps()
                    # self.pose_timer.print_sec()
                    # print()

                    # print()
                    # print('single_pose_result (type) :', type(single_pose_result))
                    # print('single_pose_result (len) :', len(single_pose_result))
                    # print()
                    # print()
                    # print('single_pose_result[0] (type) :', type(single_pose_result[0]))
                    # print('single_pose_result[0] (keys) :', single_pose_result[0].keys())
                    # print()
                    # print()
                    # print('single_pose_result[0][pred_instances] (type) :', type(single_pose_result[0].get('pred_instances')))
                    # print('single_pose_result[0][pred_instances] (keys) :', single_pose_result[0].get('pred_instances').keys())
                    # print()
                    # print('---------------------------------------------------------------------')
                    # print()
                    # print('single_pose_result[0][pred_instances][\'keypoints_visible\'] (type) :', type(single_pose_result[0].get('pred_instances')["keypoints_visible"]))
                    # print('single_pose_result[0][pred_instances][\'keypoints_visible\'] (shape) :', single_pose_result[0].get('pred_instances')["keypoints_visible"].shape)
                    # print()
                    # print('single_pose_result[0][pred_instances][\'keypoints\'] (type) :', type(single_pose_result[0].get('pred_instances')["keypoints"]))
                    # print('single_pose_result[0][pred_instances][\'keypoints\'] (shape) :', single_pose_result[0].get('pred_instances')["keypoints"].shape)
                    # print()
                    # print('single_pose_result[0][pred_instances][\'keypoint_scores\'] (type) :', type(single_pose_result[0].get('pred_instances')["keypoint_scores"]))
                    # print('single_pose_result[0][pred_instances][\'keypoint_scores\'] (shape) :', single_pose_result[0].get('pred_instances')["keypoint_scores"].shape)
                    # print()
                    # print('single_pose_result[0][pred_instances][\'bboxes\'] (type) :', type(single_pose_result[0].get('pred_instances')["bboxes"]))
                    # print('single_pose_result[0][pred_instances][\'bboxes\'] (shape) :', single_pose_result[0].get('pred_instances')["bboxes"].shape)
                    # print()
                    # print('single_pose_result[0][pred_instances][\'bbox_scores\'] (type) :', type(single_pose_result[0].get('pred_instances')["bbox_scores"]))
                    # print('single_pose_result[0][pred_instances][\'bbox_scores\'] (shape) :', single_pose_result[0].get('pred_instances')["bbox_scores"].shape)
                    # print()
                    # print()
                    # print()

                    pose_result = {
                        'bboxes': single_pose_result[0].get("pred_instances").get('bboxes'),
                        'bbox_scores': single_pose_result[0].get("pred_instances").get('bbox_scores'),
                        'keypoints_visible': single_pose_result[0].get("pred_instances").get('keypoints_visible'),
                        # original (coco 17key)
                        # 'keypoints': single_pose_result[0].get("pred_instances").get('keypoints'),
                        # 'keypoint_scores': single_pose_result[0].get("pred_instances").get('keypoint_scores')
                        # for whole body
                        'keypoints': single_pose_result[0].get("pred_instances").get('keypoints')[:,:17,:],
                        'keypoint_scores': single_pose_result[0].get("pred_instances").get('keypoint_scores')[:,:17]
                    }

                    pose_results[identification].append(pose_result)

                    # (1, 17, 2)
                    person_keypoints = single_pose_result[0].get('pred_instances').get('keypoints')
                    # (1, 17)
                    person_conf = single_pose_result[0].get('pred_instances').get('keypoint_scores')
                    # (1, 17, 1)
                    person_conf = np.expand_dims(person_conf, 2)
                    # (1, 17, 3)
                    person_info = np.concatenate((person_keypoints, person_conf), axis=2)
                    # (1, 17, 4)
                    person_info = np.concatenate((person_info, id), axis=2)

                    # (17,4)
                    person_info = np.squeeze(person_info)
                    multi_pose_results.append(person_info)

                # (M, 17, 4)
                keypoints_info = np.array(multi_pose_results)

                keypoints = keypoints_info[:,:,:2]
                keypoints_scores = keypoints_info[:,:,2]

                self.multi_pose_timer.calc_elapsed()
                self.multi_pose_duration.append(self.multi_pose_timer.get_elapsed())

                # -------------------------------------------------------
                # Action
                for key, result_lst in pose_results.items():
                    if key in now_frame_id:
                        action_label_results = []
                        # print(f'ID {int(key)} : {len(result_lst)}')
                        if len(result_lst) >= self.window_size:
                            action_input = result_lst[-(self.window_size):]

                            self.action_timer.set_prev()

                            action_result = inference_skeleton(self.action_recognizer.action_recognizer, action_input, (h, w))
                            # action_result = inference_recognizer(self.action_recognizer, pose_results)

                            self.action_timer.calc_elapsed()
                            self.action_duration.append(self.action_timer.get_elapsed())
                            # self.action_timer.print_fps()
                            # self.action_timer.print_sec()

                            action_scores = action_result.pred_score

                            while True:
                                max_pred_index = action_scores.argmax().item()
                                action_label = self.label_map[max_pred_index]
                                action_conf = action_scores[max_pred_index].item()
                                if action_conf > self.action_thr:
                                    action_label_results.append({'label': action_label, 'conf': action_conf})
                                    # Top K (K = 3)
                                    if len(action_label_results) == self.K:
                                        break
                                    action_scores[max_pred_index] = 0
                                else:
                                    break

                            filtered_bbox = [row for row in tracked_bboxes if row[4] == int(key)]

                            if len(filtered_bbox) > 0:
                                rgb_img = self.visualize_action(rgb_img, filtered_bbox[0], action_label_results)

                # -------------------------------------------------------
                # Visualize
                rgb_img = self.visualize_bbox(rgb_img, tracked_bboxes)

                rgb_img = self.visualize_pose(rgb_img, keypoints, keypoints_scores, self.kpt_thr)

                fps = self.getFps()
                rgb_img = self.draw_fps(rgb_img, fps)

                # -------------------------------------------------------
                # to screen
                result_img = np.ascontiguousarray(rgb_img)
                qimg = self.convert_qimage(result_img)
                self.signalSetImage.emit(qimg)

                # fps = self.getFps()
                print()
                print('-----------------------------------------')
                print(f'Total sec : {(1/fps):.3f}')
                print(f'Total fps : {fps:.1f}')
                print()
                det_sec = sum(self.det_duration) / len(self.det_duration)
                pose_sec = sum(self.pose_duration) / len(self.pose_duration)
                multi_pose_sec = sum(self.multi_pose_duration) / len(self.multi_pose_duration)
                action_sec = sum(self.action_duration) / len(self.action_duration)

                print(f'Det sec : {(det_sec):.3f}')
                print(f'Det fps : {(1/det_sec):.1f}')
                print()
                print(f'Pose sec : {(pose_sec):.3f}')
                print(f'Pose fps : {(1/pose_sec):.1f}')
                print()
                print(f'Multi Pose sec : {(multi_pose_sec):.3f}')
                print(f'Multi Pose fps : {(1/multi_pose_sec):.1f}')
                print()
                print(f'Action sec : {(action_sec):.3f}')
                print(f'Action fps : {(1/action_sec):.1f}')
                print()
                

            except:
                print('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))
                continue


    def getFps(self):
        elapsed = perf_counter() - self.prevTime
        if self.frame_loader.mode == 'webcam':
            fps = 1/elapsed
        else:
            delay = max(0, self.frame_loader.delay_limit - elapsed)
            sleep(delay)
            print(f'delay : {delay:.5f}')
            if delay == 0:
                fps = 1/elapsed
            else:
                fps = self.frame_loader.video_fps
        self.prevTime = perf_counter()

        return fps


    def convert_qimage(self, frame):
        h, w, ch = frame.shape
        
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        bytes_per_line = ch * w
        
        q_image = QImage(
            frame.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        return q_image.copy()
    
    def visualize_bbox(self, img, bboxes):
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (255, 153, 0), (0, 153, 255), (153, 255, 0), (0, 255, 153),
                   (153, 153, 153),(255, 255, 255)]

        bbox_len = bboxes.shape[0]
        palette_len = len(palette)
        cidx = 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        # font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.8
        thickness = 2

        for bidx in range(bbox_len):
            x_min = int(bboxes[bidx][0])
            y_min = int(bboxes[bidx][1])
            x_max = int(bboxes[bidx][2])
            y_max = int(bboxes[bidx][3])

            if bboxes.shape[-1] == 4:
                img =  cv2.rectangle(img.copy(), (x_min, y_min), (x_max, y_max),
                                     (0, 255, 0), 3)
            else:
                try:
                    id = int(bboxes[bidx][4])
                    cidx = (id-1) % palette_len

                    img = self.draw_transparency_rect(img, x_min, y_min, x_max, y_max, palette[cidx],
                                                      -1, self.bbox_alpha)

                    # img =  cv2.rectangle(img.copy(), (x_min, y_min), (x_max, y_max), palette[cidx], 3)
                    img = self.draw_transparency_rect(img, x_min, y_min, x_max, y_max, palette[cidx],
                                                      3, self.line_alpha)

                    img = self.draw_transparency_rect(img,
                                                      x_min+3, y_min+3,
                                                      x_min+70, y_min+30,
                                                      (0,0,0), -1, self.alpha)
                    
                    img = self.draw_transparency_text(img, 'ID: {0}'.format(id), (x_min+5, y_min + 25), font,
                                                      font_scale, palette[cidx], thickness, self.text_alpha)
                except:
                    self.parent.log(f'[Thread][AI][Visualization][Error] Fail to draw multi-person bbox')
                    self.parent.log('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))

        return img
    
    
    def visualize_pose(self, frame, keypoints, scores, thr=0.5):
        palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                (255, 153, 255), (153, 204, 255), (255, 102, 255),
                (255, 51, 255), (102, 178, 255),
                (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
                (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                (0, 0, 255), (255, 0, 0), (255, 255, 255),]
        
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        link_color = [
            0, 0, 0, 0, 7, 7,
            7, 9, 9, 9, 9, 9, 16, 
            16, 16, 16, 16, 16, 16
        ]
        point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]
        
        # ------------------------------------------------------------------------------------
        # for whole body
        hand_skeleton = [(9, 91), (91, 95), (91,99), (91, 103), (91, 107), (91, 111),
                         (10, 112), (112, 116), (112, 120), (112, 124), (112, 128), (112, 132)]
        skeleton = skeleton + hand_skeleton

        hand_link_color = [0, 0, 4, 4, 4, 4,
                           16, 16, 12, 12, 12, 12]
        link_color = link_color + hand_link_color

        while(len(point_color) != 133):
            point_color.append(1)

        hand_idx = [91, 95, 99, 103, 107, 111,
                    112, 116, 120, 124, 128, 132]
        for i in hand_idx:
            point_color[i] = 9
        # ------------------------------------------------------------------------------------
        

        scale = 1
        keypoints = (keypoints * scale).astype(int)

        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        overlay = img.copy()
        print(keypoints.shape)
        print(scores.shape)
        for kpts, score in zip(keypoints, scores):
            show = [0] * len(kpts)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > thr and score[v] > thr:
                    cv2.line(overlay, kpts[u], tuple(kpts[v]), palette[color], self.line_width,
                            cv2.LINE_AA)
                    show[u] = show[v] = 1
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(overlay, kpt, self.circle_radius, palette[color], -1, cv2.LINE_AA)
                    cv2.circle(overlay, kpt, self.circle_radius, (30,30,30), 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, self.skeleton_alpha, img, 1 - self.skeleton_alpha, 0)
    
    def visualize_action(self, img, bbox, labels):
        if len(labels) == 0:
            return img
        
        palette = [(255, 255, 255),
                   self.convert_hex_to_rgb('#249E94'),
                   self.convert_hex_to_rgb('#ACBAC4')]
        font_palette = (0,0,0)
        # font_palette = (255,255,255)
        bbox_len = len(bbox)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_DUPLEX
        # font = cv2.FONT_HERSHEY_TRIPLEX
        # font = cv2.FONT_HERSHEY_TRIPLEX|cv2.FONT_ITALIC
        font_scale = 0.6
        thickness = 1

        padding = 3
        up_to_bbox_padding = 3
        bboxes_gap = 2

        if bbox_len > 4:
            try:
                x_min = int(bbox[0]) 
                y_min = int(bbox[1])
                cur_y = y_min - up_to_bbox_padding

                for i in range(len(labels)):
                    text = f'[{labels[i]["conf"]:.2f}] {labels[i]["label"]}'

                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    box_h = th + baseline + padding * 2
                    box_w = tw + padding * 2

                    x1 = x_min
                    y2 = cur_y
                    y1 = y2 - box_h

                    x2 = x1 + box_w

                    img = self.draw_transparency_rect(img,
                                                    x1, y1,
                                                    x2, y2,
                                                    palette[i], -1, self.alpha)
                    text_x = x1 + padding
                    text_y = y2 - baseline # - padding
                    
                    img = self.draw_transparency_text(img, text, (text_x, text_y),
                                                      font, font_scale, font_palette, thickness, self.text_alpha)
                    
                    cur_y = y1 - bboxes_gap
                    
            except:
                self.parent.log(f'[Thread][AI][Visualization][Error] Fail to draw multi-person bbox')
                self.parent.log('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))

        return img
    
    def draw_transparency_rect(self, img, x_min, y_min, x_max, y_max, c, type, alpha):
        overlay = img.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), c, type)
        return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # def draw_transparency_line(self, img, pt1, pt2, c, w, alpha):
    #     overlay = img.copy()
    #     cv2.line(overlay, pt1, pt2, c, w, cv2.LINE_AA)
    #     return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # def draw_transparency_circle(self, img, x_min, y_min, x_max, y_max, c, type, alpha):
    #     overlay = img.copy()
    #     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), c, type)
    #     return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    def draw_transparency_text(self, img, text, org, font, scale, c, w, alpha):
        overlay = img.copy()

        cv2.putText(img, text, org, font, scale, c, w, lineType=cv2.LINE_AA)
        
        return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    
    def convert_hex_to_rgb(self, hex_code: str) -> tuple:
        hex = hex_code.lstrip('#')
        r = int(hex[0:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        return (r, g, b)
    
    def draw_fps(self, img, fps):
        text = f'FPS:{int(fps)}'
        org = (5, 25)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_TRIPLEX
        scale = 0.8
        c = (0,255,0)
        w = 1

        return cv2.putText(img, text, org, font, scale, c, w, lineType=cv2.LINE_AA)


    def stop(self):
        self._running = False
        self.quit()
        self.wait()
        
if __name__ == "__main__":
    print('Test start')
    test = AiThread()
    test.start()
    sleep(5)
    test.stop()
    print('Test complete')