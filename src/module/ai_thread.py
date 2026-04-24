import cv2
import torch
import numpy as np
import traceback
import time
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

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

class AiThread(QThread):
    # signalSetImage = Signal(QImage)
    signalSetImage = Signal(dict)
    
    def __init__(self, RESOURCE_DIR, parent=None):
        super().__init__()
        self.parent = parent
        self.RESOURCE_DIR = RESOURCE_DIR

        # print()
        print('==============================================inference_topdown===')
        
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
        
        # 카메라(webcam)
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.parent.log(f'[Thread][AI] Camera (webcam) : {self.frame_width}x{self.frame_height} @ {self.frame_rate:.2f} FPS')
        self.frame_loader = self.cap

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

        # taewon
        self.frame_drop = 0
        self.frame_count = 0
        self.current_view_mode = 'webcam'
        self.pose_skip_count = 0
        self.last_result_dict = {}
        self.ai_action_interval = 1.0/10.0
        self.last_action_time = 0.0
        self.pose_append_interval = 1.0/30.0  # 30 FPS
        # self.roi_points = [(1,1),(1,480),(639, 480),(639, 1)]
        self.roi_points = [(1,1),(1,719),(1285, 719),(1285, 1)]


        # -------------------
        # action
        # self.window_size = 32
        # self.window_size = 48
        self.window_size = 31
        # self.window_size = 64
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

        self.pose_append_timer = Duration('[AI][Pose Append]')
        self.loop_timer = Duration('[AI][Loop]')
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
        
        # loop start

        pose_results = dict()
        keypoints_per_id      = {}

        # while self._running:
        while self.cap.isOpened() and self._running:
            try:
                # cam에서 이미지 가져오기
                self.loop_timer.set_prev()
                ret, frame = self.frame_loader.read()

                
                if not ret:
                    print('Failed to capture frame from camera.')
                    if self.frame_drop > 30:
                        self.cap.release()
                        print('Camera released.')
                    time.sleep(1)  # 잠시 대기 후 재시도
                    self.frame_drop += 1
                    print(f'frame drop 발생 : {self.frame_drop}')
                    self.cap = cv2.VideoCapture(0)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    continue
                self.frame_drop = 0
                # frame = self.frame_loader.get_frame()
                # if frame is None:
                #     # print('skip !!!!!!!!')
                #     continue
                # print()
                # print('=================================================')
                rgb_img = frame[:, :, ::-1]

                h, w, _ = rgb_img.shape

                # self.det_timer.set_prev()

                det_result = inference_detector(self.person_detector.detector, frame)
                
                # self.det_timer.calc_elapsed()
                # self.det_duration.append(self.det_timer.get_elapsed())
                # self.det_timer.print_fps()
                # self.det_timer.print_sec()
                # print()

                pred_instance = det_result.pred_instances.cpu().numpy()

                output_bboxes = pred_instance.bboxes # (N, 4)
                output_labels = pred_instance.labels # (N,)
                output_scores = pred_instance.scores # (N,)
                person_idx_lst = np.logical_and(output_labels == self.det_cat_id,
                                                output_scores > self.bbox_thr)
                if not person_idx_lst.any():
                    result_dict = {
                        "frame_raw"            : np.ascontiguousarray(rgb_img),
                        "detections"           : np.empty((0, 7), dtype=np.float32),
                        "human_exists"         : False,
                        "action_results"       : {},
                        "keypoints_scores_pair": None,
                        "keypoints"            : {},
                    }
                    self.signalSetImage.emit(result_dict)
                    print(f'[Debug][AI-thread]no man!! : {result_dict["keypoints_scores_pair"]}')
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
                tracked_bboxes = np.empty((0, 7), dtype=np.float32)  # 기본값
                try:
                    if len(track_bboxes) > 0:
                        # (M, 7) -> [ [x1, y1, x2, y2, idx, cls, conf], ... ]
                        tracked_bboxes = self.tracker_list.update(torch.tensor(track_bboxes), rgb_img)
                except:
                    print('[ERROR][AI] {0}'.format(traceback.format_exc()))

                # -------------------------------------------------------
                multi_pose_results = []
                now_frame_id = []
                # keypoints_per_id      = {}
                
                self.multi_pose_timer.set_prev()

                # # 2frame마다 pose estimation 수행 (조정 필요)
                # self.frame_count += 1
                # if self.frame_count % 2 != 0:
                #     if hasattr(self, 'last_result_dict'):
                #         # self.loop_timer.calc_elapsed()
                #         # self.loop_timer.print_sec()
                #         # self.loop_timer.print_fps()
                        
                #         self.last_result_dict['frame_raw'] = np.ascontiguousarray(rgb_img)

                #         self.signalSetImage.emit(self.last_result_dict)
                #     continue

                # 경과시간으로 조정해보기
                # self.ai_action_interval = 1.0/10.0
                # self.last_action_time = 0.0
                # if self.action_timer.get_elapsed() >= self.ai_action_interval:
                #     continue
                
                roi_arr = np.array(self.roi_points, np.int32)
                for single_bbox in tracked_bboxes:
                    
                    # roi 영역만 pose 돌리기
                    x1, y1, x2, y2 = single_bbox[:4]
                    cx = (x1 + x2) / 2
                    cy = y2  # 발끝 기준
                    # ROI 안에 있는 사람만 Pose 처리
                    if cv2.pointPolygonTest(roi_arr, (cx, cy), False) < 0:
                        continue
                    
                    identification = single_bbox[4]
                    now_frame_id.append(identification)
                    if not identification in pose_results.keys():
                        pose_results[identification] = []

                    
                    # ── Pose 실행 전에 체크 ──────────────────────
                    if self.pose_append_timer.get_elapsed_now() < self.pose_append_interval:
                        # print(f'[Debug][AI] pose append skipped (elapsed : {self.pose_append_timer.get_elapsed_now():.3f} sec)')
                        continue  # Pose inference 자체를 스킵
                    self.pose_append_timer.set_prev()
                    # ─────────────────────────────────────────────

                    # original (coco 17key)
                    # id = np.expand_dims(np.array([identification]*17), 0)
                    # id = np.expand_dims(id, 2)
                    # for whole body
                    idx = np.expand_dims(np.array([identification]*133), 0)
                    idx = np.expand_dims(idx, 2)

                    self.pose_timer.set_prev()
                    

                    single_pose_result = inference_topdown(self.pose_estimator.pose_estimator,
                                                           rgb_img,
                                                           single_bbox[None, :4])
                    
                    self.pose_timer.calc_elapsed()
                    self.pose_duration.append(self.pose_timer.get_elapsed())
                    # self.pose_timer.print_fps()
                    # self.pose_timer.print_sec()
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
                    person_info = np.concatenate((person_info, idx), axis=2)

                    # (17,4)
                    person_info = np.squeeze(person_info)
                    multi_pose_results.append(person_info)
                    
                    # id별 키포인트 저장 (테스트)
                    person_keypoints_full = single_pose_result[0].get('pred_instances').get('keypoints') # shape : (1, 133, 2)
                    keypoints_per_id[int(identification)] = person_keypoints_full[0, :, :2] # (133,2)
                
                # (M, 17, 4)
                # keypoints_info = np.array(multi_pose_results)
                keypoints_info = np.array(multi_pose_results)
                if keypoints_info.ndim != 3:
                    self.last_result_dict['frame_raw'] = np.ascontiguousarray(rgb_img)
                    self.last_result_dict['detections'] = np.empty((0, 7), dtype=np.float32)
                    self.last_result_dict['keypoints_scores_pair'] = (np.empty((0, 17, 2)), np.empty((0, 17)))
                    self.signalSetImage.emit(self.last_result_dict)
                    continue

                keypoints = keypoints_info[:,:,:2]
                keypoints_scores = keypoints_info[:,:,2]

                # self.multi_pose_timer.calc_elapsed()
                # self.multi_pose_duration.append(self.multi_pose_timer.get_elapsed())
                # self.multi_pose_timer.print_fps()
                # -------------------------------------------------------
                # Action
                action_results_per_id = {}   # { int(id): [{class_idx, conf}, ...] }
                # keypoints_per_id      = {}   # { int(id): ndarray (17, 2) }
                # action_scores_np        = None
                # self.action_timer.set_prev()
                action_ran = False
                for key, result_lst in pose_results.items():
                    if key in now_frame_id:
                        action_scores_np        = None # 수정
                        action_label_results = []
                        if len(result_lst) >= self.window_size:
                            action_input = result_lst[-(self.window_size):]

                            # self.action_timer.set_prev()
                            action_result = inference_skeleton(self.action_recognizer.action_recognizer, action_input, (h, w))
                            # self.action_timer.calc_elapsed()
                            # self.action_duration.append(self.action_timer.get_elapsed())

                            action_scores_np = action_result.pred_score.cpu().numpy()  # shape (num_classes,)
                            
                            action_ran = True

                        action_results_per_id[int(key)] = action_scores_np # added      
                # if action_ran:
                #     self.action_timer.calc_elapsed()
                #     self.action_timer.print_fps()
                #     self.action_timer.print_sec()
                            # action_scores = action_result.pred_score
                            # while True:
                            #     max_pred_index = action_scores.argmax().item()
                            #     action_label   = self.label_map[max_pred_index]
                            #     action_conf    = action_scores[max_pred_index].item()
                            #     if action_conf > self.action_thr:
                            #         # action_label_results.append({'label': action_label, 'conf': action_conf})
                            #         action_label_results.append({'class_idx': max_pred_index, 'conf': action_conf})
                            #         if len(action_label_results) == self.K:
                            #             break
                            #         action_scores[max_pred_index] = 0
                            #     else:
                            #         break

                        
                """
                        # 일단 전부 날것으로
                        if action_scores_np is not None:
                            # print(f'[Debug][AI] action_scores_np : {action_scores_np}]')
                            action_results_per_id[int(key)] = action_scores_np
                        else:
                            action_results_per_id[int(key)] = None
                        
                        # # id별 action 결과 저장
                        # action_results_per_id[int(key)] = action_label_results
                
                # keypoints_per_id 구성 (tracked_bboxes 순서로)
                for single_bbox in tracked_bboxes:
                    pid = int(single_bbox[4])
                    # multi_pose_results에서 해당 id 위치 찾기
                    for idx, bbox in enumerate(tracked_bboxes):
                        if int(bbox[4]) == pid and idx < len(multi_pose_results):
                            kpts_xy = multi_pose_results[idx][:, :2]  # (17, 2)
                            keypoints_per_id[pid] = kpts_xy
                            break
                """            
                
                # -------------------------------------------------------
                # emit — 원본 numpy + AI 결과값만 전달 (시각화 X)
                result_dict = {
                    "frame_raw"            : np.ascontiguousarray(rgb_img),
                    "detections"           : tracked_bboxes,          # (M, 7)
                    "action_results"       : action_results_per_id,   # {int: [{label,conf}]}
                    "keypoints_scores_pair": (keypoints, keypoints_scores),  # (M,17,2), (M,17)
                    "keypoints"            : keypoints_per_id,               # {int: (17, 2)}
                }

                # self.loop_timer.calc_elapsed()
                # self.loop_timer.print_fps()
                # self.loop_timer.print_sec()

                self.last_result_dict = result_dict
                self.signalSetImage.emit(result_dict)
                # fps = self.getFps()
                # print()
                # print('-----------------------------------------')
                # print(f'Total sec : {(1/fps):.3f}')
                # print(f'Total fps : {fps:.1f}')
                # print()
                # det_sec = sum(self.det_duration) / len(self.det_duration)
                # pose_sec = sum(self.pose_duration) / len(self.pose_duration)
                # multi_pose_sec = sum(self.multi_pose_duration) / len(self.multi_pose_duration)
                # action_sec = sum(self.action_duration) / len(self.action_duration)

                # print(f'Det sec : {(det_sec):.3f}')
                # print(f'Det fps : {(1/det_sec):.1f}')
                # print()
                # print(f'Pose sec : {(pose_sec):.3f}')
                # print(f'Pose fps : {(1/pose_sec):.1f}')
                # print()
                # print(f'Multi Pose sec : {(multi_pose_sec):.3f}')
                # print(f'Multi Pose fps : {(1/multi_pose_sec):.1f}')
                # print()
                # # print(f'Action sec : {(action_sec):.3f}')
                # # print(f'Action fps : {(1/action_sec):.1f}')
                # print()
                

            except:
                print('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))
                continue


    # def getFps(self):
    #     elapsed = perf_counter() - self.prevTime
    #     if self.frame_loader.mode == 'webcam':
    #         fps = 1/elapsed
    #     else:
    #         delay = max(0, self.frame_loader.delay_limit - elapsed)
    #         sleep(delay)
    #         print(f'delay : {delay:.5f}')
    #         if delay == 0:
    #             fps = 1/elapsed
    #         else:
    #             fps = self.frame_loader.video_fps
    #     self.prevTime = perf_counter()

    #     return fps


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
    
    @Slot(dict)
    def update_status(self, status):
        self.current_view_mode = status['mode']
        # print(f'[Debug][AI Thread] mode : {self.current_view_mode}')
        
if __name__ == "__main__":
    print('Test start')
    test = AiThread()
    test.start()
    sleep(5)
    test.stop()
    print('Test complete')