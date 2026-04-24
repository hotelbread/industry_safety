import cv2
import time
import traceback
import numpy as np

from PIL import ImageFont, ImageDraw, Image
from src.misc.duration import Duration

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPolygon, QImage

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.current_image = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
        # self.roi_points = [(1,1),(1,480),(639, 480),(639, 1)]
        self.roi_points = [(1,1),(1,719),(1279, 719),(1279, 1)]
        self.roi_polygon = None
        self.admin_mode = False
        self.canvas_timer = Duration('canvas')

        # ai 시각화
        self.bbox_alpha     = 0.1
        self.line_alpha     = 0.5
        self.skeleton_alpha = 0.6
        self.text_alpha     = 0.3
        self.alpha          = 0.7
        self.line_width     = 3
        self.circle_radius  = 5
        
        # action recognition
        self.ACTION_LABEL_MAP = {}
        # label_map_path = 'model/action/label_map_hsd_S001.txt'
        label_map_path = 'model/action/label_map_hsd_S007.txt'
        # label_map_path = 'model/action/label_map_industry_safety3.txt'
        try:
            labels = [x.strip() for x in open(label_map_path).readlines()]
            self.ACTION_LABEL_MAP = {i: label for i, label in enumerate(labels)}
            print(f"[Canvas] Action Label Map loaded: {self.ACTION_LABEL_MAP}")
        except:
            self.ACTION_LABEL_MAP = {
                0 : 'Etc',
                1 : "붐 올리기(Raise Boom)",
                2 : "권상(Raise Load)",
                3 : "비상 정지(Emergency Stop)"
            }
        self.STAGE_CLASS_IDXS = [2,1,0]
        
        # fps
        self._prev_time = time.time()
        
        # 모드별 시각화 플래그
        self.show_bbox = True
        self.show_pose = True
        self.show_action = False
        
        try:
        # 시스템 한글 폰트 경로 (Linux 기준)
            self.font_kr = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
        except:
            self.font_kr = None
            print("[Canvas] 한글 폰트 로드 실패 — 영문 폴백")
            
        self.img_acc_panel = cv2.imread('res/UI_File/accuracy_bg.png', cv2.IMREAD_UNCHANGED)
        
        # self.scale_x = 1
        # self.scale_y = 1
                        
        # # scale & offset
        self.scale = 1
        self.offset_x = 0
        self.offset_y = 0
        
        # self.painter = QPainter(self)
    
    def map_to_canvas(self, x, y):
        
        nx = int((x - self.offset_x) * self.scale)
        ny = int((y - self.offset_y) * self.scale)
        return nx, ny    
    
    def overlay_png(self, background, overlay, x, y):
        """NumPy를 이용한 고속 투명 PNG 합성"""
        h, w = overlay.shape[:2]
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return background

        # 알파 채널 분리
        overlay_img = overlay[:, :, :3]
        overlay_img = overlay_img[:,:,::-1]
        mask = overlay[:, :, 3:] / 255.0

        # 합성
        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
        return background
    
    # def update_frame(self, q_image, countdown=None):
    def update_frame(self, result_dict, countdown=None):
        
        rgb_img = result_dict.get("frame_raw")
        detections = result_dict.get("detections", np.empty((0, 7)))
        action_results = result_dict.get("action_results", [])
        keypoints = result_dict.get("keypoints_scores_pair", np.empty((0, 133, 2)))
        current_view_mode = result_dict.get("current_view_mode", "")

        # action 시각화를 위해 테스트중
        action_num = 0
        # if len(current_view_mode) == 7:
        #     action_num = current_view_mode[-1]
        action_num = self.parent.action_flag
        if rgb_img is None:
            print('[Error][Canvas] rgb_img is None')
            self.update()

            return
        img = rgb_img.copy()
        
        # frame size
        h, w = img.shape[:2]
        cw, ch = self.width(), self.height()
        # print(f'[Debug][Canvas] img shape : {w, h}]')
        # print(f'[Debug][Canvas] canvas size : {cw, ch}]')
        # 캔버스 비율과 프레임 비율 계산
        canvas_ratio = cw / ch
        frame_ratio = w / h

        if frame_ratio > canvas_ratio:
            # 가로가 더 김 -> 가로를 자름
            target_w = int(h * canvas_ratio)
            self.offset_x = (w - target_w) // 2
            self.offset_y = 0
            cropped = img[:, self.offset_x : self.offset_x + target_w]
        else:
            # 세로가 더 김 -> 세로를 자름
            target_h = int(w / canvas_ratio)
            self.offset_x = 0
            self.offset_y = (h - target_h) // 2
            cropped = img[self.offset_y : self.offset_y + target_h, :]

        # 캔버스 크기로 리사이즈 및 현재 배율 저장
        img = cv2.resize(cropped, (cw, ch), interpolation=cv2.INTER_LINEAR)
        self.scale = cw / cropped.shape[1] # 크롭된 너비 대비 캔버스 너비 비율
        
        
        # 시각화 레이어 순서대로
        if self.show_bbox and detections.shape[0] > 0:
            img = self.visualize_bbox(img, detections)
            
        if self.show_pose and keypoints is not None:
            kpts, scores = keypoints
            # print(f'[Debug][Canvas] keypoints : {len(kpts)}')
            if len(kpts) > 0:
                img = self.visualize_pose(img, kpts, scores)
                img = self.visualize_posture_line(img, kpts, scores)
        
        # 기존 action 시각화            
        # if self.show_action and action_results:
        #     for human_id, labels in action_results.items():
        #         filtered = [d for d in detections if len(d) > 4 and int(d[4]) == human_id]
        #         if filtered:
        #             img = self.visualize_action(img, filtered[0], labels)
        
        # 태원 실험
        if self.show_action and action_results:
            roi_user_id = self._get_roi_user_action(
                action_results, 
                detections, 
                self.roi_points
                )
            # if roi_user_id is not None:
            img = self.visualize_action_overlay(img, action_results, roi_user_id, action_num)
            # img = self.visualize_action_overlay(img, action_results, active_user_id)
        
        
        
        """
        # numpy => QImage 변환
        self.current_image = self._to_qimage(img)
        
        # ROI + countdown 
        self.current_image = self.paintRoi(self.current_image, self.roi_points)
        if countdown is not None:
            self.countdown_draw(self.current_image, countdown)
        
        # 스케일 계산
        self.q_image_frame_width = self.current_image.width()
        self.q_image_frame_height = self.current_image.height()
        
        # frame size
        self.frame_width = self.width()
        self.frame_height = self.height()
        self.scale = min(self.frame_width/self.q_image_frame_width, self.frame_height/self.q_image_frame_height)
        self.offset_x = (self.frame_width - (self.q_image_frame_width * self.scale))/2
        self.offset_y = (self.frame_height - (self.q_image_frame_height * self.scale))/2
        """

        
        img = self.draw_fps(img)
        
        self.current_image = self._to_qimage(img)
        # ROI는 이미 보정된 캔버스 좌표계에서 그립니다.
        self.current_image = self.paintRoi(self.current_image, self.roi_points)
        
        self.update()
        """
        # print(f'[Debug][Canvas] frame_width : {self.frame_width} x {self.frame_height} [scale: {self.scale}]')
        # print(f'[Debug][Canvas] current_image size: {self.current_image.width()} x {self.current_image.height()}')
        # print(f'[Debug][Canvas] offset : {self.offset_x} , {self.offset_y}')
        # if self.roi is not None and self.current_image is not None:
        #     self.current_image = self.paintRoi(self.current_image, self.roi)
        self.current_image = self.paintRoi(self.current_image, self.roi_points)
        if countdown is not None:
            self.countdown_draw(self.current_image, countdown)
        
        # print(f'[Debug][Canvas] Frame updated with ROI: {self.roi}')
        """
    def _get_roi_user_action(self, action_results, detections, roi_points):
        if len(detections) == 0 or len(roi_points) <3:
            return None
        
        roi_arr = np.array(roi_points, np.int32)
        best_id = None
        best_area = 0
        
        for det in detections:
            if len(det) < 5:
                continue
            x1, y1, x2, y2 = det[:4]
            track_id = int(det[4])
            
            cx = (x1 + x2 ) /2
            cy = y2
            inside = cv2.pointPolygonTest(roi_arr, (cx, cy), False) >= 0
            if inside:
                area = (x2-x1) * (y2-y1)
                if area > best_area:
                    best_area = area
                    best_id = track_id
                    
        return best_id
        # if best_id is not None:
        #     return best_id
        
        # return action_results.get(best_id)
        
    def set_visualization_mode(self, show_bbox=True, show_pose=True, show_action=False):
        # main.py에서 모드 전환시
        self.show_bbox = show_bbox
        self.show_pose = show_pose
        self.show_action = show_action
        
        
    def _to_qimage(self, rgb_img):
        if rgb_img is None:
            print('[Error][Canvas] img is None')
            return None

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(rgb_img)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image
    
    def mousePressEvent(self, event):
        if self.admin_mode:
            
            mx = event.pos().x()
            my = event.pos().y()
            
            # # 영상 영역 안에서만 클릭 허용
            # if (
            #     mx < self.offset_x
            #     or mx > self.offset_x + self.frame_width * self.scale
            #     or my < self.offset_y
            #     or my > self.offset_y + self.frame_height * self.scale
            # ):
            #     return
            

            # # canvas -> frame 변환
            # frame_x = (mx - self.offset_x) / self.scale
            # frame_y = (my - self.offset_y) / self.scale
            
            # 캔버스 좌표 -> 원본 프레임 좌표 역변환
            frame_x = int(mx / self.scale + self.offset_x)
            frame_y = int(my / self.scale + self.offset_y)
            
            self.roi_points.append((frame_x, frame_y))
            
            print(f"Added ROI point: ({frame_x}, {frame_y})")
            self.parent.ai_thread.roi_points = self.roi_points
            print(f'[Debug][Canvas] Sent ROI points to AI thread: {self.roi_points}')
            self.update()
        else:
            print(f"Current points: {self.roi_points}")
    
    def paintRoi(self, image, roi_points):
        """
        painter = QPainter(image)
        painter.setPen(QColor(255, 195, 0))
        if len(roi_points) > 0:
            if len(roi_points) >= 3:
                self.roi_polygon = QPolygon([QPoint(int(x), int(y)) for x, y in roi_points])

            for x, y in self.roi_points:
                painter.drawEllipse(x-4, y-4, 8, 8)
            
            if self.roi_polygon:
                painter.drawPolygon(self.roi_polygon)
        else:
            self.roi_polygon = None
        painter.end()
            
        return image
        """
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing) 
        painter.setPen(QColor(255, 195, 0))
        
        # 원본 좌표 리스트를 캔버스용 좌표 리스트로 변환
        display_points = []
        for x, y in roi_points:
            nx, ny = self.map_to_canvas(x, y)
            display_points.append(QPoint(nx, ny))

        if len(display_points) >= 3:
            self.roi_polygon = QPolygon(display_points)
            painter.drawPolygon(self.roi_polygon)

        for pt in display_points:
            # painter.drawEllipse(pt.x() - 4, pt.y() - 4, 8, 8)
            painter.drawEllipse(pt, 5, 5)
        
        painter.end()
        return image
        
    def countdown_draw(self, q_image, countdown):
        painter = QPainter(q_image)

        color_map = {'green' : QColor(0,255,0), 'red' : QColor(255,0,0),
                     'blue' : QColor(0, 0, 255), 'black': QColor(0, 0, 0)}
        painter.setPen(color_map.get(countdown['color'], QColor(0, 0, 0)))
        
        text_dict = {"green":"Start!!!", "blue":"Done!!!", "red" : "Stop!!!", "black":'test!!!'}

        font = painter.font()
        font.setPointSize(90)
        painter.setFont(font)
        text = text_dict[countdown['color']] if countdown['time'] < 0 else str(countdown['time'])
        text_rect = painter.boundingRect(0, 0, 0, 0, Qt.AlignCenter, text)

        # 화면 중앙 계산
        center_x = q_image.width() // 2
        center_y = q_image.height() // 2
        
        # 텍스트 위치 조정 (폰트 크기 고려)
        text_x = center_x - text_rect.width() // 2
        text_y = center_y - text_rect.height() // 2

        painter.drawText(text_x, text_y, text)

        painter.end()
    
    def paintEvent(self, event):
        # self.canvas_timer.set_prev()

        painter = QPainter(self)

        # painter.fillRect(self.rect(), Qt.black)
        painter.fillRect(self.rect(), QColor("#2C303C"))

        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            widget_width = self.width()
            widget_height = self.height()
            
            img_ratio = img_width / img_height
            widget_ratio = widget_width / widget_height
            
            if img_ratio > widget_ratio:
                new_width = widget_width
                new_height = int(widget_width / img_ratio)
            else:
                new_height = widget_height
                new_width = int(widget_height * img_ratio)
            
            x = (widget_width - new_width) // 2
            y = (widget_height - new_height) // 2

            target_rect = QRect(x, y, new_width, new_height)
            painter.drawImage(target_rect, self.current_image)
        
        painter.end()

        # self.canvas_timer.calc_elapsed()
        # self.canvas_timer.print_fps()
        # self.canvas_timer.print_sec()
        # print()

    # def clearPaint(self):
    #     self.painter.fillRect(self.rect(), QColor("#2C303C"))
    
    def draw_fps(self, img):
        # org = (5, 25)
        org = (20, 40)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_TRIPLEX
        scale = 1
        c = (0,255,0)
        w = 1
        now = time.time()
        elapsed = now - self._prev_time
        self._prev_time = now
        fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
        text = f'FPS:{int(fps)}'
        
        return cv2.putText(img.copy(), text, org, font, scale, c, w, lineType=cv2.LINE_AA)

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
            # x_min = int(bboxes[bidx][0])
            # y_min = int(bboxes[bidx][1])
            # x_max = int(bboxes[bidx][2])
            # y_max = int(bboxes[bidx][3])
            # x_min = int((bboxes[bidx][0] - self.offset_x) * self.scale)
            # y_min = int((bboxes[bidx][1] - self.offset_y) * self.scale)
            # x_max = int((bboxes[bidx][2] - self.offset_x) * self.scale)
            # y_max = int((bboxes[bidx][3] - self.offset_y) * self.scale)
            x_min, y_min = self.map_to_canvas(bboxes[bidx][0], bboxes[bidx][1])
            x_max, y_max = self.map_to_canvas(bboxes[bidx][2], bboxes[bidx][3])

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
                    print(f'[Thread][AI][Visualization][Error] Fail to draw multi-person bbox')
                    print('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))

        return img
    
    
    def visualize_pose(self, img, kpts, scores, thr=0.5):
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
        

        # scale = 1
        # keypoints = (keypoints * scale).astype(int)
        keypoints = kpts.copy()
        keypoints[:, :, 0] = (keypoints[:, :, 0] - self.offset_x) * self.scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - self.offset_y) * self.scale
        keypoints = keypoints.astype(int)

        # img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        overlay = img.copy()
        # print(keypoints.shape)
        # print(scores.shape)
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
    
    def visualize_posture_line(self, img, keypoints, scores, thr=0.5):
        """
        몸통 기준선 + 수직 기준선 + 기울기 각도 시각화
        - 어깨 중심(5,6) <-> 엉덩이 중심(11,12) 직선
        - 어깨 중심에서 수직 하향 기준선
        - 두 선 사이의 각도(°) 텍스트 표시
        """
        if keypoints is None or scores is None:
            return img

        overlay = img.copy()

        for kpts, score in zip(keypoints, scores):
            # 신뢰도 체크: 4개 키포인트 모두 thr 이상일 때만
            idxs = [5, 6, 11, 12]
            if not all(score[i] > thr for i in idxs):
                continue
            
            # s_mid_raw = (kpts[5] + kpts[6]) / 2
            # h_mid_raw = (kpts[11] + kpts[12]) / 2
            # shoulder_mid = self.map_to_canvas(s_mid_raw[0], s_mid_raw[1])
            # hip_mid = self.map_to_canvas(h_mid_raw[0], h_mid_raw[1])

            # # 중심점 계산
            shoulder_mid = ((kpts[5] + kpts[6]) / 2).astype(int)
            hip_mid      = ((kpts[11] + kpts[12]) / 2).astype(int)
            shoulder_mid = self.map_to_canvas(shoulder_mid[0], shoulder_mid[1])
            hip_mid = self.map_to_canvas(hip_mid[0], hip_mid[1])


            # 각도 계산
            shoulder_mid_arr = np.array(shoulder_mid)
            hip_mid_arr = np.array(hip_mid)
            trunk_vec = hip_mid_arr - shoulder_mid_arr  # 어깨 -> 엉덩이
            norm = np.linalg.norm(trunk_vec)
            if norm < 1e-6:
                continue

            trunk_vec_norm = trunk_vec / norm
            # (0,1)과의 내적 = trunk_vec_norm[1] (y성분만 남음)
            cos_angle = np.clip(trunk_vec_norm[1], -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            # 수직 기준선 끝점: 어깨 중심에서 hip_mid까지의 거리만큼 아래로
            # line_len = int(norm)
            # vertical_end = (shoulder_mid[0], shoulder_mid[1] + line_len)
            # 오른발 기준으로 line_len 고쳐봄
            # vertical_end = (shoulder_mid[0], kpts[16][1].astype(int))  # 오른발(16) y좌표 기준으로 수직선 길이 조정
            _, foot_y_fixed = self.map_to_canvas(0, kpts[16][1])
            vertical_end = (shoulder_mid[0], int(foot_y_fixed))

            # ── 몸통 기준선 (초록) ──────────────────────────────
            cv2.line(overlay,
                    tuple(shoulder_mid), tuple(hip_mid),
                    (0, 255, 120), 2, cv2.LINE_AA)

            # ── 수직 기준선 (하늘색, 점선 느낌으로 두께 얇게) ──
            cv2.line(overlay,
                    tuple(shoulder_mid), vertical_end,
                    (100, 200, 255), 2, cv2.LINE_AA)

            # ── 중심점 마커 ─────────────────────────────────────
            cv2.circle(overlay, tuple(shoulder_mid), 5, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(hip_mid),      5, (0, 255, 120), -1, cv2.LINE_AA)

            # ── 각도 호(arc) - 사잇각 강조 ──────────────────────
            arc_radius = 35
            # 호의 시작/끝 각도: OpenCV는 3시 방향이 0°, 시계방향
            # 수직선은 90°, 몸통 벡터 방향을 atan2로 계산
            trunk_angle_cv = np.degrees(np.arctan2(trunk_vec[1], trunk_vec[0]))
            cv2.ellipse(overlay,
                        tuple(shoulder_mid),
                        (arc_radius, arc_radius),
                        0,
                        90,                      # 수직선 방향
                        trunk_angle_cv,          # 몸통 벡터 방향
                        (255, 220, 50), 2, cv2.LINE_AA)

            # ── 각도 텍스트 ──────────────────────────────────────
            text = f'{angle_deg:.1f} deg'
            text_pos = (shoulder_mid[0] + arc_radius + 5, shoulder_mid[1] + 20)
            cv2.putText(overlay, text, text_pos,
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 220, 50), 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    
    def visualize_action_overlay(self, img, action_results, roi_user_id, action_num):
        """
        우측 상단에 동작명 + conf 바 형태로 표시.
        표시할 클래스만 골라서 우측 상단에
        roi_user_id  : _get_roi_user_action()이 반환한 track_id (int 또는 None)
        action_num   : 현재 평가 중인 동작 번호 (1, 2, 3) — 이 클래스의 conf를 표시
        action_results: { int(id): np.ndarray(num_classes,) }
        action tesx mapping 필요???
        self.STAGE_CLASS_IDXS = [2,1,0]
        """
        # TARGET_CLASSES = [1,2,3]
        
        # if not action_results or active_user_id not in action_results:
        #     return img

        if roi_user_id:
            labels = action_results[roi_user_id]
            # print(f'[Debug][Canvas] action_results : {action_results}]')
            # print(f'[Debug][Canvas] labels : {labels}')
            scores = action_results[roi_user_id]
            if scores is None:
                scores = [0] * len(self.ACTION_LABEL_MAP)
        else:
            labels = None
            scores = [0] * len(self.ACTION_LABEL_MAP)
        
        
        # try:
        #     class_idx = int(action_num)
        try:
            class_idx = int(self.STAGE_CLASS_IDXS[ int(action_num) - 1])
        except (ValueError, TypeError):
            return img
        
        # if class_idx <= 0 or class_idx >= len(scores):
        #     return img
        ### debugging
        try:
            conf    = float(scores[class_idx])
        except:
            # print(f'scores : {scores}')
            # print(f'class_idx : {class_idx}')
            conf = 0.0            
            # print(f'conf : {conf}')
            
        label_text = self.ACTION_LABEL_MAP.get(class_idx, f'Class {class_idx}') # conf가 0.0 일경우 etc로 표시하는걸로 변경할지 고민
        conf_text = f'{conf:.2f}'
        
        if conf >= 0.8:
            text_color = (160, 230, 0)   # 민트
            bar_color  = (140, 210, 0)
        elif conf >= 0.5:
            text_color = (255, 165, 0) # 주황
            bar_color = (230, 150, 0)
        else:
            text_color = (220, 60, 60) # 
            bar_color  = (200, 50, 50)

        # if not labels:
        #     return img
        
        
        # 시안 적용 버전
        h, w = img.shape[:2]
        pw, ph = self.img_acc_panel.shape[1], self.img_acc_panel.shape[0]
        # x0, y0 = w - self.img_acc_panel.shape[1] - 20, 20
        x0, y0 = w - pw - 25, 25
        
        img = self.overlay_png(img, self.img_acc_panel, x0, y0)

        # 4. 점수 텍스트 표출 (배경 위에 얹기)
        # 위치(x0 + N, y0 + N)는 이미지 내 'Accuracy' 글자 옆 빈 공간에 맞춰보세요.
        conf_percent = f"{int(conf * 100)}"
        
        # 숫자 출력 (큰 폰트)
        cv2.putText(img, conf_percent, (x0 + 120, y0 + 65), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, text_color, 2, cv2.LINE_AA)
        
        # % 표시 (작은 폰트)
        cv2.putText(img, "%", (x0 + 195, y0 + 65), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        
        """
        img = img.copy()
        h, w = img.shape[:2]

        # ── 레이아웃 상수 ──────────────────────────────────────────
        box_w      = 260    # 전체 패널 너비
        row_h      = 30     # 한 줄 높이
        padding    = 10     # 내부 여백
        bar_max_w  = 160    # conf 바 최대 너비
        margin_r   = 15     # 화면 우측 여백
        margin_t   = 15     # 화면 상단 여백

        # n = len(labels)
        panel_h = row_h + padding * 2

        x0 = w - box_w - margin_r
        y0 = margin_t

        # ── 패널 배경 ──────────────────────────────────────────────
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + panel_h), (20, 20, 30), -1)
        img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)

        # ── 각 동작 행 ────────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness  = 1

        # class_idx = item.get("class_idx", -1)
        # class_idx = action_num
        # 라벨 매핑
        

        # row_y = y0 + padding + row_h
        row_y = y0 + padding + 20


        # 동작명
        # cv2.putText(img, label_text,
        #             (x0 + padding, row_y - 10),
        #             font, font_scale, text_color, thickness, cv2.LINE_AA)
        img = self.draw_text_kr(img, label_text,
                        (x0 + padding, row_y - 20),
                        color=text_color, font_size = 12)
        # print(f'[Debug][Canvas] textColor : {text_color}')
        
        # conf 숫자
        # if labels is not None:
            # conf      = labels.get("conf", 0.0)
            # conf      = labels[action_num]
            # conf_text  = f"{conf:.2f}"
        cv2.putText(img, conf_text,
                    (x0 + box_w - 50, row_y - 10),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

        # conf 바 배경
        bar_x = x0 + padding
        bar_y = row_y 
        cv2.rectangle(img,
                    (bar_x, bar_y),
                    (bar_x + bar_max_w, bar_y + 6),
                    (50, 50, 60), -1)

        # conf 바
        filled = int(bar_max_w * min(conf, 1.0))
        if filled > 0:
            cv2.rectangle(img,
                        (bar_x, bar_y),
                        (bar_x + filled, bar_y + 6),
                        bar_color, -1)

        # ── 패널 테두리 및 배경색───────────────────────────────────────────
        overlay_2 = img.copy()
        cv2.rectangle(overlay_2, (x0, y0), (x0 + box_w, y0 + panel_h), (0, 180, 120), 1)
        alpha = 0.3
        cv2.addWeighted(overlay_2, alpha, img, 1 - alpha, 0, img)
        """
        return img
    
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
                print(f'[Thread][AI][Visualization][Error] Fail to draw multi-person bbox')
                print('[ERROR][AI] Fail : {0}'.format(traceback.format_exc()))

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
    
    def draw_text_kr(self, img, text, pos, color=(0, 230, 160), font_size=20):
        """
        한글 텍스트를 img에 그려서 반환.
        img: numpy RGB, pos: (x, y), color: RGB tuple
        """
        if self.font_kr is None:
            # 폰트 없으면 cv2 영문 폴백
            return cv2.putText(img.copy(), text, pos,
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

        img_pil = Image.fromarray(img)
        draw    = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size
            )
        except:
            font = self.font_kr
        draw.text(pos, text, font=font, fill=color)
        return np.array(img_pil)