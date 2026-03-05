import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QApplication
from PySide6.QtCore import Qt, QMutex, Slot

from src.ai_module.pose_estimater import PoseEstimator
from src.ai_module.person_detector import PersonDetector
from src.ai_module.action_recognizer import ActionRecognizer

from src.gui_module.canvas import Canvas
from src.module.video_thread import VideoThread
# from src.module.inf_thread import InferenceThread
import src.misc.tools as tools

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

class MW(QMainWindow):
    def __init__(self, app):
        """ Constructor for Empty Window Class """
        super().__init__()

        self.mutex = QMutex()

        print()
        self.log('-------------------')
        self.log('-------------------')
        self.log('-----GUI START-----')
        self.log('-------------------')
        self.log('-------------------')
        print()

        # -------------
        # Monitor info
        screen = app.primaryScreen()
        size = screen.size()
        rect = screen.availableGeometry()

        print()
        self.log('[GUI] Screen: %s' % screen.name())
        self.log('[GUI] Size: %d x %d' % (size.width(), size.height()))
        self.log('[GUI] Available: %d x %d' % (rect.width(), rect.height()))
        print()

        self.WINDOW_WIDTH = size.width()
        self.WINDOW_HEIGHT = size.height()

        self.AVAIL_WIDGTH = rect.width()
        self.AVAIL_HEIGHT = rect.height()

        # -------------
        # Const
        self.COLOR_DEEP_GRAY = '#383838'
        self.COLOR_GRAY = '#303030'
        self.COLOR_LIGHT_GRAY = '#3D3D3D'
        self.COLOR_RED = '#E6544E'
        self.COLOR_DEEP_RED = '#870601'
        self.COLOR_GREEN = '#17A079'
        self.COLOR_DEEP_GREEN = '#024D3B'
        self.COLOR_YELLOW = '#FEEEA7'
        self.COLOR_DEEP_YELLOW = '#FBD153'

        # -------------
        # Resource relative path
        self.RESOURCE_DIR = get_resource_dir()

        self.PATH_LOGO = os.path.join(self.RESOURCE_DIR, 'res/UI_File/logo_white.png')
        self.PATH_BTN_EXIT = os.path.join(self.RESOURCE_DIR, './res/UI_File/btnClose.png')

        # -------------
        # Size
        self.SPACING = 3
        # -----------------------------------------------------------------
        self.WIDGET_SUB_WIDTH = (int) (self.WINDOW_WIDTH*(20/100))
        self.WIDGET_SUB_HEIGHT = self.WINDOW_HEIGHT

        self.WIDGET_SUB_TOP_WIDTH = self.WIDGET_SUB_WIDTH - (self.SPACING*2)
        self.WIDGET_SUB_TOP_HEIGHT = (int) (self.WIDGET_SUB_HEIGHT*(7/100))

        # self.LABEL_LOGO_WIDTH = (int) (self.WIDGET_SUB_WIDTH*(70/100))
        self.LABEL_LOGO_HEIGHT = self.WIDGET_SUB_TOP_HEIGHT

        # self.BTN_EXIT_WIDTH = (int) (self.WIDGET_SUB_WIDTH*(30/100))
        self.BTN_EXIT_HEIGHT = self.WIDGET_SUB_TOP_HEIGHT

        self.BTN_LOAD_WIDTH = (int) (self.WIDGET_SUB_WIDTH*(80/100))
        self.BTN_LOAD_HEIGHT  = self.WIDGET_SUB_TOP_HEIGHT
        # -----------------------------------------------------------------
        self.WIDGET_CENTRAL_WIDTH = (int) (self.WINDOW_WIDTH*(80/100))
        self.WIDGET_CENTRAL_HEIGHT = self.WINDOW_HEIGHT

        self.CANVAS_CONTAINER_WIDGET = self.WIDGET_CENTRAL_WIDTH - (self.SPACING*2)
        self.CANVAS_CONTAINER_HEIGHT = self.WIDGET_CENTRAL_HEIGHT*(80/100) - (self.SPACING)

        self.CANVAS_WIDTH = self.CANVAS_CONTAINER_WIDGET
        self.CANVAS_HEIGHT = self.CANVAS_CONTAINER_HEIGHT

        self.CONTROLLER_CONTAINER_WIDGET = self.WIDGET_CENTRAL_WIDTH  - (self.SPACING*2)
        self.CONTROLLER_CONTAINER_HEIGHT = self.WIDGET_CENTRAL_HEIGHT*(20/100) - (self.SPACING)
        # -----------------------------------------------------------------

        self.init_res()
        # self.init_ai_model()
        self.init_ui()
        self.setup_thread()
        # -----------------------------------------------------------------
        # 태원 추가
        self.current_view_mode = 'video'  # 'video' or 'webcam'
        self.roi = np.array([[100, 100], [500, 100], [500, 500], [100, 500]])  # 예시 ROI 좌표 (x1, y1, x2, y2, x3, y3, x4, y4)
        self.id_enter_time = {} 
        self.id_last_seen = {}
        self.active_user_id = None
        self.roi_threshold_sec = 3 # 체류 기준
        self.new_user_threshold_sec = 2 # 사라짐 기준
        self.countdown_start = None
        self.countdown_duration = 3  # seconds

    def init_res(self):
        self.pixmap_logo = tools.get_resized_pixmap_based_h(self.PATH_LOGO, self.LABEL_LOGO_HEIGHT)
        self.pixmap_btn_exit = tools.get_resized_pixmap_based_h(self.PATH_BTN_EXIT, self.BTN_EXIT_HEIGHT)

    def init_ai_model(self):
        """set up the AI model"""
        # self.device_name =  cpu'        # for cpu
        self.device_name = 'cuda:0'     # for GPU Desktop
        self.person_detector = PersonDetector(self.device_name, self.RESOURCE_DIR)
        self.pose_estimator = PoseEstimator(self.device_name, self.RESOURCE_DIR)
        self.action_recognizer = ActionRecognizer(self.device_name, self.RESOURCE_DIR)

    def init_ui(self):
        ########################################
        #               init GUI               #
        ########################################
        self.canvas = Canvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        self.canvas.setFixedSize(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)

        # Canvas container widget v layout
        self.v_layout_canvas_container = QVBoxLayout()
        self.v_layout_canvas_container.setContentsMargins(0, 0, 0, 0)
        self.v_layout_canvas_container.addStretch()
        self.v_layout_canvas_container.addWidget(self.canvas)
        self.v_layout_canvas_container.addStretch()

        # Canvas container widget
        self.widget_canvas_container = QWidget()
        self.widget_canvas_container.setContentsMargins(0, 0, 0, 0)
        self.widget_canvas_container.setFixedSize(self.CANVAS_CONTAINER_WIDGET, self.CANVAS_CONTAINER_HEIGHT)
        self.widget_canvas_container.setStyleSheet("border: 2px solid #ffffff")
        self.widget_canvas_container.setLayout(self.v_layout_canvas_container)
        
        # Controller container widget v layout
        self.v_layout_controller_container = QVBoxLayout()

        # Controller container widget
        self.widget_controller_container = QWidget()
        self.widget_controller_container.setContentsMargins(0, 0, 0, 0)
        self.widget_controller_container.setFixedSize(self.CONTROLLER_CONTAINER_WIDGET, self.CONTROLLER_CONTAINER_HEIGHT)
        self.widget_controller_container.setLayout(self.v_layout_controller_container)
        self.widget_controller_container.setStyleSheet("border: 2px solid #ffffff")

        # Label logo
        self.label_logo = QLabel()
        self.label_logo.setAlignment(Qt.AlignCenter)
        self.label_logo.setPixmap(self.pixmap_logo)
        
        # Label btn exit
        self.label_btn_exit = QLabel()
        self.label_btn_exit.setAlignment(Qt.AlignCenter)
        self.label_btn_exit.setPixmap(self.pixmap_btn_exit)
        self.label_btn_exit.mouseReleaseEvent = self.label_btn_exit_mouseReleaseEvent

        # Sub top widget h layout
        self.h_layout_sub_top = QHBoxLayout()
        self.h_layout_sub_top.setContentsMargins(0, 0, 0, 0)
        self.h_layout_sub_top.addStretch()
        self.h_layout_sub_top.addWidget(self.label_logo)
        self.h_layout_sub_top.addStretch()
        self.h_layout_sub_top.addWidget(self.label_btn_exit)

        # Sub top widget
        self.widget_sub_top = QWidget()
        self.widget_sub_top.setContentsMargins(0, 0, 0, 0)
        self.widget_sub_top.setFixedSize(self.WIDGET_SUB_TOP_WIDTH, self.WIDGET_SUB_TOP_HEIGHT)
        self.widget_sub_top.setLayout(self.h_layout_sub_top)
        self.widget_sub_top.setStyleSheet("border: 0px")


        # Central widget v layout
        self.v_layout_central = QVBoxLayout()
        self.v_layout_central.setContentsMargins(0, 0, 0, 0)
        self.v_layout_central.setAlignment(Qt.AlignCenter)
        self.v_layout_central.addWidget(self.widget_canvas_container)
        self.v_layout_central.addWidget(self.widget_controller_container)

        # Sub widget v layout
        self.v_layout_sub = QVBoxLayout()
        self.v_layout_sub.setContentsMargins(0, 0, 0, 0)
        self.v_layout_sub.setAlignment(Qt.AlignCenter)
        self.v_layout_sub.addWidget(self.widget_sub_top)
        self.v_layout_sub.addStretch()       
        
        # Central widget
        self.widget_central = QWidget()
        self.widget_central.setContentsMargins(self.SPACING, self.SPACING, self.SPACING, self.SPACING)
        self.widget_central.setFixedSize(self.WIDGET_CENTRAL_WIDTH, self.WIDGET_CENTRAL_HEIGHT)
        self.widget_central.setLayout(self.v_layout_central)
        self.widget_central.setStyleSheet("border: 2px solid #ffffff")

        # Sub widget
        self.widget_sub = QWidget()
        self.widget_sub.setContentsMargins(self.SPACING, self.SPACING, self.SPACING, self.SPACING)
        self.widget_sub.setFixedSize(self.WIDGET_SUB_WIDTH, self.WIDGET_SUB_HEIGHT)
        self.widget_sub.setLayout(self.v_layout_sub)
        self.widget_sub.setStyleSheet("border: 2px solid #ffffff")

        # Main horizontal layout
        self.h_layout_main = QHBoxLayout()
        self.h_layout_main.setContentsMargins(0, 0, 0, 0)
        self.h_layout_main.setAlignment(Qt.AlignCenter)
        self.h_layout_main.addWidget(self.widget_central)
        self.h_layout_main.setSpacing(0)
        self.h_layout_main.addWidget(self.widget_sub)

        # Main widget
        self.widget_main = QWidget()
        self.widget_main.setContentsMargins(0, 0, 0, 0)
        self.widget_main.setLayout(self.h_layout_main)
        self.widget_main.setStyleSheet("background-color: #041929;")

        # ----------------------------------------------------------
        # Main Window
        self.setCentralWidget(self.widget_main)
        self.setStyleSheet("background-color: #2C303C;")




    #################################
    ###    Btn action listener    ###
    #################################
    def label_btn_exit_mouseReleaseEvent(self, event):
        self.video_thread.stop()
        self.inf_thread.stop()
        sys.exit(0)


    ########################################
    #                main                  #
    ########################################

    # global shortcut
    def keyReleaseEvent(self, event):
       if event.key() == Qt.Key_Q:
            # sys.exit(0)
            self.label_btn_exit_mouseReleaseEvent(event)
       else:
           pass

    # log
    @Slot(str)
    def log(self, text=''):
        self.mutex.lock()
        now = datetime.now()
        nowTime = now.strftime('%y/%m/%d_%H:%M:%S.%f')[:-3]
        try:
            print('[{0}] {1}'.format(nowTime, text))
            # self.log('[{0}] {1}'.format(nowTime, text))
        except:
            # print('[{0}][ERROR][log] pathLog: {1}, msg: {2}'.format(nowTime, pathLog, traceback.format_exc()))
            print('[{0}][ERROR][log] msg: {1}'.format(nowTime, traceback.format_exc()))
            # self.log('[{0}][ERROR][log] msg: {1}'.format(nowTime, traceback.format_exc()))

        self.mutex.unlock()


    def setup_thread(self):
        
        mode = 'webcam'
        self.video_thread = VideoThread(mode,
                                        parent=self)
        
        self.inf_thread = InferenceThread(self.video_thread,
                                          self.person_detector,
                                          self.pose_estimator,
                                          self.action_recognizer,
                                          parent=self)
        self.inf_thread.signalSetImage.connect(self.canvas.update_frame)
        
        self.video_thread.start()
        self.inf_thread.start()

    def thread_close(self):
        self.inf_thread.stop()
        self.video_thread.stop()
        # event.accept()
        
    ############################################
    #              controller                  #
    ############################################
    # ROI 안에 bbox의 하단이 있는지 확인
    def is_inside_roi(self, bbox):
        x1, y1, x2, y2 = bbox
        #bbox 하단 찾기
        cx = (x1 + x2) / 2
        cy = y2
        
        result = cv2.pointPolygonTest(self.roi, (cx, cy), False)
        
        return result >= 0
    
    # 오래된 id 제거
    def cleanup_old_ids(self, now):

        remove_ids = []

        for human_id, last_seen in self.id_last_seen.items():

            if human_id == self.active_user_id:
                continue

            if now - last_seen > 10: # 10초 이상 보이지 않는 id는 제거
                remove_ids.append(human_id)

        for human_id in remove_ids:

            self.roi_enter_time.pop(human_id, None)
            self.id_last_seen.pop(human_id, None)
            
    # ROI 처리
    def handle_roi(self, detections):
        
        now = time.time()
        ids_in_roi = []
                
        for det in detections:
            bbox = det['bbox']
            human_id = det['id']
            if self.is_inside_roi(bbox):
                
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                ids_in_roi.append({'id': human_id, 
                                   'area': area, 
                                   'bbox': bbox})
            
        target_id = None
        if len(ids_in_roi) > 0:
            # ROI 안에 여러 명이 있다면, 가장 큰 bbox를 가진 사람을 선택
            target_id = max(ids_in_roi, key=lambda x: x['area'])['id']

            if target_id not in self.roi_enter_time: # 처음 들어온 사람이라면
                self.roi_enter_time[target_id] = now

            self.id_last_seen[target_id] = now
        
        if self.current_view_mode == 'video':
            if target_id is not None:
                stay = now - self.roi_enter_time[target_id]
                if stay >= self.roi_threshold_sec:
                    self.active_user_id = target_id
                    self.current_view_mode = 'webcam'
                    print('Switch to webcam mode!')
        
         
                    
        elif self.current_view_mode == 'webcam':
            if self.active_user_id is not None: # active_user_id이 존재한다면
                
                if self.active_user_id not in ids_in_roi: # active_user_id이 roi 안에 있는지 확인 (없는경우, 탈주)
                    check_time = now - self.id_last_seen[self.active_user_id] # 탈주
                    if check_time >= self.lost_threshold_sec:
                        if ids_in_roi: # 다른 사람이 roi에 들어온 경우, active_user_id 초기화
                            new_active_id = max(ids_in_roi, key=lambda x: x['area'])['id']
                            stay = now - self.roi_enter_time.get(new_active_id, now)
                            if stay >= self.new_user_threshold_sec:
                                self.active_user_id = new_active_id # active_user_id 변경
                                # self.current_view_mode = 'webcam'
                                print('Active user left, but another person is in the ROI. Switching to new active user!')
                                break
                        # roi내 사람이 안들어왔네    
                        print('Switch to video mode!')
                        self.current_view_mode = 'video'
                        self.roi_enter_time.pop(self.active_user_id, None)
                        self.id_last_seen.pop(self.active_user_id, None)
                        self.active_user_id = None
                        # 추후에 countdown 기능 추가 예정
                else: # active_user_id이 roi 안에 있다면
                    self.id_last_seen[self.active_user_id] = now # active_user_id가 roi에 있으므로 last_seen 업데이트
            else: # active_user_id이 존재하지 않는다면 (탈주상황)
                print('[Debug] No active user found. Checking for new users in ROI...')
                if ids_in_roi: # 다른 사람이 roi에 들어온 경우, active_user_id 초기화
                    new_active_id = max(ids_in_roi, key=lambda x: x['area'])['id']
                    stay = now - self.roi_enter_time.get(new_active_id, now)
                    if stay >= self.new_user_threshold_sec:
                        self.active_user_id = new_active_id # active_user_id 변경
                        print('No active user, but another person is in the ROI. Switching to new active user!')
                pass
     


            
    @Slot(dict)
    def update_controller(self, result_dict):
        
        frame = result["frame"]
        detections = result["detections"]
        human_exists = result["human_exists"]
        self.handle_roi(detections)
        
        if len(self.id_last_seen) > 50:
            print(f"[ROI] tracking ids : {len(self.id_last_seen)}")

            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MW(app=app)
    window.showFullScreen()
    sys.exit(app.exec())

