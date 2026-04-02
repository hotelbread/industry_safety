import os
import sys
import cv2
import csv
import time
import bisect
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

# from gtts import gTTS
from src.module.audio_manager import AudioManager
from src.module.evaluation_module import ExperienceEvaluator

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QApplication, QStackedWidget, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView 
from PySide6.QtCore import Qt, QMutex, Slot, QUrl, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput #동영상용
from PySide6.QtMultimediaWidgets import QVideoWidget        #동영상용

# from src.ai_module.pose_estimater import PoseEstimator
# from src.ai_module.person_detector import PersonDetector
# from src.ai_module.action_recognizer import ActionRecognizer

from src.gui_module.canvas import Canvas
from src.module.ai_thread import AiThread
# from src.module.vid_thread import VidThread
# from src.module.video_thread import VideoThread
# from src.module.inf_thread import InferenceThread
import src.misc.tools as tools
# import numpy as np
# from PyQt5.QtWidgets import QTableWidgetItem
# from PyQt5 import QtCore
# from PyQt5.QtGui import QColor
# from django.utils.timezone import now

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

class MW(QMainWindow):
    signalviewinfo = Signal(dict)
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
        
        self.COLOR_BRIGHT_GRAY = '#CECECE'
        self.COLOR_NEW_BG   = '#1A6B4A'   # 새 행 배경
        self.COLOR_NEW_TEXT = '#A8FFDA'   # 새 행 텍스트
        self.COLOR_ODD_BG   = '#3D3D3D'   # 기존 홀수 행 배경 (기존 AlternatingRowColors 대체)
        self.COLOR_EVEN_BG  = '#2C2C2C'   # 기존 짝수 행 배경
        self.COLOR_TEXT = '#E2E8F0'
        
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
        
        ##################### 태원 추가#######################
        self.admin_mode = False
        self.video_path = os.path.join(self.RESOURCE_DIR, 'res/Vid/visol.mp4')
        self.current_view_mode = 'webcam'  # 'video' or 'webcam'
        self.roi = [(1200, 450), (1200, 850), (800, 450), (850, 800)]  # 예시 ROI 좌표 (x1, y1, x2, y2, x3, y3, x4, y4) roi = [(x_video, y_video)]
        # self.roi = np.array([[1200, 450], [1200, 850], [800, 450], [850, 800]])  # 예시 ROI 좌표 (x1, y1, x2, y2, x3, y3, x4, y4)
        # self.ids_in_roi = []
        self.roi_enter_time = {}
        self.id_last_seen = {}
        self.active_user_id = None
        self.roi_threshold_sec = 3 # 체류 기준
        self.lost_threshold_sec = 5 # n초 동안 active_user가 없으면 탈주
        self.new_user_threshold_sec = 2 # 새로 들어온 사람은 n초 이상 서있으면 인정
        self.countdown = None
        self.countdown_duration = 3  # seconds
        self.rank = 1
        self.stage = 0 # current_view 아래에서 세부단위 조절용
        self.human_time = 0 # roi내에 서있던시간
        self.experience_start_check = 0
        self.action_recogniton = False
        self.countdown_off = True
        self._eval_stage = -1 # 0, 1, 2 / -1은 수집X 
        self._highlighted_rank_row = None # 마지막 rank table 업데이트한 행 기억

        # self.STAGE_NAMES = ["붐 올리기", "권상", "비상 정지"]
        self.STAGE_NAMES = {}
        label_map_path = 'model/action/label_map_hsd_S001.txt'
        try:
            labels = [x.strip() for x in open(label_map_path).readlines()]
            self.STAGE_NAMES = {i: label for i, label in enumerate(labels)}
        except:
            self.STAGE_NAMES = {
                0 : 'Etc',
                1 : "Raise Boom",
                2 : "Raise Load",
                3 : "Emergency Stop"
            }
        
        # self.STAGE_NAMES = ['Etc', "Raise Boom", "Raise Load", "Emergency Stop"]
        self.STAGE_CLASS_IDXS = [2,1,0]
        self.exp_evaluator = ExperienceEvaluator(self.STAGE_NAMES, self.STAGE_CLASS_IDXS)
        self.audio = AudioManager()
        self.action_flag = 0
           
        ######################################################
        # -----------------------------------------------------------------
        self.init_res()
        # self.init_ai_model()
        self.init_ui()
        self.setup_thread()
        # -----------------------------------------------------------------

    def init_res(self):
        self.pixmap_logo = tools.get_resized_pixmap_based_h(self.PATH_LOGO, self.LABEL_LOGO_HEIGHT)
        self.pixmap_btn_exit = tools.get_resized_pixmap_based_h(self.PATH_BTN_EXIT, self.BTN_EXIT_HEIGHT)

    # def init_ai_model(self):
    #     """set up the AI model"""
    #     # self.device_name =  cpu'        # for cpu
    #     self.device_name = 'cuda:0'     # for GPU Desktop
    #     self.person_detector = PersonDetector(self.device_name, self.RESOURCE_DIR)
    #     self.pose_estimator = PoseEstimator(self.device_name, self.RESOURCE_DIR)
    #     self.action_recognizer = ActionRecognizer(self.device_name, self.RESOURCE_DIR)

    def init_ui(self):
        ########################################
        #               init GUI               #
        ########################################
        """
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
        """
        # Canvas 
        self.canvas = Canvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        # self.canvas.setFixedSize(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        
        # self.layout_cam = QHBoxLayout()
        # self.layout_cam.setContentsMargins(0, 0, 0, 0)
        
        # self.layout_cam.addWidget(self.widget_sub, 2)
        # self.layout_cam.addWidget(self.canvas, 5)

        # # Central widget
        # self.widget_central = QWidget()
        # self.widget_central.setContentsMargins(self.SPACING, self.SPACING, self.SPACING, self.SPACING)
        # # self.widget_central.setFixedSize(self.WIDGET_CENTRAL_WIDTH, self.WIDGET_CENTRAL_HEIGHT)
        # self.widget_central.setStyleSheet("border: 2px solid #ffffff")
        
        # 메인 화면쪽
        # self.stack = QStackedWidget()
        
        # layout_central = QVBoxLayout()
        # layout_central.setContentsMargins(0, 0, 0, 0)
        # layout_central.addWidget(self.stack)
        
        # self.widget_central.setLayout(layout_central)
        
        # self.page_video = QWidget()
        
        
        # self.video_widget = QVideoWidget()
        # self.layout_video = QVBoxLayout()
        # self.layout_video.addWidget(self.video_widget)
        
        # self.page_video.setLayout(self.layout_video)
        
        # self.media_player = QMediaPlayer()
        
        # self.audio_output = QAudioOutput()
        # self.media_player.setAudioOutput(self.audio_output)
        
        # self.media_player.setVideoOutput(self.video_widget)
        # self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
        
        # self.media_player.play()
        # self.media_player.mediaStatusChanged.connect(self.loop_video)
        
        
        # 웹캠 버전!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.page_webcam = QWidget() 
        
        self.layout_cam = QHBoxLayout()
        self.layout_cam.setContentsMargins(0, 0, 0, 0)
        
        # stack에 추가!!!(sub)
        # self.stack.addWidget(self.page_webcam)
        
        # Sub widget
        self.widget_sub = QStackedWidget()
        # self.widget_sub.setContentsMargins(self.SPACING, self.SPACING, self.SPACING, self.SPACING)
        self.widget_sub.setStyleSheet("border: 2px solid #ffffff")
        
        # Sub widget 내 단계별 레이아웃 추가 (stack활용)
        self.widget_sub4_0 = QWidget() # 처음 빈 화면
        self.widget_sub4_1 = QWidget() # 시작예고
        self.widget_sub4_2 = QWidget() # 체험중
        self.widget_sub4_3 = QWidget() # 평가화면
        
        self.anounce_box_1 = QLabel()
        self.anounce_box_1.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.anounce_box_1.setFixedHeight(100)
                
        self.anounce_box_2 = QLabel()
        self.anounce_box_2.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.anounce_box_2.setFixedHeight(100)
        
        self.anounce_box_3 = QLabel()
        self.anounce_box_3.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.anounce_box_3.setFixedHeight(100)
        
        self.explain_box_2 = QLabel(f"1 단계. '붐 올리기' 동작을 취해보세요")
        self.explain_box_2.setStyleSheet(f"background-color : {self.COLOR_BRIGHT_GRAY};color: white; font-size: 20px; font-weight: bold;")
        self.explain_box_2.setFixedHeight(100)
        
        self.explain_box_3 = QLabel(f"1 단계. '붐 올리기' 동작을 취해보세요")
        self.explain_box_3.setStyleSheet(f"background-color : {self.COLOR_BRIGHT_GRAY};color: white; font-size: 20px; font-weight: bold;")
        self.explain_box_3.setFixedHeight(100)
      
        self.widget_scoreboard = QLabel("수신호 동작 평가")
        self.widget_scoreboard.setAlignment(Qt.AlignCenter)
        self.widget_scoreboard.setStyleSheet(f"background-color: {self.COLOR_BRIGHT_GRAY}; color: black; font-size: 20px; font-weight: bold;")
        self.widget_scoreboard.setFixedHeight(40)
        self.widget_scoreboard.setContentsMargins(0, 200, 0, 200)
        
        self.widget_scoretable = QTableWidget()
              
        self.widget_scoretable.setRowCount(5)
        self.widget_scoretable.setColumnCount(5)
        
        self.widget_scoretable.setHorizontalHeaderLabels(['No.', '동작', '정확도', '안정성', '평균'])
        self.widget_scoretable.horizontalHeader().setFixedHeight(60)
        self.widget_scoretable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.widget_scoretable.verticalHeader().setDefaultSectionSize(60)
        self.widget_scoretable.verticalHeader().setVisible(False)
        self.widget_scoretable.setSortingEnabled(True) # 등수에 따라 자동 정렬해주는거 추가해야함. 아직 안함
        self.widget_scoretable.setAlternatingRowColors(False)
        # self.widget_scoretable.setAlternatingRowColors(True) # 행마다 색깔 다르게 (가독성 위해)
        
        self.widget_scoretable.setStyleSheet("""QTableWidget {
                                    background-color: white;
                                    gridline-color: gray;
                                    font-size: 14px;
                                }

                                QHeaderView::section {
                                    background-color: #888;
                                    color: white;
                                }
                                """)
        
        
      
        self.widget_scoretext = QLabel(f"축하합니다! 성공하셨습니다.\n 현재 {self.rank}입니다.")
        self.widget_scoretext.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        
        
        self.layout_sub4_1 = QVBoxLayout()
        self.layout_sub4_1.addWidget(self.anounce_box_1)
        self.layout_sub4_1.addStretch()
        
        self.layout_sub4_2 = QVBoxLayout()
        self.layout_sub4_2.addWidget(self.anounce_box_2)
        self.layout_sub4_2.addWidget(self.explain_box_2)
        self.layout_sub4_2.addStretch()

        
        self.layout_sub4_3 = QVBoxLayout()
        self.layout_sub4_3.addWidget(self.anounce_box_3)
        self.layout_sub4_3.addWidget(self.explain_box_3)
        self.layout_sub4_3.addWidget(self.widget_scoreboard)
        self.layout_sub4_3.addWidget(self.widget_scoretable)
        self.layout_sub4_3.addWidget(self.widget_scoretext)
        
        self.widget_sub4_1.setLayout(self.layout_sub4_1)
        self.widget_sub4_2.setLayout(self.layout_sub4_2)
        self.widget_sub4_3.setLayout(self.layout_sub4_3)
        
        self.widget_sub.addWidget(self.widget_sub4_0)
        self.widget_sub.addWidget(self.widget_sub4_1)
        self.widget_sub.addWidget(self.widget_sub4_2)
        self.widget_sub.addWidget(self.widget_sub4_3)
        
                
        self.layout_cam.addWidget(self.widget_sub, 2)
        self.layout_cam.addWidget(self.canvas, 5)
        
        self.page_webcam.setLayout(self.layout_cam)
        
        # stack에 추가!!!(메인)
        # self.stack.addWidget(self.page_video)
        # self.stack.addWidget(self.page_webcam)
        
        
        # 메인화면 레이아웃 (4, 4-1보조설명widget_sub, 4-2메인화면widget_central)
        self.h_layout_main = QHBoxLayout()
        self.h_layout_main.setContentsMargins(0, 0, 0, 0)
        self.h_layout_main.addWidget(self.page_webcam)
        # self.h_layout_main.addWidget(self.stack)
        # self.h_layout_main.setContentsMargins(0, 0, 0, 0)
        # self.h_layout_main.setAlignment(Qt.AlignCenter)
        # self.h_layout_main.addWidget(self.widget_sub, 2)
        # self.h_layout_main.setSpacing(0)
        # self.h_layout_main.addWidget(self.widget_central, 5)
        # self.page_webcam.setLayout(self.h_layout_main)
        # page_webcam에 담기
        
        # 화면 + 버튼(3, 3-2 버튼(시작, 체험안내, 동작시연, 동작수행, 평가))
        self.v_layout_second = QVBoxLayout()
        self.v_layout_second.setContentsMargins(0, 0, 0, 0)
        self.v_layout_second.addLayout(self.h_layout_main)
        
        """
        # 전환버튼
        self.btn_layout = QHBoxLayout()
        self.btn_layout.setContentsMargins(0, 0, 0, 0)
  
        self.btn_start = QPushButton("시작")
        self.btn_start.setStyleSheet('font-size:20px; color:white')
        self.btn_guide = QPushButton("체험안내")
        self.btn_guide.setStyleSheet('font-size:20px; color:white')
        self.btn_demo = QPushButton("동작시연")
        self.btn_demo.setStyleSheet('font-size:20px; color:white')
        self.btn_action = QPushButton("동작수행")
        self.btn_action.setStyleSheet('font-size:20px; color:white')
        self.btn_score = QPushButton("평가")
        self.btn_score.setStyleSheet('font-size:20px; color:white')

        self.btn_layout.addWidget(self.btn_start)
        self.btn_layout.addWidget(self.btn_guide)
        self.btn_layout.addWidget(self.btn_demo)
        self.btn_layout.addWidget(self.btn_action)
        self.btn_layout.addWidget(self.btn_score)
        
        # self.btn_start.clicked.connect(self.video_start)
        self.btn_guide.clicked.connect(self.webcam_start)
        self.btn_demo.clicked.connect(self.mode_1)
        self.btn_action.clicked.connect(self.mode_2_1)
        self.btn_score.clicked.connect(self.mode_3)
        
        self.v_layout_second.addLayout(self.btn_layout)
        """
        
        ## rank board layout (2-2, 시계 + 제목 + 점수보드 + 취소버튼)
        # label date
        self.clock_date = QLabel()
        self.clock_date.setText('0000. 00. 00 (Mon)')
        self.clock_date.setStyleSheet('font-size: 30px; color: black;')
        self.clock_date.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

        # label time
        self.clock_time = QLabel()
        self.clock_time.setText('00:00:00')
        self.clock_time.setStyleSheet('font-size: 50px; color: black;')
        self.clock_time.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        # layout date & time
        self.layout_clock = QVBoxLayout()
        self.layout_clock.setAlignment(Qt.AlignTop)
        self.layout_clock.addWidget(self.clock_date,1)
        # self.layout_clock.addStretch()
        self.layout_clock.addWidget(self.clock_time,1)
        # self.layout_clock.addStretch()
        
        self.widget_rankboard = QLabel("평가랭킹")
        self.widget_rankboard.setAlignment(Qt.AlignCenter)
        self.widget_rankboard.setStyleSheet(f"background-color: {self.COLOR_BRIGHT_GRAY}; color: black; font-size: 20px; font-weight: bold;")
        self.widget_rankboard.setFixedHeight(40)
        self.widget_rankboard.setContentsMargins(0, 100, 0, 100)
        
        self.widget_rank = QTableWidget()
        
        self.widget_rank.setRowCount(0)
        # self.widget_rank.setRowCount(30)
        self.widget_rank.setColumnCount(2)
        
        # self.widget_rank.setHorizontalHeaderLabels(['순위', '이름', '체험시간', '점수'])
        self.widget_rank.setHorizontalHeaderLabels(['순위', '점수'])
        self.widget_rank.horizontalHeader().setFixedHeight(60)
        self.widget_rank.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.widget_rank.verticalHeader().setDefaultSectionSize(60)
        self.widget_rank.verticalHeader().setVisible(False)
        self.widget_rank.setSortingEnabled(True) # 등수에 따라 자동 정렬해주는거 추가해야함. 아직 안함
        self.widget_rank.setAlternatingRowColors(True) # 행마다 색깔 다르게 (가독성 위해)
        
        self.widget_rank.setStyleSheet("""QTableWidget {
                                    background-color: white;
                                    gridline-color: gray;
                                    font-size: 14px;
                                }

                                QHeaderView::section {
                                    background-color: #888;
                                    color: white;
                                }
                                """)
        
        # 체험초기화 버튼
        self.btn_rank_reset = QPushButton("체험 초기화")
        self.btn_rank_reset.setStyleSheet(f"background-color: {self.COLOR_BRIGHT_GRAY}; color: black; font-size: 20px;")
        self.btn_rank_reset.setFixedHeight(40)
        self.btn_rank_reset.clicked.connect(self._on_reset_btn)
        
        
        self.layout_rank = QVBoxLayout()
        self.layout_rank.setContentsMargins(10, 10, 10, 10)
        self.layout_rank.addLayout(self.layout_clock,2)
        self.layout_rank.addWidget(self.widget_rankboard, 1)
        self.layout_rank.addWidget(self.widget_rank,7)
        self.layout_rank.addWidget(self.btn_rank_reset, 1)
        self.layout_rank_bg = QWidget()
        self.layout_rank_bg.setLayout(self.layout_rank)
        self.layout_rank_bg.setStyleSheet(f"background-color: {self.COLOR_LIGHT_GRAY}")
        
        # Main horizontal layout (2, 화면2-1 + 점수보드2-2)
        self.h_layout_board = QHBoxLayout()
        self.h_layout_board.setContentsMargins(0, 0, 0, 0)
        self.h_layout_board.addLayout(self.v_layout_second,9)
        self.h_layout_board.addWidget(self.layout_rank_bg,2)
        
        # head layout (Label logo + 글자 + Btn exit)
        self.head_logo = QLabel()
        self.head_logo.setPixmap(self.pixmap_logo)
        
        self.head_text = QLabel("Industrial Safety")
        self.head_text.setStyleSheet("color: #ffffff; font-size: 40px; font-weight: bold;")
        self.head_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.head_btn_exit = QLabel()
        self.head_btn_exit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.head_btn_exit.setPixmap(self.pixmap_btn_exit)
        self.head_btn_exit.mouseReleaseEvent = self.label_btn_exit_mouseReleaseEvent
        
        self.head_layout = QHBoxLayout()
        self.head_layout.setContentsMargins(0, 0, 0, 0)
        self.head_layout.addWidget(self.head_logo)
        self.head_layout.addWidget(self.head_text)
        self.head_layout.addStretch()
        self.head_layout.addWidget(self.head_btn_exit)
        # self.widget_head = QWidget()
        # self.widget_head.setContentsMargins(0, 0, 0, 0)
        # self.widget_head.setStyleSheet("border: 2px solid #ffffff")
                
        # first Main Vertical layout (1, 1-1, 1-2)
        self.v_layout_first = QVBoxLayout()
        self.v_layout_first.addLayout(self.head_layout)
        self.v_layout_first.addLayout(self.h_layout_board)
        self.v_layout_first.setContentsMargins(0, 0, 0, 0)
        # self.v_layout_first.setStyleSheet("border: 2px solid #ffffff")
        
        # Main widget
        self.widget_main = QWidget()
        self.widget_main.setContentsMargins(0, 0, 0, 0)
        self.widget_main.setLayout(self.v_layout_first)
        self.widget_main.setStyleSheet("background-color: #041929;")

        # ----------------------------------------------------------
        # Main Window
        self.setCentralWidget(self.widget_main)
        self.setStyleSheet("background-color: #2C303C;")
        
        # 초기셋팅 (0번:영상, 1번:웹캠)
        # self.stack.setCurrentIndex(1)
        # self.stack.setCurrentIndex(1)

    def loop_video(self, status):
        
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.setPosition(0)
            self.media_player.play()
            
    def timeout(self):
        now = datetime.now()
        self.clock_date.setText(f'{now.strftime("%Y. %m. %d (%a)")}')
        self.clock_time.setText(f'{now.strftime("%H:%M:%S")}')

    #################################
    ###    Btn action listener    ###
    #################################
    # def mousePressEvent(self, event):
        
    #     if not self.admin_mode:
    #         return
        
    #     if event.button() == Qt.LeftButton:
    #         x = event.position().x()
    #         y = event.position().y()
            
    #         self.roi.append((x, y))
            
    #         print(f'Added point: ({x}, {y})')
    #         # self.canvas.update()
            
    def video_start(self):
        self.current_view_mode = 'video'
        self.stack.setCurrentIndex(0)
        print(f'[Debug][Main] Switched to video mode')
        
        self.active_user_id = None
    
    def webcam_start(self):
        self.current_view_mode = 'webcam'
        # self.stack.setCurrentIndex(1)
        self.widget_sub.setCurrentWidget(self.widget_sub4_0)
        print(f'[Debug][Main] Switched to webcam mode')
        
    def mode_1(self):
        self.widget_sub.setCurrentWidget(self.widget_sub4_1)
        ready_time = 3
        act_time = 5
        self.anounce_box_1.setText(f'준비되셨나요. \n 그럼 {ready_time}초 후 시작합니다! \n 각 동작을 {act_time}초간 진행하세요')        
        print(f'[Debug][button1] mode1')
        
    def mode_2_1(self):
        self.widget_sub.setCurrentWidget(self.widget_sub4_2)
        ready_time = 3
        act_time = 5
        self.anounce_box_2.setText(f'준비되셨나요. \n 그럼 {ready_time}초 후 시작합니다! \n 각 동작을 {act_time}초간 진행하세요')        
        self.explain_box_2.setText(f'1단계. \n \"붐 올리기\" 동작을 취해보세요.')
        print(f'[Debug][button2] mode2_1')
    
    def mode_2_2(self):
        self.widget_sub.setCurrentWidget(self.widget_sub4_2)
        ready_time = 3
        act_time = 5
        self.anounce_box_2.setText(f'준비되셨나요. \n 그럼 {ready_time}초 후 시작합니다! \n 각 동작을 {act_time}초간 진행하세요')        
        self.explain_box_2.setText(f'2단계. \n \"권상\" 동작을 취해보세요.')
        print(f'[Debug][button2] mode2_2')
        
    def mode_2_3(self):
        self.widget_sub.setCurrentWidget(self.widget_sub4_2)
        ready_time = 3
        act_time = 5
        self.anounce_box_2.setText(f'준비되셨나요. \n 그럼 {ready_time}초 후 시작합니다! \n 각 동작을 {act_time}초간 진행하세요')        
        self.explain_box_2.setText(f'3단계. \n \"비상 정지\" 동작을 취해보세요.')
        print(f'[Debug][button2] mode2_3')
        
    def mode_3(self):
        self.widget_sub.setCurrentWidget(self.widget_sub4_3)
        ready_time = 3
        act_time = 5
        self.anounce_box_3.setText(f'준비되셨나요. \n 그럼 {ready_time}초 후 시작합니다! \n 각 동작을 {act_time}초간 진행하세요')        
        self.explain_box_3.setText(f'3단계. \n \"비상 정지\" 동작을 취해보세요.')
        print(f'[Debug][button3] mode3')
        
    def label_btn_exit_mouseReleaseEvent(self, event):
        # self.video_thread.stop()
        # self.video_thread.wait()
        # self.ai_thread.stop()
        # self.ai_thread.wait()
        # event.accept()
        # self.inf_thread.stop()
        sys.exit(0)

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_Q:
    #         self.label_btn_exit_mouseReleaseEvent(event)
    #     elif event.key() == Qt.Key_F:
    #         if self.isFullScreen():
    #             self.showNormal()
    #         else:
    #             self.showFullScreen()
    #     elif event.key() == Qt.Key_P:
    #         self.canvas.roi_points = []
    #         if self.admin_mode == False:
    #             self.admin_mode = True
    #             self.canvas.admin_mode = True
    #             print("Admin mode activated")
    #     elif event.key() == Qt.Key_D:
    #         print(f'roi_enter_time : {self.roi_enter_time}')
    #         print(f'id_last_seen : {self.id_last_seen}')
    #         print(f'active_user_id : {self.active_user_id}')
    #     elif event.key() == Qt.Key_Escape:
    #         if self.admin_mode == True:
    #             self.admin_mode = False
    #             self.canvas.admin_mode = False
    #             print("Admin mode deactivated")
    #     else:
    #         pass
        
    def save_rank_csv(self):
        now = datetime.now()
        file_name = now.strftime('%y%m%d_%H%M%S') + '_rank_reset_log.csv'
        with open(file_name, 'w', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            headers = ['Rank', 'Name', 'Time', 'Score']
            writer.writerow(headers)
            for row in range(self.widget_rank.rowCount()):
                row_data = []
                for column in range(self.widget_rank.columnCount()):
                    item = self.widget_rank.item(row, column)
                    row_data.append(item.text() if item else '')
                writer.writerow(row_data)
        
    def clear_rank_table(self):        
        self.widget_rank.clearContents()

    def rank_reset(self):

        self.save_rank_csv() # csv파일로 저장 기능
        self.clear_rank_table()
        
    def reset_experience(self):
        print('[Debug][Main] Reset experience mode]')
        # Ai 감지
        self.active_user_id = None
        self.roi_enter_time = {}
        self.id_last_seen = {}
        # 체험흐름
        self.current_view_mode = 'webcam'
        self.countdown = None
        self.countdown_off = True
        self.experience_start_check = 0
        self.human_time = 0
        self.action_recogniton = False
        # 평가 모듈
        self._eval_stage = -1
        self.exp_evaluator.reset()
        # gui 초기화
        self.webcam_start()
        print('[Debug][Main] Experience mode reset 완료!!!]')
        
    def _on_reset_btn(self):
        # self.rank_reset()           # csv 저장 + 테이블초기화
        self.reset_experience()     # 체험 상태 초기화
        
        
    ########################################
    #                main                  #
    ########################################

    # global shortcut
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Q:
            # sys.exit(0)
            self.label_btn_exit_mouseReleaseEvent(event)
        elif event.key() == Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_P:
            self.canvas.roi_points = []
            if self.admin_mode == False:
                self.admin_mode = True
                self.canvas.admin_mode = True
                print("Admin mode activated")
        elif event.key() == Qt.Key_D:
            print(f'roi_enter_time : {self.roi_enter_time}')
            print(f'id_last_seen : {self.id_last_seen}')
            print(f'active_user_id : {self.active_user_id}')
        elif event.key() == Qt.Key_Escape:
            if self.admin_mode == True:
                self.admin_mode = False
                self.canvas.admin_mode = False
                print("Admin mode deactivated")
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
        
        # mode = 'webcam'
        # self.video_thread = VideoThread(mode,
        #                                 parent=self)
        # self.video_thread = VidThread(self.video_path, parent=self)
        # self.video_thread.signalSetImage.connect(self.canvas.update_frame)
        self.ai_thread = AiThread(self.RESOURCE_DIR, parent=self)
        
        # self.ai_thread = AiThread(self.webcam_thread, parent=self)
        # self.inf_thread = InferenceThread(self.video_thread,
        #                                   self.person_detector,
        #                                   self.pose_estimator,
        #                                   self.action_recognizer,
        #                                   parent=self)
        # self.inf_thread.signalSetImage.connect(self.canvas.update_frame)
        self.signalviewinfo.connect(self.ai_thread.update_status) # main -> ai_thread
        self.ai_thread.signalSetImage.connect(self.update_controller) # ai_thread -> main
        
        # self.video_thread.start()
        self.ai_thread.start()
        # self.inf_thread.start()
        
        # timer
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)

        self.timer.start()
        
        # 오디오 상태 업데이트 전용 타이머
        self.audio_timer = QTimer()
        self.audio_timer.setInterval(100)   # 0.2초마다 체크
        self.audio_timer.timeout.connect(self.audio.update)
        self.audio_timer.start()

    # def thread_close(self):
    #     # self.inf_thread.stop()
    #     self.video_thread.stop()
    #     self.video_thread.wait()
    #     event.accept()
        
    ############################################
    #              controller                  #
    ############################################
    # ROI 안에 bbox의 하단이 있는지 확인
    def is_inside_roi(self, bbox):
        x1, y1, x2, y2 = bbox
        #bbox 하단 찾기
        cx = (x1 + x2) / 2
        cy = y2
        
        if len(self.canvas.roi_points)<3:
            return False
        
        self.roi = np.array(self.canvas.roi_points, np.int32)
        result_dict = cv2.pointPolygonTest(self.roi, (cx, cy), False)
        
        return result_dict >= 0
    
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
            
    def countdown_fun(self, check_time, countdown_time=5):
        self.countdown_off = False
        if check_time > self.roi_threshold_sec:
            if self.experience_start_check == 0:
                self.experience_start_check = time.time()
            duration = countdown_time + 1 - int(time.time() - self.experience_start_check)
            font_color = {10 : 'red', 3: 'green', 5 :'blue', 1 : 'black'}
            self.countdown = {'color' : font_color[countdown_time],
                         'time' : duration-1}
            if duration < 0:
                # self.human_time = 0
                self.experience_start_check = 0
                self.countdown_off = True
                self.countdown = None
        else :
            self.countdown = None
            self.experience_start_check = 0 # 다시 초기화
            self.countdown_off = True
        # self.canvas.update_frame(frame, countdown)
    # ROI 처리
    def handle_roi(self, detections):
        """_summary_

        Args:
            detections (_type_): tracked_bboxes, (M, 7)  ex) [x1, y1, x2, y2, id, class, score]
            
        """
        now = time.time()
        ids_in_roi = []
        target_id = None
        
        if len(detections) > 0:
            for det in detections:
                if len(det) != 7: # tracking이 되지 않은 경우
                    continue
                bbox = det[:4]
                human_id = int(det[4])
                if self.is_inside_roi(bbox): # True 면 내부에 존재
                    
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    ids_in_roi.append({'id': human_id, 
                                    'area': area, 
                                    'bbox': bbox})
        if len(ids_in_roi) > 0:
            # ROI 안에 여러 명이 있다면, 가장 큰 bbox를 가진 사람을 선택
            target_id = max(ids_in_roi, key=lambda x: x['area'])['id']
            # print(f'[Debug][handle_roi] type: {type(target_id)}, target_id : {target_id} )')

            if target_id not in self.roi_enter_time: # 처음 들어온 사람이라면
                self.roi_enter_time[target_id] = now
            self.id_last_seen[target_id] = now
            if target_id is not None:
                stay = now - self.roi_enter_time[target_id]
                if stay >= self.roi_threshold_sec:
                    self.active_user_id = target_id
        
        # active_user_id가 있는지 체크(target) ----
        if self.active_user_id is not None: # active_user_id이 존재한다면
            # print(f'[Debug][handle_roi] ids_in_roi {ids_in_roi}')
            # ids_in_roi : id, area, bbox 묶음을 리스트로
            # ids_in_roi_ids : id 만 추출해서 리스트
            ids_in_roi_ids = [x['id'] for x in ids_in_roi]
            # print(f'[Debug][handle_roi] ids_in_roi_ids {ids_in_roi_ids}')
            if self.active_user_id not in ids_in_roi_ids: # active_user_id이 roi 안에 있는지 확인 (없는경우, 탈주)
                check_time = now - self.id_last_seen[self.active_user_id] # 탈주
                if check_time >= self.lost_threshold_sec:
                    print(f'[Debug][handle_roi] User {self.active_user_id} lost for {check_time} seconds!]')
                    if ids_in_roi_ids: # 다른 사람이 roi에 들어온 경우, active_user_id 초기화
                        new_active_id = max(ids_in_roi, key=lambda x: x['area'])['id'] # target_id 랑 같을수도???
                        stay = now - self.roi_enter_time.get(new_active_id, now)
                        print(f'[Debug][handle_roi] 새로운사람 등장 id : {new_active_id}')
                        print(f'[Debug][handle_roi] stay : {stay}')
                        if stay >= self.new_user_threshold_sec: # 2초이상 머물고 있다면 이사람이다!
                            self.active_user_id = new_active_id # active_user_id 변경
                            self.countdown = None
                            # self.countdown_off =False # 지워야할 수 도있음
                            self.experience_start_check = 0 # 초기화
                            # self.current_view_mode = 'webcam'
                            print('Active user left, but another person is in the ROI. Switching to new active user!')
                            print(f'New active user ID: {self.active_user_id}')
                    else:       
                        # roi내 사람이 안들어왔네    
                        self.countdown_fun(check_time, 10) # 10초동안 
                        if self.experience_start_check == 0:
                            print('User lost. Reset experience mode')
                            self.reset_experience()
                            # self.roi_enter_time.pop(self.active_user_id, None)
                            # self.id_last_seen.pop(self.active_user_id, None)
                            # self.active_user_id = None
            else: # active_user_id가 roi 안에 있다면
                self.id_last_seen[self.active_user_id] = now # active_user_id가 roi에 있으므로 last_seen 업데이트
        else:
            print('[Debug][handle_roi] No active user found.')
            return

    def show_video_mode(self):
        
        self.stack.setCurrentWidget(self.page_video)
        
        self.widget_sub.hide()
        
        # self.h_layout_main.setStretch(0, 0)
        # self.h_layout_main.setStretch(1, 1)
        
    def show_webcam_mode(self):
        # self.stack.setCurrentWidget(self.page_webcam)
        
        self.widget_sub.show()
        
        # self.h_layout_main.setStretch(0, 2)
        # self.h_layout_main.setStretch(1, 5)
        
    def update_scoretable(self, stage_result, row:int):
        if stage_result is None:
            return
        # index + 동작명
        q_index = QTableWidgetItem(str(row+1))
        q_index.setTextAlignment(Qt.AlignCenter)
        result_name = QTableWidgetItem( stage_result.action_name)
        result_name.setTextAlignment(Qt.AlignCenter)
        self.widget_scoretable.setItem(row, 0, q_index)
        self.widget_scoretable.setItem(row, 1, result_name)
        
        # 정확도 = conf_mean, 평균값
        result_conf = QTableWidgetItem(f'{stage_result.conf_mean:.1f}')
        result_conf.setTextAlignment(Qt.AlignCenter)
        self.widget_scoretable.setItem(row, 2, result_conf)
        
        # 안정성 = detect_ratio, 쓰레쓰홀드 이상의 프레임 갯수비율
        result_ratio = QTableWidgetItem(f'{stage_result.detect_ratio:.1f}')
        result_ratio.setTextAlignment(Qt.AlignCenter)
        self.widget_scoretable.setItem(row, 3, result_ratio)
        
        # 평균 = total_score, = 안정성 *0.35 + 평균*0.5 + 맥스값*0.15 
        item_score = QTableWidgetItem(f'{stage_result.total_score:.1f}')
        item_score.setTextAlignment(Qt.AlignCenter)
        self.widget_scoretable.setItem(row, 4, item_score)
        
        print(f'[Debug][update_scoretable] row: {row}, 업데이트: {stage_result.action_name} -> {stage_result.total_score:.1f}점]')
    
            
    def update_rank_table(self):
        
        # score table에 있는 값을 여기서 직접 최종 마무리짓고 rank table에 넣기
        # score table에 최종 행 넣기########################
        accu_score1 = self.widget_scoretable.item(0, 2)
        accu_score1 = float(accu_score1.text())
        accu_score2 = self.widget_scoretable.item(1, 2)
        accu_score2 = float(accu_score2.text())
        accu_score3 = self.widget_scoretable.item(2, 2)
        accu_score3 = float(accu_score3.text())

        accu_final = QTableWidgetItem(f'{(accu_score1+accu_score2+accu_score3)/3:.1f}')
        accu_final.setTextAlignment(Qt.AlignCenter)
        
        safe_score1 = self.widget_scoretable.item(0, 3)
        safe_score1 = float(safe_score1.text())
        safe_score2 = self.widget_scoretable.item(1, 3)
        safe_score2 = float(safe_score2.text())
        safe_score3 = self.widget_scoretable.item(2, 3)
        safe_score3 = float(safe_score3.text())
        
        safe_final = QTableWidgetItem(f'{(safe_score1+safe_score2+safe_score3)/3:.1f}')
        safe_final.setTextAlignment(Qt.AlignCenter)
        
        total_score1 = self.widget_scoretable.item(0, 4)
        total_score1 = float(total_score1.text())
        total_score2 = self.widget_scoretable.item(1, 4)
        total_score2 = float(total_score2.text())
        total_score3 = self.widget_scoretable.item(2, 4)
        total_score3 = float(total_score3.text())
        
        total_final = QTableWidgetItem(f'{(total_score1+total_score2+total_score3)/3:.1f}')        
        total_final.setTextAlignment(Qt.AlignCenter)
        
        result_name = QTableWidgetItem('최종')
        result_name.setTextAlignment(Qt.AlignCenter)
        
        self.widget_scoretable.setItem(3, 1, result_name)
        self.widget_scoretable.setItem(3, 2, accu_final)
        self.widget_scoretable.setItem(3, 3, safe_final)
        self.widget_scoretable.setItem(3, 4, total_final)
        
        # score table 내용을 rank table에 업데이트하기#########        
        scores = []
        for row in range(self.widget_rank.rowCount()):
            item = self.widget_rank.item(row, 1)
            if item and item.text():
                try:
                    scores.append(float(item.text()))
                except ValueError:
                    pass
        total_final = float(total_final.text())
        
        scores.append(total_final)
        scores.sort(reverse=True)
        
        # new_row_idx = None # 현재 점수 랭크
        # for idx in range(len(scores)-1, -1, -1):
        #     if scores[idx] == total_final:
        #         new_row_idx = idx
        #         break
        for idx in range(len(scores)):
            if scores[idx] == total_final:
                new_row_idx = idx
                break

        self.widget_rank.setSortingEnabled(False)
        self.widget_rank.insertRow(new_row_idx)

        # 이전 하이라이트 복구
        if self._highlighted_rank_row is not None:
            prev = self._highlighted_rank_row
            actual_prev = prev if prev < new_row_idx else prev + 1
            for col in range(self.widget_rank.columnCount()):
                item = self.widget_rank.item(actual_prev, col)
                if item:
                    bg = self.COLOR_ODD_BG if actual_prev % 2 == 0 else self.COLOR_EVEN_BG
                    item.setBackground(QColor(bg))
                    item.setForeground(QColor(self.COLOR_TEXT)) 
        
        # 순위
        rank_item = QTableWidgetItem(str(new_row_idx + 1))
        rank_item.setTextAlignment(Qt.AlignCenter)
            
        # 점수
        score_item = QTableWidgetItem(f'{total_final:.1f}')
        score_item.setTextAlignment(Qt.AlignCenter)
        
        for item in (rank_item, score_item):
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QColor(self.COLOR_NEW_BG))
            item.setForeground(QColor(self.COLOR_NEW_TEXT))
        
        self.widget_rank.setItem(new_row_idx, 0, rank_item)
        self.widget_rank.setItem(new_row_idx, 1, score_item)

        prev_score = None
        prev_rank = 0
        for row in range(self.widget_rank.rowCount()):
            rank_item  = self.widget_rank.item(row, 0)
            score_item = self.widget_rank.item(row, 1)
            if score_item and score_item.text() and rank_item:
                score = float(score_item.text())
                if score != prev_score:
                    prev_rank = row + 1
                    prev_score = score
                rank_item.setText(str(prev_rank))
        
        # for row in range(new_row_idx + 1, self.widget_rank.rowCount()):
        #     old_rank = self.widget_rank.item(row, 0)
        #     if old_rank and old_rank.text():
        #         old_rank.setText(str(int(old_rank.text()) + 1))
        
        self._highlighted_rank_row = new_row_idx
        
        self.widget_rank.scrollToItem(self.widget_rank.item(new_row_idx, 0))
        rank_display = new_row_idx +1
        print(f'[Debug][update_rank_table] 현재 {rank_display}위입니다.')
            
        self.widget_scoretext.setText(f'축하합니다! 성공하셨습니다!\
                                      \n총점 : {total_final:.1f}점 | 현재 {rank_display}위 입니다.')
        self.rank = rank_display
        print(f'[Debug][update_rank_table] 랭킹 테이블 업데이트, {rank_display}위,  점수: {total_final:.1f}]')
        
    @Slot(dict)
    def update_controller(self, result_dict):
        
        # frame = result_dict["frame"]
        detections = result_dict["detections"]
        # action_results = result_dict.get("action_results", {})
        keypoints_dict = result_dict.get("keypoints", {})
        
        self.handle_roi(detections)
                
        self.cleanup_old_ids(time.time())
        
        if len(self.id_last_seen) > 50:
            print(f"[ROI] tracking ids : {len(self.id_last_seen)}")
        
        # if self.current_view_mode == 'video':
        #     self.show_video_mode()
            
        # elif self.current_view_mode == 'webcam':
        #     self.show_webcam_mode()
        #     self.canvas.update_frame(frame, detections, self.roi, self.active_user_id, webcam_mode=True)
        # print(f'[Debug] Active user ID: {self.active_user_id}, Current view mode: {self.current_view_mode}, IDs in ROI: {[x["id"] for x in detections if self.is_inside_roi(x["bbox"])]}')
        # print(f'[Debug] ROI enter times: {self.roi_enter_time}, ID last seen: {self.id_last_seen}')
        # print(f'[Debug] Detections: {detections}')
        # print(f'[Debug] Human exists: {human_exists}')
        
        # 사람이 roi 안에 일정시간 이상 존재하면 체험 스타트 시그널 발생
        
        # if self.! > self.roi_threshold_sec:
        #     if self.experience_start_check == 0:
        #         self.experience_start_check = time.time()
        #     duration = 4 - int(time.time() - self.experience_start_check)
        #     countdown = {'color' : 'red',
        #                  'time' : duration-1}
        #     if duration < 0:
        #         self.human_time = 0
        #         self.experience_start_check = 0
        # else :
        #     countdown = None
        

        if self.active_user_id:
            # print(f'[Debug][update_controller] active user id: {self.active_user_id}, human time: {self.human_time}')
                          
            
            if self.current_view_mode == 'webcam':
                self.mode_1()
                if self.audio.play_sound("res/Sound/hello.mp3"):
                    print('[Debug][update_controller] hello.mp3 음성 출력 완료')
                self.current_view_mode = 'mode2_1'
                    
                    
            elif self.current_view_mode == 'mode2_1':
                if self.stage == 0:
                    if self.audio.get_current() == 'hello.mp3':
                        # print(f'[Debug][update_controller_0] {self.audio.get_current()}')
                        pass
                        
                    elif self.audio.get_current() == None:
                        self.audio.play_sound('res/Sound/action_1.mp3')
                        print('[Debug][update_controller_1] try to play action_1.mp3]')
                        print(f'[Debug][update_controller_1] audio current: {self.audio.get_current()}]')
                        if self.audio.get_current() == 'res/Sound/action_1.mp3': # 당연한 조건이라서 지울 예정
                            self.mode_2_1()
                            self.stage = 1 # 액션음악플레이중
                            print('[Debug][update_controller_1] action_1.mp3 음성 출력 완료')
                elif self.stage == 1:
                    if self.audio.get_current() == 'res/Sound/action_1.mp3':
                        # print(f'[Debug][update_controller_2] {self.audio.get_current()}')
                        pass
                    elif self.audio.get_current() == None:
                        # self.countdown_fun(10,1)
                        self.countdown_fun(10,3)
                        # print(f'[Debug][update_controller_3] count_down?')
                        if self.countdown_off:
                            self.countdown = None
                            self.stage = 2 # 동작카운트
                elif self.stage == 2: # 동작인식모드
                    # self.countdown_fun(10,1)
                    self.countdown_fun(10,5)
                    self.action_flag = 1
                    self.action_recogniton = True
                    if self.action_recogniton:
                        
                        """
                        if self._eval_stage != 0:
                            self._eval_stage = 0
                            self.exp_evaluator.start_stage(0)
                        """ 
                            
                        action_results = result_dict.get("action_results", {}).get(self.active_user_id, [])
                        ksp = result_dict.get("keypoints_scores_pair")
                        keypoints_dict = ksp[0] if ksp is not None else None
                        # keypoints_dict = result_dict.get("keypoints_scores_pair").get(self.active_user_id, [])[0] # id가 체험자가 맞는지 지금은 모르는 상태
                        self.exp_evaluator.add_frame(self.STAGE_CLASS_IDXS[self.action_flag-1], action_results, keypoints=keypoints_dict)
                    # print(f'[Debug][update_controller_4] do it !!!')
                    if self.countdown_off:
                        
                        stage_result = self.exp_evaluator.end_stage(self.STAGE_CLASS_IDXS[self.action_flag-1])
                        self.update_scoretable(stage_result, row = 0)
                        
                        """
                        if self._eval_stage == 0:
                            stage_result = self.exp_evaluator.end_stage()
                            self._eval_stage = -1
                            self.update_scoretable(stage_result, row = 0)
                        """
                         
                        self.current_view_mode = 'mode2_2'
                        self.stage = 0
                        # self.countdown_off = False
                        self.countdown = None
                        self.action_recogniton = False
                        self.audio.play_sound("res/Sound/action_2.mp3")
                        self.action_flag = 2
                        self.mode_2_2()
                        print(f'[Debug][update_controller_5] mode change to {self.current_view_mode}')
                        print('[Debug][update_controller_5] action_2.mp3 음성 출력 완료')
            
            elif self.current_view_mode == 'mode2_2':
                if self.stage == 0:
                    if self.audio.get_current() == 'res/Sound/action_2.mp3':
                        # print(f'[Debug][update_controller_6] {self.audio.get_current()}')
                        pass
                    elif self.audio.get_current() == None:
                        # self.countdown_fun(10,1)
                        self.countdown_fun(10,3)
                        # print(f'[Debug][update_controller_7] count_down?')
                        if self.countdown_off:
                            self.countdown = None
                            # print(f'[Debug][update_controller_8] {self.audio.get_current()}')
                            self.stage=2
                elif self.stage == 2:
                    # self.countdown_fun(10, 1)
                    self.countdown_fun(10, 5)
                    self.action_recogniton = True
                    
                    """
                    if self.action_recogniton:
                        if self._eval_stage != 0:
                            self._eval_stage = 0
                            self.exp_evaluator.start_stage(1)
                    """
                         
                    action_results = result_dict.get("action_results", {}).get(self.active_user_id, [])
                    ksp = result_dict.get("keypoints_scores_pair")
                    keypoints_dict = ksp[0] if ksp is not None else None
                    # keypoints_dict = result_dict.get("keypoints_scores_pair").get(self.active_user_id, [])[0]
                    self.exp_evaluator.add_frame(self.STAGE_CLASS_IDXS[self.action_flag-1], action_results, keypoints=keypoints_dict)
                    # print(f'[Debug][update_controller_9] do it !!!')
                    
                    if self.countdown_off:
                        
                        stage_result = self.exp_evaluator.end_stage(self.STAGE_CLASS_IDXS[self.action_flag-1])
                        self.update_scoretable(stage_result, row = 1)
                        
                        """
                        if self._eval_stage == 0:
                            stage_result = self.exp_evaluator.end_stage()
                            self._eval_stage = -1
                            self.update_scoretable(stage_result, row = 1)
                        """
                         
                        self.current_view_mode = 'mode2_3'
                        self.stage = 0
                        # self.countdown_off = False
                        self.countdown = None
                        self.action_recogniton = False
                        self.audio.play_sound("res/Sound/action_3.mp3")
                        self.action_flag = 3
                        self.mode_2_3()
                        
                        
                        print(f'[Debug][update_controller_10] mode change to {self.current_view_mode}')
                        print('[Debug][update_controller_10] action_3.mp3 음성 출력 완료')
                        
            elif self.current_view_mode == 'mode2_3':
                if self.stage == 0:
                    if self.audio.get_current() == 'res/Sound/action_3.mp3':
                        # print(f'[Debug][update_controller_11] {self.audio.get_current()}')
                        pass
                    elif self.audio.get_current() == None:
                        # self.countdown_fun(10, 1)
                        self.countdown_fun(10, 3)
                        # print(f'[Debug][update_controller_12] count_down?')
                        if self.countdown_off:
                            self.countdown = None
                            self.stage=2
                            # print(f'[Debug][update_controller_13] {self.audio.get_current()}')
                elif self.stage == 2:
                    # self.countdown_fun(10, 1)
                    self.countdown_fun(10, 5)
                    self.action_recogniton = True
                    
                    """
                    if self.action_recogniton:
                        if self._eval_stage != 0:
                            self._eval_stage = 0
                            self.exp_evaluator.start_stage(2)
                    """
                         
                    action_results = result_dict.get("action_results", {}).get(self.active_user_id, [])
                    ksp = result_dict.get("keypoints_scores_pair")
                    keypoints_dict = ksp[0] if ksp is not None else None
                    # keypoints_dict = result_dict.get("keypoints_scores_pair").get(self.active_user_id, [])[0]
                    self.exp_evaluator.add_frame(self.STAGE_CLASS_IDXS[self.action_flag-1], action_results, keypoints=keypoints_dict)
                    # print(f'[Debug][update_controller_14] do it !!!')
                    
                    if self.countdown_off:
                        
                        stage_result = self.exp_evaluator.end_stage(self.STAGE_CLASS_IDXS[self.action_flag-1])
                        self.update_scoretable(stage_result, row = 2)
                        
                        """
                        if self._eval_stage == 0:
                            stage_result = self.exp_evaluator.end_stage()
                            self._eval_stage = -1
                            self.update_scoretable(stage_result, row = 2)
                        """
                         
                        self.current_view_mode = 'mode_3'
                        self.stage=1
                        # self.countdown_off = False
                        self.countdown = None
                        self.action_recogniton = False
                        self.action_flag = 0
                        self.mode_3()
                        
                        print(f'[Debug][update_controller_15] mode change to {self.current_view_mode}')
                        print('[Debug][update_controller_15] 음성 더이상 없음')
        
            elif self.current_view_mode == 'mode_3':
                if self.stage == 1:
                    print('평가모듈 실행중!!!!!!!!!!!')
                    self.update_rank_table()
                    self.exp_evaluator.reset()
                    print('[Debug][update_controller_16] score table updated')
                    print('[Debug][update_controller_16] rank table updated')
                    self.stage = 0
                # 평가모듈을 실행하여 그 결과를 table에 넣고, rank table에도 자동으로 업데이트
                
                
                
                
                
                # if self.human_time == 0:
                #     now = time.time()
                #     stay_time = now - self.roi_enter_time[self.active_user_id]
                #     self.human_time = stay_time
                    
                #     self.countdown_fun(stay_time)
                    # text = "안녕하세요. 파이썬 음성 테스트입니다."
                    # tts = gTTS(text=text, lang='ko')
                    # tts.save("hello.mp3")
                
        if self.current_view_mode == 'webcam':
            self.canvas.set_visualization_mode(show_bbox=True, show_pose=False, show_action=False)
            
        elif self.current_view_mode in ('mode2_1', 'mode2_2', 'mode2_3', 'mode_3'):
            self.canvas.set_visualization_mode(show_bbox=True, show_pose=True, show_action=False)
            
        if self.action_recogniton:
            self.canvas.set_visualization_mode(show_bbox=True, show_pose=True, show_action=True)
        
        result_dict['current_view_mode']=self.current_view_mode
        
        # self.canvas.update_frame(frame, self.countdown)
        self.canvas.update_frame(result_dict, self.countdown)
            
        view_mode = { 'mode' : self.current_view_mode , 'action_recognition' : self.action_recogniton}
        
        self.signalviewinfo.emit(view_mode)
        
       

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MW(app=app)
    window.showFullScreen()
    sys.exit(app.exec())

