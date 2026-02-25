import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, QMutex, Slot

from src.ai_module.pose_estimater import PoseEstimator
from src.ai_module.person_detector import PersonDetector
from src.ai_module.action_recognizer import ActionRecognizer

from src.gui_module.canvas import Canvas
from src.module.video_thread import VideoThread
from src.module.inf_thread import InferenceThread
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
        self.init_ai_model()
        self.init_ui()
        self.setup_thread()

    

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



