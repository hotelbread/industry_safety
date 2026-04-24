import os

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QPainter, QPixmap, QFontDatabase
from PySide6.QtCore import Qt

from src.gui_module.progressbar_panel import ProgressbarPanel
from src.gui_module.instruction_panel import InstructionPanel
from src.gui_module.eval_table import EvaluationTable
from src.gui_module.canvas import Canvas
from src.gui_module.clock_panel import ClockPanel
# from src.gui_module.result_panel import ResultPanel
from src.gui_module.rank_board import RankBoard
import time

# # 폰트 등록
# font_id = QFontDatabase.addApplicationFont("res/fonts/public/static/Pretendard-Bold.otf")
# if font_id == -1:
#     print("Error loading font!")
# else:
#     font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
#     print(f'Loaded font: {font_family}')
    
class BackgroundWidget(QWidget):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            # 중앙 위젯의 크기에 맞춰 스케일링
            scaled_bg = self.pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            # 중앙 정렬 계산
            x = (self.width() - scaled_bg.width()) // 2
            y = (self.height() - scaled_bg.height()) // 2
            painter.drawPixmap(x, y, scaled_bg)
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Signal Game")
        
        self.setMinimumSize(1280, 720) 
        
        # 폰트
        self.PATH_FONT2 = './res/fonts/public/static/Pretendard-SemiBold.otf'
        fontId2 = QFontDatabase.addApplicationFont(self.PATH_FONT2)
        self.fontFamilies2 = QFontDatabase.applicationFontFamilies(fontId2)
        
        # 배경 이미지 로드 (한 번만 로드해서 재사용)
        self.bg_pixmap = QPixmap("res/UI_File/background.png")
       
        # 메인 컨테이너 (여기에 레이아웃이 올라감)
        self.widget_main = BackgroundWidget(self.bg_pixmap)
        self.setCentralWidget(self.widget_main)
        
        # 전체 레이아웃 (H)
        self.main_layout = QHBoxLayout(self.widget_main)
        self.main_layout.setContentsMargins(88, 70, 17, 31) # 외곽 여백
        self.main_layout.setSpacing(20) # 위젯 간 간격

        # 3단 분할 레이아웃
        self.first_layout = QVBoxLayout()
        self.first_container = QWidget()
        self.first_container.setFixedWidth(385) # 디자인 수치 입력
        self.first_container.setLayout(self.first_layout)
        self.first_layout.setSpacing(20) # 위젯 간 간격
        
        self.second_layout = QVBoxLayout()
        
        self.third_layout = QVBoxLayout()
        self.third_container = QWidget()
        self.third_container.setFixedWidth(270) # 디자인 수치 입력
        self.third_container.setLayout(self.third_layout)
        
        # 모듈 객체 생성
        self.progressbar_panel = ProgressbarPanel(self)
        self.instruction_panel = InstructionPanel(self)
        self.canvas = Canvas(self)
        # self.result_panel = ResultPanel()
        self.eval_table = EvaluationTable(self)
        self.RankBoard = RankBoard(self)
        
        # 시계만 따로
        self.ClockPanel = ClockPanel(self)
        self.ClockPanel.move(1659, 14)
        
        # 위젯 배치
        self.first_layout.addWidget(self.progressbar_panel, alignment=Qt.AlignTop) # 상단 정렬
        self.first_layout.addWidget(self.instruction_panel, alignment=Qt.AlignTop) # 상단 정렬
        self.first_layout.addWidget(self.eval_table, alignment=Qt.AlignTop) # 상단 정렬
        # self.first_layout.addStretch(1) # 아래로 밀리지 않게 여백 추가

        self.second_layout.addWidget(self.canvas)
        
        # self.third_layout.addWidget(self.result_panel)
        # self.third_layout.addWidget(self.ClockPanel, alignment=Qt.AlignTop) # 상단 정렬
        self.third_layout.addWidget(self.RankBoard, alignment=Qt.AlignTop) # 상단 정렬
        # self.third_layout.addStretch(1) # 아래로 밀리지 않게 여백 추가
        
        # 레이아웃 조립 (비율 설정 중요!)
        self.main_layout.addWidget(self.first_container)  # 좌측
        self.main_layout.addLayout(self.second_layout, 1) # 중앙 (카메라 크게)
        self.main_layout.addWidget(self.third_container)  # 우측

        # 프로그래스방 실험테스트
        # self.instruction_panel.start_timer(5, "#FFA400", message_type="motion") 
        # self.progressbar_panel.start_timer(3, "#FFFFFF", message_type="start") 
        # self.instruction_panel.set_instruction("1", rank=1)
        # self.instruction_panel.set_instruction("1")
