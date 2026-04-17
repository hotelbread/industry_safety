import os

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtCore import Qt

from src.gui_module.instruction_panel import InstructionPanel
from src.gui_module.canvas import Canvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Signal Game")
        
        self.setMinimumSize(1280, 720) 
        
                # 배경 이미지 로드 (한 번만 로드해서 재사용)
        self.bg_pixmap = QPixmap("res/UI_File/background.png")
       
        # 메인 컨테이너 (여기에 레이아웃이 올라감)
        self.widget_main = QWidget()
        self.setCentralWidget(self.widget_main)
        
        # 전체 레이아웃 (H)
        self.main_layout = QHBoxLayout(self.widget_main)
        self.main_layout.setContentsMargins(30, 30, 30, 30) # 외곽 여백
        self.main_layout.setSpacing(20) # 위젯 간 간격

        # 3단 분할 레이아웃
        self.first_layout = QVBoxLayout()
        self.second_layout = QVBoxLayout()
        self.third_layout = QVBoxLayout()
        
        # 모듈 객체 생성
        self.instruction_panel = InstructionPanel()
        self.canvas = Canvas()
        
        # 위젯 배치
        self.first_layout.addWidget(self.instruction_panel)
        self.first_layout.addStretch() # 아래로 밀리지 않게 여백 추가

        self.second_layout.addWidget(self.canvas)
        
        # 레이아웃 조립 (비율 설정 중요!)
        self.main_layout.addLayout(self.first_layout, 1)  # 좌측
        self.main_layout.addLayout(self.second_layout, 4) # 중앙 (카메라 크게)
        self.main_layout.addLayout(self.third_layout, 1)  # 우측

    def paintEvent(self, event):
        """배경 이미지를 비율에 맞춰 그리기 (스타일시트보다 안정적)"""
        painter = QPainter(self)
        if not self.bg_pixmap.isNull():
            # 이미지를 창 크기에 맞춰 꽉 채움 (비율 유지)
            scaled_bg = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled_bg)
       
        # self.widget_main = QWidget()
        # self.widget_main.setObjectName("main")
        # self.setCentralWidget(self.widget_main)
        
        # self.widget_main.setStyleSheet("""
        #     QWidget#main {
        #         background-image: url(res/UI_File/background.png);
        #         background-repeat: no-repeat;
        #         background-size: cover;
        #     }
        # """)
        # # 크게 3개의 레이아웃으로 (horizontal안에 3개의 vertical)
        # self.main_layout = QHBoxLayout(self.widget_main)
        # self.first_layout = QVBoxLayout()
        # self.second_layout = QVBoxLayout()
        # self.third_layout = QVBoxLayout()
        
        # self.instruction_panel = InstructionPanel()
        # self.canvas = Canvas()
        
        # self.first_layout.addWidget(self.instruction_panel)
        # self.second_layout.addWidget(self.canvas)
        
        # self.main_layout.addLayout(self.first_layout, 1)  # 좌측 1
        # self.main_layout.addLayout(self.second_layout, 3) # 중앙 3 (더 넓게)
        # self.main_layout.addLayout(self.third_layout, 1)  # 우측 1