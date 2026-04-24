from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

class ClockPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # clock container
        self.setFixedSize(72, 33)
        
        # self.clock_font = QFont("Pretendard")
        # self.clock_font.setPixelSize(13)
        # self.clock_font.setLetterSpacing(QFont.PercentageSpacing, 112)
        
        # self.clock_time = QLabel()
        # self.clock_time.setText('00:00:00')
        # self.clock_time.setFont(self.clock_font)
        
        self.parent = parent
        font = QFont(self.parent.fontFamilies2)
        
        # label time
        self.clock_time = QLabel()
        self.clock_time.setFont(font)
        self.clock_time.setText('00:00:00')
        self.clock_time.setStyleSheet('font-size: 13px; color: #7E898F; letter-spacing: 2px;')
        self.clock_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # label region
        self.region_name = QLabel()
        self.region_name.setText('Seoul')
        self.region_name.setStyleSheet('font-family: Pretendard; font-size: 13px; color: #D3D3D3; letter-spacing: 2px;')
        self.region_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # layout date & time
        self.layout_clock = QVBoxLayout(self)
        self.layout_clock.setContentsMargins(0, 0, 0, 0)
        self.layout_clock.addWidget(self.clock_time,1)
        self.layout_clock.addWidget(self.region_name,1)
        
        # timer
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)
        
        self.timer.start()
        
        self.timeout() # 초기 시간 설정

    def timeout(self):
        now = datetime.now()
        self.clock_time.setText(f'{now.strftime("%H:%M:%S")}')