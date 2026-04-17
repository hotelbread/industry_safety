from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt

class InstructionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. 레이아웃 설정
        self.layout = QVBoxLayout(self)
        
        # 2. 안내 문구 (Title)
        self.lbl_status = QLabel("GET READY")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        
        # 3. 카운트다운 숫자
        self.lbl_timer = QLabel("3")
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        self.lbl_timer.setStyleSheet("font-size: 60px; color: yellow;")
        
        # 4. 프로그레스 바
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        
        # 레이아웃에 위젯 추가
        self.layout.addWidget(self.lbl_status)
        self.layout.addWidget(self.lbl_timer)
        self.layout.addWidget(self.progress)

    # 외부(Main)에서 텍스트나 상태를 변경할 때 쓸 함수들
    def set_status(self, text):
        self.lbl_status.setText(text)

    def set_timer(self, count):
        self.lbl_timer.setText(str(count))
