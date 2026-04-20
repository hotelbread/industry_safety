from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar 
from PySide6.QtCore import Qt, QTimer

class ProgressbarPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. 레이아웃 설정
        # self.layout = QVBoxLayout(self)
        self.setFixedSize(385, 400)
        self.max_count = 3
        self.current_count = self.max_count
        self.current_color = "whited" # 초기 색상 (예시)
        
        # 2. 안내 문구 (Title)
        self.lbl_status = QLabel("GET READY", self)
        self.lbl_status.setAlignment(Qt.AlignLeft)
        self.lbl_status.setStyleSheet("font-family: Pretendard; font-size: 36px; font-weight: 700; color: #FFA400; font-style: italic;")
        self.lbl_status.setFixedWidth(355)
        self.lbl_status.setFixedHeight(38)
        self.lbl_status.move(0, 0) # 위치 조정 (InstructionPanel 내에서)
        
        # 3. 카운트다운 숫자
        self.lbl_timer = QLabel("3", self)
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        self.lbl_timer.setStyleSheet("font-family: Pretendard; font-size: 128px; color: #1D1D1D; font-style: italic;")
        self.lbl_timer.setFixedWidth(96)
        self.lbl_timer.setFixedHeight(109)
        self.lbl_timer.move(4, 71) # 위치 조정 (InstructionPanel 내에서)
        
        # 4. 프로그래스바 안내문구
        time_text = self.lbl_timer.text()
        self.lbl_progress = QLabel(self)
        self.lbl_progress.setText(f"SESSION STARTS IN <span style='color: white;'>{time_text}</span> SECONDS")
        self.lbl_progress.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        # self.lbl_progress.setAlignment(Qt.AlignLeft)
        self.lbl_progress.setStyleSheet("font-family: Pretendard; font-size: 16px; color: #7E898F;")
        self.lbl_progress.setFixedWidth(355)
        self.lbl_progress.setFixedHeight(143)
        self.lbl_progress.move(4, 87) # 위치 조정 (InstructionPanel 내에서)
        
        # 5. 프로그레스 바
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1000)
        self.progress.setValue(99) # 초기값 (예시)
        self.progress.setTextVisible(False) # 숫자 제거
        self.progress.setStyleSheet("""
            QProgressBar { 
                background-color: #333333;
                border: 1px solid #333333;
                border-radius: 5px; 
            }
            QProgressBar::chunk {
                background-color: #FFFFFF;
                border-radius: 3px;          /* 게이지 양쪽 둥글게 */
                margin: 1px;       
        }""")
        self.progress.setFixedWidth(360)
        self.progress.setFixedHeight(11)
        self.progress.move(4, 187)
        
        self.count_timer = QTimer(self) # 숫자용 (1초)
        self.count_timer.timeout.connect(self.update_number)
        
        self.gauge_timer = QTimer(self) # 게이지용 (0.05초)
        self.gauge_timer.timeout.connect(self.update_gauge)

        self.max_seconds = 3
        self.elapsed_ms = 0 # 경과 시간 (밀리초 단위)
        # self.count_timer = QTimer(self)
        # self.count_timer.timeout.connect(self.update_countdown)
        
        # 레이아웃에 위젯 추가
        # self.layout.addWidget(self.lbl_status)
        # self.layout.addWidget(self.lbl_timer)
        # self.layout.addWidget(self.lbl_progress)
        # self.layout.addWidget(self.progress)

    # 외부(Main)에서 텍스트나 상태를 변경할 때 쓸 함수들
    def set_status(self, text):
        self.lbl_status.setText(text)

    def set_timer(self, count):
        self.lbl_timer.setText(str(count))


    def update_progress_style(self, color):
        self.progress.setStyleSheet(f"""
            QProgressBar {{ 
                background-color: #333333;
                border: 1px solid #333333;
                border-radius: 5px; 
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;          /* 게이지 양쪽 둥글게 */
                margin: 1px;       
        }}""")
        
    def start_timer(self, seconds=3, color="white", message_type="start"):
        self.max_seconds = seconds
        self.elapsed_ms = 0
        self.update_progress_style(color)

        # 메시지 유형에 따라 다른 텍스트 설정
        if message_type == "start":
            self.lbl_progress.setText(f"SESSION STARTS IN <span style='color: {color};'>{seconds}</span> SECONDS")
        elif message_type == "motion":
            self.lbl_progress.setText(f"EACH MOTION : <span style='color: {color};'>{seconds}</span> SECONDS")

        self.lbl_timer.setText(str(seconds))
        # self.update_ui()
        self.count_timer.start(1000)
        self.gauge_timer.start(50) # 0.05초마다 업데이트 (초당 20번)

    def update_number(self):
        """1초마다 숫자 업데이트"""
        current_num = int(self.lbl_timer.text())
        if current_num > 0:
            self.lbl_timer.setText(str(current_num - 1))
            # self.lbl_progress.setText(f"SESSION STARTS IN <span style='color: {self.current_color};'>{current_num - 1}</span> SECONDS")
        else:
            self.count_timer.stop()

    def update_gauge(self):
        """0.05초마다 게이지 업데이트"""
        self.elapsed_ms += 50
        total_ms = self.max_seconds * 1000
        
        if self.elapsed_ms <= total_ms:
            # 비율 계산 (0 ~ 1000)
            val = int((self.elapsed_ms / total_ms) * 1000)
            self.progress.setValue(val)
        else:
            self.gauge_timer.stop()
