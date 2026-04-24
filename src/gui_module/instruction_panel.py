from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class InstructionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 폰트
        self.parent = parent
        font = QFont(self.parent.fontFamilies2)
        
        # 디자인 가이드에 맞춘 고정 사이즈
        self.setFixedSize(365, 322)
        time_text= "5" # 초기값 (예시)
        # 1. 배경 카드 스타일 (반투명 검정)
        self.container = QFrame(self)
        self.container.setFixedSize(365, 322)
        self.container.setStyleSheet("""
            QFrame {
                border-image: url(res/UI_File/Current_instruction.png) 0 0 0 0 stretch stretch;
                border-radius: 10px;
            }
            QLabel {
                background: transparent;
            }
        """)
        
        # 2. 내부 레이아웃
        # self.layout = QVBoxLayout(self.container)
        # self.layout.setContentsMargins(10, 15, 10, 10)
        # self.layout.setSpacing(10)

        # 3. 상단 소제목 (CURRENT INSTRUCTION)
        self.lbl_head = QLabel("CURRENT INSTRUCTION", self)
        self.lbl_head.setFont(font)
        self.lbl_head.setStyleSheet("""
            color: #7E898F; 
            font-size: 16px; 
            font-weight: 600;
            background: transparent;
        """)
        self.lbl_head.move(11, 16) # 위치 조정 (InstructionPanel 내에서)
        
        # 4. 메인 안내 문구
        self.lbl_content_1 = QLabel("첫 동작이 곧 시작됩니다.", self) 
        self.lbl_content_1.setWordWrap(True) # 자동 줄바꿈
        # self.lbl_content_1.setFixedHeight(100)
        self.lbl_content_1.setFixedSize(340, 100) # 임시
        self.lbl_content_1.setStyleSheet("""
            color: white; 
            font-family: Pretendard; 
            font-size: 24px;
            background: transparent;
            font-weight: 700;
            line-height: 1000; /* 줄 간격 조정 */
        """)
        self.lbl_content_1.move(11, 34) # 위치 조정 (InstructionPanel 내에서)
        
        self.lbl_content_2 = QLabel(self)
        self.lbl_content_2.setText(f"안내에 따라 해당 동작들을 <br> <span style='color: white;'>{time_text}</span>초 동안 유지하세요.")
        self.lbl_content_2.setStyleSheet("""
            color: #7E898F; 
            font-family: Pretendard; 
            font-size: 24px; 
            font-weight: 700;
            background: transparent;
            line-height: 1.5; /* 줄 간격 조정 */
        """)
        self.lbl_content_2.move(11,118)

        # self.layout.addWidget(self.lbl_head)
        # self.layout.addWidget(self.lbl_content_1)
        # self.layout.addWidget(self.lbl_content_2)
        # self.layout.addStretch()

    # 외부에서 안내 문구를 바꿀 때 사용하는 함수
    def set_instruction(self, state, rank=None):
        if state == "0":
            text_1 = "첫 동작이 곧 시작됩니다."
            text_2 = "안내에 따라 해당 동작들을 <br> <span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "1":
            text_1 = "1단계<br> <span style='color: #FFA400;'>\"붐 올리기\"</span>동작을 취해보세요."
            text_2 = "<span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "2":
            text_1 = "동작이 곧 시작됩니다."
            text_2 = "안내에 따라 해당 동작을 <br> <span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "3":
            text_1 = '2단계<br> <span style="color: #FFA400;">\"권상\"</span>동작을 취해보세요.'
            text_2 = "<span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "4":
            text_1 = "동작이 곧 시작됩니다."
            text_2 = "안내에 따라 해당 동작을 <br> <span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "5":
            text_1 = '3단계<br> <span style="color: #FFA400;">\"비상 정지\"</span>동작을 취해보세요.'
            text_2 = "<span style='color: white;'>5</span>초 동안 유지하세요."
        elif state == "6":
            text_1 = f"축하합니다! 성공하셨습니다.<br>현재 <span style='color: white;'>{rank}위</span> 입니다."
            text_2 = ""
        elif state == "7":
            text_1 = f"아쉽네요. 실패하셨습니다.<br>현재 <span style='color: white;'>{rank}위</span> 입니다."
            text_2 = ""
        self.lbl_content_1.setText(text_1)
        self.lbl_content_2.setText(text_2)