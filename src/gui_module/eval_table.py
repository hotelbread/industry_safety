from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

class EvaluationTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.setFixedSize(365, 325) # 디자인 가이드에 맞춘 사이즈
        self.setObjectName("EvaluationCard")
        self.setStyleSheet("""
            QWidget#EvaluationCard {
                background-color: #161616; 
                border-radius: 12px;
            }
        """)
        # 1. 메인 레이아웃 및 제목
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.lbl_title = QLabel("HAND SIGNAL EVALUATION")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("""
            font-family: Pretendard; 
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            background: #161616;
            color: #7E898F; 
            font-size: 16px; 
            font-weight: 700; 
            letter-spacing: 0.5px;""")
        self.lbl_title.setFixedHeight(60) # 제목 높이 조정
        self.layout.addWidget(self.lbl_title)

        # 2. 테이블 생성
        self.table = QTableWidget(4, 4) # 4행 4열
        self.setup_table()
        self.layout.addWidget(self.table)

    def setup_table(self):
        # 헤더 텍스트 설정
        self.table.setHorizontalHeaderLabels(["Motion", "Acc", "Stab", "Avg"])
        
        # 기본 설정: 수정 불가, 선택 불가, 테두리 제거
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(True) # 격자선 표시
        
        # 헤더 사이즈 조절: 균등하게 배분
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setFixedHeight(42) # 헤더 높이 조정
        header.setStyleSheet("QHeaderView::section { background-color: #2A2A2B; color: #7E898F; font-family: Pretendard; font-size: 16px; font-weight: 700; border: none; border-bottom: none; padding: 1px; }")
        self.table.verticalHeader().setVisible(False) # 왼쪽 행 번호 숨기기
        self.table.setRowHeight(0, 53) # 첫 번째 행 높이 조정
        self.table.setRowHeight(1, 53) # 두 번째 행 높이 조정
        self.table.setRowHeight(2, 53) # 세 번째 행 높이 조정
        self.table.setRowHeight(3, 53) # 네 번째 행 높이 조정

        # 데이터 초기화 (시안 내용 반영)
        data = [
            ["붐 올리기", "-", "-", "-"],
            ["권상", "-", "-", "-"],
            ["비상 정지", "-", "-", "-"],
            ["최종", "-", "-", "-"]
        ]

        for r, row_data in enumerate(data):
            for c, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                
                # 색상 입히기 (시안과 동일하게)
                if c == 1: 
                    item.setForeground(QColor("#F97316")) # 오렌지 (Acc)
                elif c == 2: 
                    item.setForeground(QColor("#4789C8")) # 블루 (Stab)
                elif c == 3 or c == 0: 
                    item.setForeground(QColor("#FFFFFF")) # 흰색 (Avg, Motion)
                
                self.table.setItem(r, c, item)

        # 스타일시트 적용 (다크 테마 + 격자선 색상)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #161616;
                font-size: 17px;
                border: none;
                gridline-color: #36393B; /* 격자선 색상 */
                font-family: 'Pretendard';
                /* 하단 라운드 처리를 위해 추가 */
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
        """)

    # 외부에서 데이터를 실시간 업데이트할 함수
    def update_data(self, row, acc, stab, avg):
        self.table.item(row, 1).setText(f"{acc}%")
        self.table.item(row, 2).setText(f"{stab}%")
        self.table.item(row, 3).setText(f"{avg}")
        # 색상 다시 흰색 계열로 변경
        self.table.item(row, 1).setForeground(QColor("#F97316"))
        self.table.item(row, 2).setForeground(QColor("#4789C8"))
        self.table.item(row, 3).setForeground(QColor("#FFFFFF"))
        
        # 평균값이 80 이상이면 녹색, 미만이면 빨간색으로 변경
        if avg >= 80:
            for r in range(1, 4):
                self.table.item(row, r).setForeground(QColor("#4789C8")) # 
        else:
            for r in range(1, 4):
                self.table.item(row, r).setForeground(QColor("#F97316")) # 붉은색
                