from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame, QPushButton
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon

class ResetButton(QPushButton):
    """리셋 버튼만 담당하는 클래스"""
    def __init__(self, parent=None):
        super().__init__("  RESET EXPERIENCE", parent)
        # self.setFixedHeight(57)
        self.setFixedSize(250,57)
        
        # 원형아이콘
        self.setIcon(QIcon("res/UI_File/reset_button_icon.png"))
        self.setIconSize(QSize(12,12))
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #FFA400;
                color: #000000;
                font-family: 'Pretendard';
                font-size: 16px;
                font-weight: 700;
                border-radius: 12px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #FFCA8B;
                color: #F97316;
            }
            QPushButton:pressed {
                background-color: #F97316;
                color: #000000;
            }
        """)
        
        #border-bottom-right-radius: 12px;

class RankBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.setFixedSize(250, 969) # 창 높이에 맞춰 조정
        self.setObjectName("RankCard")
        self.setStyleSheet("QWidget#RankCard { background-color: transparent;}")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 제목 (TOP PERFORMERS)
        self.lbl_title = QLabel("TOP PERFORMERS")
        # self.lbl_title.setFixedSize(158,17)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("color: #7E898F; font-size: 16px; font-weight: 700; margin-bottom: 22px; letter-spacing: 0.5px")
        self.main_layout.addWidget(self.lbl_title)
        
        # 랭킹 리스트들을 담을 컨테이너
        self.rank_card = QFrame()
        self.rank_card.setFixedSize(250, 741)
        self.rank_card.setObjectName("RankListCard")
        self.rank_card.setStyleSheet("""
            QFrame#RankListCard {
                background-color: #161616;
                border-radius: 12px;
                border: 1px solid #212121;
            }
        """)
        self.card_layout = QVBoxLayout(self.rank_card)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(0)

        # 고정 영역 (1, 2, 3위 메달권)
        self.fixed_area = QWidget()
        # self.fixed_area.setFixedHeight(50)
        self.fixed_layout = QVBoxLayout(self.fixed_area)
        self.fixed_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_layout.setSpacing(0)
        self.card_layout.addWidget(self.fixed_area)

        # 스크롤 영역 (4위 이하)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        # self.scroll.setFixedHeight(50) # 행 하나의 높이 고정을 원해
        self.scroll.setFrameShape(QFrame.NoFrame) # 테두리 제거
        self.scroll.setStyleSheet("background: transparent;")
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 스크롤바 숨기기
        
        self.scroll_content = QWidget()
        # self.scroll_content.setFixedSize(245, 1200) # 높이는 자동조절, 너비 고정
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(0)
        self.scroll.setWidget(self.scroll_content)
        # self.scroll_layout.addStretch()
        
        self.card_layout.addWidget(self.scroll)

        self.main_layout.addWidget(self.rank_card)
        
        # 4. 리셋 버튼 (맨 아래 고정)
        self.main_layout.addStretch(1)
        self.btn_reset = ResetButton()
        # self.btn_reset.clicked.connect(self.reset)
        self.main_layout.addWidget(self.btn_reset, alignment=Qt.AlignCenter)

        self.db_scores = [98.5, 97.2, 95.0, 85.0, 82.3, 78.1, 75.0, 72.5, 70.0, 68.2, 65.0, 62.1, 60.0, 58.3, 55.0, 52.1, 50.0, 48.2, 45.0, 42.1]
        # self.display_scores = ["-"]*20
        # 데이터 초기화 예시
        self.init_ranks()

    def init_ranks(self):
        # 1~3위 추가
        for i in range(3):
            item = RankItem(i, self.db_scores[i], is_medal=True)
            self.fixed_layout.addWidget(item)

        # 4~20위 추가 (테스트용)
        for i in range(3, 20):
            item = RankItem(i, self.db_scores[i])
            self.scroll_layout.insertWidget(self.scroll_layout.count(), item)

    def refresh_ui(self, highlight_val=None):

        # 레이아웃 비우기 로직
        self.clear_layouts()

        highlight_bool = True
                            
        for i, score in enumerate(self.db_scores):
            rank = i
            is_medal = rank < 3
            item = RankItem(rank, score, is_medal)
            
            # 레이아웃 배치
            if is_medal:
                self.fixed_layout.addWidget(item)
            else:
                self.scroll_layout.insertWidget(self.scroll_layout.count(), item)

            # 내가 방금 낸 점수라면 하이라이트 실행
            if highlight_val is not None and score == highlight_val and highlight_bool:
                item.set_highlight(True)
                highlight_bool = False
                if rank > 3:
                    QTimer.singleShot(200, lambda i=item: self.scroll.ensureWidgetVisible(i))
                    # self.scroll.ensureWidgetVisible(item) # 4위 이하면 자동 스크롤
                    # print(f"Debug: Scrolled to Rank {item.rank + 1} with score {item.lbl_score.text()}")
        
        if len(self.db_scores) < 15:
            for j in range(len(self.db_scores), 15) :
                rank = j
                scores = '-'
                is_medal = rank < 3
                item = RankItem(rank, scores, is_medal)
                if is_medal:
                    self.fixed_layout.addWidget(item)
                else:
                    self.scroll_layout.insertWidget(self.scroll_layout.count(), item)
        
                
    def clear_layouts(self):
        """모든 랭크 아이템 제거"""
        # 고정 영역 클리어
        while self.fixed_layout.count():
            child = self.fixed_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # 스크롤 영역 클리어
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    
    def refresh_ui_reset(self):
        self.clear_layouts()
        
        for i in range(20):
            rank = i
            is_medal = rank <= 3
            # 점수 대신 "-" 문자열을 전달
            item = RankItem(rank, "-", is_medal)
            
            if is_medal:
                self.fixed_layout.addWidget(item)
            else:
                self.scroll_layout.insertWidget(self.scroll_layout.count(), item)
    
    def update_score(self, new_score):
        """새 점수가 들어오면 등수 계산 후 하이라이트"""
        print(f"[Debug] 새 점수 입력: {new_score}")
        # 1. 새 점수 추가 (동점자 처리를 위해 append)
        self.db_scores.append(new_score)
        
        # 2. 정렬 (나중에 들어온 동점자가 위로 오게 처리)
        self.db_scores.sort(reverse=True)
        
        # 3. UI 새로고침
        self.refresh_ui(highlight_val=new_score)
        
    def reset(self):
        """랭킹 데이터 초기화"""
        self.db_scores = []
        self.refresh_ui_reset()
        print("랭킹이 초기화되었습니다.")

class RankItem(QFrame):
    def __init__(self, rank, score, is_medal=False, parent=None):
        super().__init__()
        self.setFixedSize(240,57)
        self.setObjectName("RankItem")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(14, 0, 14, 0)
        
            # QLabel {font-family: 'Pretendard'; font-size: 30px; color: #FFFFFF; font-weight: 700; background-color: transparent; }
        # 기본 스타일: 하단 구분선만 표시
        self.default_style = """
            QFrame#RankItem { border-bottom: 1px solid #36393B; background-color: #161616; }
            QLabel#rank_label {  background-color : #28323F; color: #FFFFFF; font-family: 'Pretendard'; font-size: 12px; font-weight: 700; border-radius : 3px;}
            QLabel#score_label {  background-color : transparent; color: #FFFFFF; font-family: 'Pretendard'; font-size: 16px; font-weight: 700;}
        """
        # 하이라이트 스타일: 보내주신 border-image 적용
        self.highlight_style = """
            QFrame#RankItem { 
                border-image: url(res/UI_File/Rank_Checkbar.png) 0 0 0 0 stretch stretch;
                border: none;
            }
            QLabel#rank_label {  background-color : #28323F; color: #FFFFFF; font-family: 'Pretendard'; font-size: 12px; font-weight: 700; border-radius : 3px;}
            QLabel#score_label { background-color : transparent; color: #FFA400; font-family: 'Pretendard'; font-size: 16px; font-weight: 700;}
        """
        self.setStyleSheet(self.default_style)

        
        self.rank = rank
        self.is_medal = is_medal
        
        
        # if self.rank <= 2:  # 1~3위는 메달 표시
        #     self.setFixedSize(240, 57)
        #     item = 
            
        # for i in range(3):
        #     item = RankItem(i, self.db_scores[i], is_medal=True)
        #     self.fixed_layout.addWidget(item)

        # # 4~20위 추가 (테스트용)
        # for i in range(3, 20):
        #     item = RankItem(i, self.db_scores[i])
        #     self.scroll_layout.insertWidget(self.scroll_layout.count(), item)
        
        
        

        # 순위 표시
        self.lbl_rank = QLabel(str(self.rank+1))
        self.lbl_rank.setObjectName("rank_label")
        self.lbl_rank.setFixedSize(20, 20)
        self.lbl_rank.setAlignment(Qt.AlignCenter)
        

        
        # 점수 + 메달아이콘
        self.score_container = QWidget()
        self.score_layout = QHBoxLayout(self.score_container)
        self.score_layout.setContentsMargins(0, 0, 0, 0)
        self.score_layout.setSpacing(8)
        
        if self.is_medal:
            # 메달 아이콘 추가 (1~3위)
            self.medal_icon = QLabel()
            self.medal_icon.setFixedSize(20, 20)
            if self.rank == 0:
                self.medal_icon.setStyleSheet("background: url(res/UI_File/Medal_gold.png) no-repeat center;")
            elif rank == 1:
                self.medal_icon.setStyleSheet("background: url(res/UI_File/Medal_silver.png) no-repeat center;")
            elif rank == 2:
                self.medal_icon.setStyleSheet("background: url(res/UI_File/Medal_bronze.png) no-repeat center;")
            self.score_layout.addWidget(self.medal_icon)
        
        self.lbl_score = QLabel(f"{score:.1f}" if isinstance(score, float) else str(score))
        self.lbl_score.setObjectName("score_label")
        self.lbl_score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # self.lbl_score.setStyleSheet("font-size: 16px; font-weight: 700")
        self.score_layout.addWidget(self.lbl_score)

        self.layout.addWidget(self.lbl_rank)
        self.layout.addStretch()
        # self.layout.addWidget(self.lbl_score)
        self.layout.addWidget(self.score_container)

        # self.set_highlight(False) # 초기 상태는 일반 모드

    def set_highlight(self, active):
        self.setStyleSheet(self.highlight_style if active else self.default_style)