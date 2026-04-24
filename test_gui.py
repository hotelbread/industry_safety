import sys
import os
import time

# 현재 디렉토리를 경로에 추가 (혹시 모를 경로 에러 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from src.gui_module.main_window import MainWindow

from src.module.ai_thread import AiThread


def main():
    # 1. Qt Application 인스턴스 생성
    app = QApplication(sys.argv)
    
    # 폰트 등록
    font_id = QFontDatabase.addApplicationFont("res/fonts/public/static/Pretendard-Bold.otf")
    if font_id == -1:
        print("Error loading font!")
    else:
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        print(f'Loaded font: {font_family}')
        
    # 2. 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.showFullScreen()
    
    # --- 테스트 시뮬레이션 시작 ---
    test_scores = [90.0, 99.0, 10.0, 50.0, 70.1,70.0, 30.0, 71.0, 70.0, 20.0 ,99.0, 100.0]
    index = 0

    def run_test():
        nonlocal index
        if index < len(test_scores):
            score = test_scores[index]
            print(f"Testing score: {score}")
            # MainWindow 내부의 rank_board 객체 이름이 'rank_board'인지 확인하세요.
            window.RankBoard.update_score(score) 
            index += 1
        else:
            test_timer.stop()
            print("Test finished")

    # 5초마다 run_test 함수 실행
    test_timer = QTimer()
    test_timer.timeout.connect(run_test)
    test_timer.start(5000) # 5000ms = 5초
    # ----------------------------
    
    # 3. 이벤트 루프 시작
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()