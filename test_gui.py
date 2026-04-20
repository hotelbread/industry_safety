import sys
import os

# 현재 디렉토리를 경로에 추가 (혹시 모를 경로 에러 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from src.gui_module.main_window import MainWindow


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
    
    # 3. 이벤트 루프 시작
    sys.exit(app.exec())

if __name__ == "__main__":
    main()