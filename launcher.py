import os
os.environ["PYTHONUTF8"] = "1"
os.environ["QT_MEDIA_BACKEND"] = "ffmpeg"
# os.environ["QT_FFMPEG_HW_DEC"] = "qsv"
os.environ["QT_FFMPEG_HW_DEC"] = "dxva2"
import sys
# sys.path.insert(0, '/mnt/hdd1/003_SOLUTION/industrial_safety/mmaction2')
import time
import platform
os_info = platform.system()
if os_info == 'Windows':
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)


from datetime import datetime
from pathlib import Path

# from main import MW
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

def show_splash():
    SPLASH_DIR = get_resource_dir()
    splash_img = os.path.join(SPLASH_DIR, 'res/logo_white.png')

    pixmap = QPixmap(splash_img)

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    return splash

def init_splash_worker(splash):
    worker = SplashWorker()
    worker.progress.connect(lambda msg: splash.showMessage(msg, Qt.AlignBottom | Qt.AlignCenter, Qt.white))
    worker.start()

    return worker

# def check_datetime():
#     strNow = datetime.now().strftime('%Y%m')
#     print('strNow: "{0}"'.format(strNow))
#     if strNow != '202512':
#         print('[{0}] license is expired.'.format(strNow))
#         sys.exit()
#     else:
#         pass

class SplashWorker(QThread):
    progress = Signal(str)
    stop_flag = False

    def run(self):
        msg_list = [
            "Initializing modules...",
            "Loading resources...",
            "Preparing UI...",
            "Almost ready..."
        ]

        i = 0
        while not self.stop_flag:
            self.progress.emit(msg_list[i % len(msg_list)])
            i += 1
            time.sleep(0.4) 

def init_finish(worker, splash):
    worker.stop_flag = True
    worker.wait()
    splash.finish(window)

if __name__ == '__main__':
    app = QApplication(sys.argv)


    splash = show_splash()
    worker = init_splash_worker(splash=splash)

    # check_datetime()

    from main import MW

    window = MW(app=app)

    init_finish(worker=worker, splash=splash)
    
    window.showFullScreen()
    
    sys.exit(app.exec())