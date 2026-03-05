import cv2
import threading
import time

class VideoPlaybackThread(threading.Thread):

    def __init__(self, video_path, parent=None):
        super().__init__()
        self.parent = parent
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        self.lock = threading.Lock()
        self.frame = None
        self._running = True
        self._play = True  # 화면에 보여줄지 여부

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = 1 / self.fps if self.fps > 0 else 0.03

    def run(self):
        while self._running:
            if not self._play:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret:
                # 영상 끝나면 처음으로
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with self.lock:
                self.frame = frame

            time.sleep(self.delay)

        self.cap.release()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def set_play(self, enable: bool):
        self._play = enable

    def stop(self):
        self._running = False