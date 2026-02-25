import cv2
import threading

from src.misc.duration import Duration

# np.set_printoptions(suppress=True)

########################################
#              ThreadAi                #
########################################
class VideoThread(threading.Thread):
    def __init__(self, mode, video_path=None, parent=None):
        super().__init__()

        self.parent = parent
        if self.parent is not None:
            self.canvas_width  = self.parent.CANVAS_WIDTH
            self.canvas_height  = self.parent.CANVAS_HEIGHT
            self.canvas_ratio = self.canvas_width / self.canvas_height

        self.mode = mode
        self.video_path = video_path

        self.frame = None
        self.lock = threading.Lock()
        self._running = True

        self.line_width = 3
        self.circle_radius = 5

        resolutions = [
            # (640, 480, 2, 3),
            (1280, 720, 3, 5),
            # (1920, 1080, 5, 7),
            # (2560, 1440, 7, 10),
        ]

        if self.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
            for w, h, lw, cr in resolutions:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                set_res_result = self.set_res_success_check(w, h, self.cap)
                if set_res_result:
                    self.line_width = lw
                    self.circle_radius = cr
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        else :
            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        if self.cap.isOpened():
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.delay_limit = 1 / self.video_fps

            print()
            self.parent.log(f'[Thread][Video] FPS : {self.video_fps}')
            self.parent.log(f'[Thread][Video] Delay limit (sec) : {self.delay_limit:.2f}')
            self.parent.log(f'[Thread][Video] Frame count : {self.total_frame_count}')
            self.parent.log(f'[Thread][Video] Size : ({self.frame_width}x{self.frame_height})')
            print()

    def run(self):
        if self.cap.isOpened():
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self.parent.log(f'[Thread][Video][Error] Fail to read frame')
                    break
                else:
                    with self.lock:
                        self.frame = frame

        self.cap.release()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    
    def set_res_success_check(self, target_w, target_h, cap) -> bool:
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if target_w == actual_w and target_h == actual_h:
            return True
        else:
            return False
        
    def stop(self):
        self._running = False
        self.cap.release()