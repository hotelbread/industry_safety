# video_thread_safe.py
import av
import numpy as np
import threading
from time import sleep, perf_counter

from PySide6.QtCore import QThread, Signal

class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    progress_changed = Signal(int)
    loop_reset = Signal()

    def __init__(self, path):
        super().__init__()
        
        self.path = path
        self._running = True
        self._paused = False
        self._seek_request = None
        self._seek_lock = threading.Lock()
        self._seeking = False

        meta = av.open(self.path, options={'threads': 'auto'})
        self.stream_meta = meta.streams.video[0]
        self.video_fps = self.stream_meta.average_rate
        meta.close()

        self.container = None
        self.stream = None
        self.decoder = None

        self.current_frame_index = 0
        self.current_pts = 0
        self.prevTime = perf_counter()

    def _open_container(self):
        self.container = av.open(self.path, options={'threads': 'auto'})
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def _create_decoder(self):
        if self.container is None or self.stream is None:
            raise RuntimeError("Container/stream not initialized")
        self.decoder = self.container.decode(self.stream)

    def _frames_to_pts(self, frame_index: int) -> int:
        seconds = frame_index / (self.video_fps or 30.0)
        tb = float(self.stream.time_base)  # seconds per pts unit
        pts = int(seconds / tb)
        return pts

    def video_pause(self):
        self._paused = True

    def video_play(self):
        self._paused = False

    def video_stop(self):
        self._running = False
        self.quit()
        self.wait()

    def seek_to_frame(self, frame_idx):
        with self._seek_lock:
            self._seek_request = int(frame_idx)


    def forward_frames(self, frames=30):
        with self._seek_lock:
            cur = self.current_frame_index
            try:
                total = int(self.stream.frames)
                target = min(total - 1, cur + frames)
            except Exception:
                target = cur + frames
            self._seek_request = int(target)

    def backward_frames(self, frames=30):
        with self._seek_lock:
            cur = self.current_frame_index
            target = max(0, cur - frames)
            self._seek_request = int(target)

    def forward_seconds(self, seconds=1):
        frames = int(seconds * self.video_fps)
        self.forward_frames(frames)

    def backward_seconds(self, seconds=1):
        frames = int(seconds * self.video_fps)
        self.backward_frames(frames)

    def get_current_pts(self):
        return getattr(self, "current_pts", 0)

    def _execute_seek_internal(self, target_frame_index: int):
        self._seeking = True
        emitted = False

        try:
            MAX_RETRY = 5
            retry_count = 0

            while retry_count < MAX_RETRY and not emitted:
                try:
                    if self.container:
                        self.container.close()
                except:
                    pass
                self._open_container()
                target_pts = self._frames_to_pts(target_frame_index)

                try:
                    self.container.seek(target_pts, any_frame=False, backward=True, stream=self.stream)
                except:
                    self.container.seek(target_pts)

                self._create_decoder()

                target_seconds = target_frame_index / (self.video_fps or 30.0)
                consumed = 0
                max_consume = 300

                while True:
                    try:
                        f = next(self.decoder)
                    except StopIteration:
                        retry_count += 1
                        break
                    except Exception:
                        self._create_decoder()
                        continue

                    consumed += 1
                    pts_seconds = (float(f.pts) * float(self.stream.time_base)) if f.pts else None

                    if pts_seconds is None or pts_seconds >= target_seconds or consumed >= max_consume:
                        try:
                            arr = f.to_ndarray(format="rgb24")
                            self.current_frame_index = target_frame_index
                            self.current_pts = f.pts if f.pts else self.current_pts

                            self.frame_ready.emit(arr)
                            self.progress_changed.emit(self.current_frame_index)

                            emitted = True
                        except:
                            pass
                        break

            if not emitted:
                try:
                    self._open_container()
                    self._create_decoder()
                    f = next(self.decoder)
                    arr = f.to_ndarray(format="rgb24")
                    self.current_frame_index = 0
                    self.current_pts = f.pts if f.pts else 0
                    self.frame_ready.emit(arr)
                    self.progress_changed.emit(0)
                except:
                    pass
            self._create_decoder()

        finally:
            self._seeking = False


    def run(self):
        # initialize container/decoder inside thread
        try:
            self._open_container()
            self._create_decoder()
        except Exception as e:
            # fatal: notify and exit
            print("[VideoThread] failed to open:", e)
            return

        while self._running:
            # check seek request (thread-safe)
            if self._seek_request is not None:
                with self._seek_lock:
                    target = self._seek_request
                    self._seek_request = None
                # execute seek safely here
                self._execute_seek_internal(target)
                sleep(0.001)
                continue

            # pause handling
            if self._paused:
                sleep(0.03)
                continue

            # fetch next frame via next(decoder)
            try:
                frame = next(self.decoder)
            except StopIteration:
                # end-of-stream: stop or loop depending on your policy
                try:
                    self.container.close()
                except:
                    pass

                self.container = av.open(self.path)
                self.stream = self.container.streams.video[0]

                self.decoder = self.container.decode(self.stream)

                # 재생 위치 초기화
                self.current_pts = 0
                self.frame_index = 0
                self.current_frame_index= 0
                self.loop_reset.emit()
                continue

            except Exception as e:
                try:
                    self._create_decoder()
                    continue
                except Exception:
                    # unrecoverable
                    print("[VideoThread] decoder error:", e)
                    break
            elapsed = perf_counter() - self.prevTime
            delay = max(0.0, 1.0 / (self.video_fps or 30.0) - elapsed)
            if delay > 0:
                sleep(delay)
            self.prevTime = perf_counter()
            self.current_frame_index += 1
            self.current_pts = frame.pts if frame.pts is not None else self.current_pts

            if not self._seeking:
                self.progress_changed.emit(self.current_frame_index)

            try:
                arr = frame.to_ndarray(format="rgb24")
                self.frame_ready.emit(arr)
            except Exception:
                continue

        # cleanup
        try:
            if self.container is not None:
                self.container.close()
        except Exception:
            pass

    def get_total_frame_count(self):
        return self.stream_meta.frames
    
    def get_fps(self):
        return self.video_fps
