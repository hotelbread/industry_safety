import time

from src.misc.duration import Duration

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPolygon

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
        self.roi_points = [(1,1),(1,480),(639, 480),(639, 1)]
        self.roi_polygon = None
        self.admin_mode = False

        self.canvas_timer = Duration('canvas')

        # self.scale_x = 1
        # self.scale_y = 1
                        
        # # scale & offset
        # self.scale = 1
        # self.offset_x = 0
        # self.offset_y = 0
        
        # self.painter = QPainter(self)
    
    def update_frame(self, q_image, countdown=None):
        self.current_image = q_image
        self.q_image_frame_width = self.current_image.width()
        self.q_image_frame_height = self.current_image.height()
        
        # frame size
        self.frame_width = self.width()
        self.frame_height = self.height()
        self.scale = min(self.frame_width/self.q_image_frame_width, self.frame_height/self.q_image_frame_height)
        self.offset_x = (self.frame_width - (self.q_image_frame_width * self.scale))/2
        self.offset_y = (self.frame_height - (self.q_image_frame_height * self.scale))/2
        
        # print(f'[Debug][Canvas] frame_width : {self.frame_width} x {self.frame_height} [scale: {self.scale}]')
        # print(f'[Debug][Canvas] current_image size: {self.current_image.width()} x {self.current_image.height()}')
        # print(f'[Debug][Canvas] offset : {self.offset_x} , {self.offset_y}')
        # if self.roi is not None and self.current_image is not None:
        #     self.current_image = self.paintRoi(self.current_image, self.roi)
        self.current_image = self.paintRoi(self.current_image, self.roi_points)
        if countdown is not None:
            self.countdown_draw(self.current_image, countdown)
        
        # print(f'[Debug][Canvas] Frame updated with ROI: {self.roi}')
        self.update()
    
    def mousePressEvent(self, event):
        if self.admin_mode:
            
            mx = event.pos().x()
            my = event.pos().y()
            
            # 영상 영역 안에서만 클릭 허용
            if (
                mx < self.offset_x
                or mx > self.offset_x + self.frame_width * self.scale
                or my < self.offset_y
                or my > self.offset_y + self.frame_height * self.scale
            ):
                return
            

            # canvas -> frame 변환
            frame_x = (mx - self.offset_x) / self.scale
            frame_y = (my - self.offset_y) / self.scale

            self.roi_points.append((frame_x, frame_y))
            
            print(f"Added ROI point: ({frame_x}, {frame_y})")
            self.update()
        else:
            print(f"Current points: {self.roi_points}")
    
    def paintRoi(self, image, roi_points):
        
        painter = QPainter(image)
        painter.setPen(QColor(255, 195, 0))
        if len(roi_points) > 0:
            if len(roi_points) >= 3:
                self.roi_polygon = QPolygon([QPoint(int(x), int(y)) for x, y in roi_points])
                # print(f'updated polygon created')
            if len(roi_points) >= 1:
                # if len(roi) == 2:
                #     temp_polygon = QPolygon([QPoint(x, y) for x, y in roi])
                #     painter.drawPoints(temp_polygon)
            
                for x, y in self.roi_points:
                    painter.drawEllipse(x-4, y-4, 8, 8)
            
            if self.roi_polygon:
                painter.drawPolygon(self.roi_polygon)
        else:
            self.roi_polygon = None
        painter.end()
            
        return image
    
    def countdown_draw(self, q_image, countdown):
        painter = QPainter(q_image)
        if countdown['color'] == 'green':
            painter.setPen(QColor(0, 255, 0))
        elif countdown['color'] == 'red':
            painter.setPen(QColor(255, 0, 0))
        elif countdown['color'] == 'blue':
            painter.setPen(QColor(0, 0, 255))

        text_dict = {"green":"Start!!!", "blue":"Done!!!", "red" : "Stop!!!"}

        font = painter.font()
        font.setPointSize(90)
        painter.setFont(font)
        if countdown['time'] < 0:
            text = text_dict[countdown['color']]
            # text = str('Start!!!')
        else : 
            text = str(countdown['time'])
        text_rect = painter.boundingRect(0, 0, 0, 0, Qt.AlignCenter, text)

        # 화면 중앙 계산
        center_x = q_image.width() // 2
        center_y = q_image.height() // 2

        # 텍스트 위치 조정 (폰트 크기 고려)
        text_x = center_x - text_rect.width() // 2
        text_y = center_y - text_rect.height() // 2

        painter.drawText(text_x, text_y, text)
        painter.end()
    
    def paintEvent(self, event):
        # self.canvas_timer.set_prev()

        painter = QPainter(self)

        # painter.fillRect(self.rect(), Qt.black)
        painter.fillRect(self.rect(), QColor("#2C303C"))

        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            widget_width = self.width()
            widget_height = self.height()
            
            img_ratio = img_width / img_height
            widget_ratio = widget_width / widget_height
            
            if img_ratio > widget_ratio:
                new_width = widget_width
                new_height = int(widget_width / img_ratio)
            else:
                new_height = widget_height
                new_width = int(widget_height * img_ratio)
            
            x = (widget_width - new_width) // 2
            y = (widget_height - new_height) // 2

            target_rect = QRect(x, y, new_width, new_height)
            painter.drawImage(target_rect, self.current_image)
        
        painter.end()

        # self.canvas_timer.calc_elapsed()
        # self.canvas_timer.print_fps()
        # self.canvas_timer.print_sec()
        # print()

    # def clearPaint(self):
    #     self.painter.fillRect(self.rect(), QColor("#2C303C"))

