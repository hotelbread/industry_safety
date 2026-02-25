import time

from src.misc.duration import Duration

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        self.canvas_timer = Duration('canvas')

        # self.painter = QPainter(self)
    
    def update_frame(self, q_image):
        self.current_image = q_image
        self.update() 
    
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

