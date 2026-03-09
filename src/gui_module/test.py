from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QPushButton, QLabel

class StackedExample(QWidget):
    def __init__(self):
        super().__init__()

        self.stack = QStackedWidget()

        
        self.page1 = QWidget()
        layout1 = QVBoxLayout()
        self.label1 = QLabel("첫 번째 페이지")
        self.btn1 = QPushButton("2페이지로 이동")
        self.btn1.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout1.addWidget(self.label1)
        layout1.addWidget(self.btn1)
        self.page1.setLayout(layout1)
        
        
        self.page2 = QWidget()
        layout2 = QVBoxLayout()
        self.label2 = QLabel("두 번째 페이지")
        self.btn2 = QPushButton("1페이지로 이동")
        self.btn2.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout2.addWidget(self.label2)
        layout2.addWidget(self.btn2)
        self.page2.setLayout(layout2)

        
        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)

        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

        self.setWindowTitle("QStackedWidget 예제")
        self.resize(300, 200)

if __name__ == "__main__":
    app = QApplication([])
    window = StackedExample()
    window.show()
    app.exec()
        