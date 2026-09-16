# MainWindow.py
# Created by: Jason Sikes
#

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow

from MeshGLWidget import MeshGLWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Winged-Edge Mesh in Python")
        self.resize(700, 700)
        self.move(100, 100)

        self.gl_widget = MeshGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self._create_menus()

    def _create_menus(self):
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Q"))
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(quit_action)
