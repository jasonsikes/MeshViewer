# MainWindow.py
# Created by: Jason Sikes
#

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
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
        self._mesh_group = QActionGroup(self)
        self._mesh_group.setExclusive(True)

        file_menu = self.menuBar().addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        bunny_menu = self.menuBar().addMenu("&Bunny")
        self._add_mesh_action(bunny_menu, "&Low-Res Bunny",
                              "bunny", "bunny_centroid")
        self._add_mesh_action(bunny_menu, "Subdivided &1 Iteration",
                              "subdivided_bunny", "bunny_centroid")
        self._add_mesh_action(bunny_menu, "Subdivided &2 Iterations",
                              "subdivided_bunny2", "bunny_centroid")

        tetra_menu = self.menuBar().addMenu("&Tetrahedron")
        self._add_mesh_action(tetra_menu, "&Tetrahedron",
                              "tetrahedron", "tetrahedron_centroid",
                              checked=True)
        self._add_mesh_action(tetra_menu, "Subdivided &1 Iteration",
                              "subdivided_tetrahedron", "tetrahedron_centroid")
        self._add_mesh_action(tetra_menu, "Subdivided &2 Iterations",
                              "subdivided_tetrahedron2", "tetrahedron_centroid")
        self._add_mesh_action(tetra_menu, "Subdivided &3 Iterations",
                              "subdivided_tetrahedron3", "tetrahedron_centroid")
        self._add_mesh_action(tetra_menu, "Subdivided &4 Iterations",
                              "subdivided_tetrahedron4", "tetrahedron_centroid")

        cube_menu = self.menuBar().addMenu("&Cube")
        self._add_mesh_action(cube_menu, "&Triangulated Cube",
                              "tri_cube", "cube_centroid")
        self._add_mesh_action(cube_menu, "Subdivided &1 Iteration",
                              "subdivided_tri_cube", "cube_centroid")
        self._add_mesh_action(cube_menu, "Subdivided &2 Iterations",
                              "subdivided_tri_cube2", "cube_centroid")
        self._add_mesh_action(cube_menu, "Subdivided &3 Iterations",
                              "subdivided_tri_cube3", "cube_centroid")

        display_menu = self.menuBar().addMenu("&Display")
        self._add_display_toggle(display_menu, "&Shading", "shade")
        self._add_display_toggle(display_menu, "&Cull Backfaces", "cull")
        self._add_display_toggle(display_menu, "&Vertex Annotation", "annotate")
        self._add_display_toggle(display_menu, "S&mooth Shading", "smooth")
        self._add_display_toggle(display_menu, "&Texture", "texture")

    def _add_mesh_action(self, menu, title, mesh_attr, centroid_attr, checked=False):
        action = QAction(title, self)
        action.setCheckable(True)
        action.setChecked(checked)
        self._mesh_group.addAction(action)
        action.triggered.connect(
            lambda checked, m=mesh_attr, c=centroid_attr:
                self._on_mesh_triggered(checked, m, c))
        menu.addAction(action)

    def _on_mesh_triggered(self, checked, mesh_attr, centroid_attr):
        if not checked:
            return
        mesh = getattr(self.gl_widget, mesh_attr)
        centroid = getattr(self.gl_widget, centroid_attr)
        self.gl_widget.select_mesh(mesh, centroid)

    def _add_display_toggle(self, menu, title, option_name):
        action = QAction(title, self)
        action.setCheckable(True)
        action.setChecked(getattr(self.gl_widget, option_name))
        action.toggled.connect(
            lambda enabled, name=option_name:
                self.gl_widget.set_display_option(name, enabled))
        menu.addAction(action)
