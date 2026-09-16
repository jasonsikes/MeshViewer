# MeshGLWidget.py
# Created by: Jason Sikes
#

from OpenGL.GL import *
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class MeshGLWidget(QOpenGLWidget):
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.12, 0.12, 0.14, 1.0)
        # self._log_gl_context()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def resizeGL(self, width, height):
        glViewport(0, 0, max(1, width), max(1, height))

    def _log_gl_context(self):
        vendor = _gl_string(glGetString(GL_VENDOR))
        renderer = _gl_string(glGetString(GL_RENDERER))
        version = _gl_string(glGetString(GL_VERSION))
        print(f"OpenGL vendor:   {vendor}")
        print(f"OpenGL renderer: {renderer}")
        print(f"OpenGL version:  {version}")


def _gl_string(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
