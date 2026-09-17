# MeshGLWidget.py
# Created by: Jason Sikes
#

import copy
import os
from math import atan2, cos, pi, sin, tan

from numpy import array, ascontiguousarray, cross, empty, float32, identity, uint8, zeros
from numpy.linalg import norm
from OpenGL.GL import *
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from Bunny import Bunny
from Mesh import Mesh

TEXTURE_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "block_texture.png")
TEXTURE_ENCODING = GL_RGBA

AMBIENT = (0.6, 0.6, 0.6, 1)
DIFFUSE = (0.5, 0.5, 0.5, 1)
SPECULAR = (0.5, 0.5, 0.5, 1)
SHININESS = 51.2

LIGHT_MODEL_AMBIENT = (0.1, 0.1, 0.1, 1)
LIGHT0_POSITION = (30, 30, 30, 0)
LIGHT0_AMBIENT = (0.3, 0.3, 0.3, 1)
LIGHT0_DIFFUSE = (1, 1, 1, 1)
LIGHT0_SPECULAR = (1, 1, 1, 1)

ORBIT_RADIANS_PER_PIXEL = pi / 180
ORBIT_PHI_EPSILON = 1e-7
ZOOM_IN_FACTOR = 0.95


def calculate_texture_coordinates(vertices, indices):
    ssi = 0
    tsi1 = 1
    tsi2 = 2

    retval = empty((len(indices), 3, 2), dtype=float32)
    centroid = zeros(3)
    for vert in vertices:
        centroid += vert
    centroid /= len(vertices)

    maxssi = max(vert[ssi] for vert in vertices) * 1.01
    minssi = min(vert[ssi] for vert in vertices) * 1.01

    for f in range(len(indices)):
        is_in_q1 = False
        is_in_q4 = False
        for v in range(len(indices[f])):
            vertex = vertices[indices[f][v]]
            retval[f, v, 0] = (vertex[ssi] - minssi) / (maxssi - minssi)
            retval[f, v, 1] = (atan2(vertex[tsi1] - centroid[tsi1],
                                     vertex[tsi2] - centroid[tsi2])
                               / pi / 2.0 + 0.5)
            if retval[f, v, 1] < 0.25:
                is_in_q1 = True
            if retval[f, v, 1] > 0.75:
                is_in_q4 = True

        if is_in_q1 and is_in_q4:
            for v in range(len(indices[f])):
                if retval[f, v, 1] < 0.25:
                    retval[f, v, 1] += 1
    return retval


def spherical_to_cartesian(r, theta, phi):
    return array([r * cos(theta) * sin(phi),
                  r * sin(theta) * sin(phi),
                  r * cos(phi)])


def _as_gl_matrix(matrix):
    return ascontiguousarray(matrix, dtype=float32).flatten("F")


def look_at_matrix(eye, center, up):
    eye = array(eye, dtype=float)
    center = array(center, dtype=float)
    up = array(up, dtype=float)
    forward = center - eye
    forward = forward / norm(forward)
    side = cross(forward, up)
    side = side / norm(side)
    upn = cross(side, forward)
    matrix = identity(4, dtype=float)
    matrix[0, 0:3] = side
    matrix[1, 0:3] = upn
    matrix[2, 0:3] = -forward
    matrix[0, 3] = -side.dot(eye)
    matrix[1, 3] = -upn.dot(eye)
    matrix[2, 3] = forward.dot(eye)
    return matrix


def perspective_matrix(fovy_degrees, aspect, z_near, z_far):
    f = 1.0 / tan(fovy_degrees * pi / 360.0)
    matrix = zeros((4, 4), dtype=float)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (z_far + z_near) / (z_near - z_far)
    matrix[3, 2] = -1.0
    matrix[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    return matrix


def project_to_window(object_xyz, modelview, projection, viewport):
    clip = projection @ modelview @ array(
        [object_xyz[0], object_xyz[1], object_xyz[2], 1.0], dtype=float)
    if clip[3] <= 0:
        return None
    ndc = clip[0:3] / clip[3]
    win_x = viewport[0] + viewport[2] * (ndc[0] + 1.0) * 0.5
    win_y = viewport[1] + viewport[3] * (ndc[1] + 1.0) * 0.5
    win_z = (ndc[2] + 1.0) * 0.5
    return win_x, win_y, win_z


def get_centroid(mesh):
    centroid = zeros(3)
    for vert in mesh.verts:
        centroid += vert.position
    centroid /= len(mesh.verts)
    return centroid


class MeshGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.shade = False
        self.cull = False
        self.annotate = False
        self.smooth = False
        self.texture = False
        self._buffers_dirty = False

        self.eye_radius = 2.5
        self.eye_phi = pi / 4
        self.eye_theta = pi / 4
        self.lookat = array([0.0, 0.0, 0.0])
        self.up = array([0.0, 0.0, 1.0])
        self._orbiting = False
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._view_matrix = identity(4, dtype=float)
        self._proj_matrix = identity(4, dtype=float)
        self._viewport = (0, 0, 1, 1)

        self.vertices_buffer_id = 0
        self.smooth_normals_buffer_id = 0
        self.flat_normals_buffer_id = 0
        self.texture_buffer_id = 0
        self.texture_id = 0
        self._label_texture_id = 0

        self.bunny = None
        self.subdivided_bunny = None
        self.subdivided_bunny2 = None
        self.tetrahedron = None
        self.subdivided_tetrahedron = None
        self.subdivided_tetrahedron2 = None
        self.subdivided_tetrahedron3 = None
        self.subdivided_tetrahedron4 = None
        self.tri_cube = None
        self.subdivided_tri_cube = None
        self.subdivided_tri_cube2 = None
        self.subdivided_tri_cube3 = None
        self.tetrahedron_centroid = None
        self.cube_centroid = None
        self.bunny_centroid = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.12, 0.12, 0.14, 1.0)
        # self._log_gl_context()

        (self.vertices_buffer_id,
         self.smooth_normals_buffer_id,
         self.flat_normals_buffer_id,
         self.texture_buffer_id) = glGenBuffers(4)

        self._create_meshes()
        self.tetrahedron_centroid = get_centroid(self.tetrahedron)
        self.cube_centroid = get_centroid(self.tri_cube)
        self.bunny_centroid = get_centroid(self.bunny)
        self.mesh = self.tetrahedron
        self._upload_mesh()

        self.set_lookat(self.tetrahedron_centroid)
        self._set_projection(self.width(), self.height())
        self._init_lighting()
        self._init_texture()
        self._label_texture_id = glGenTextures(1)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glShadeModel(GL_SMOOTH)
        glCullFace(GL_BACK)
        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDisable(GL_LIGHTING)

    def paintGL(self):
        if self._buffers_dirty:
            self._upload_mesh()
            self._buffers_dirty = False

        self._restore_fixed_function()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_COLOR_MATERIAL)
        self._set_projection(self._viewport[2], self._viewport[3])
        self._apply_view()
        self._init_lighting()
        glColor3f(1, 1, 1)

        if self.shade:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)

        if self.cull:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer_id)
        glVertexPointer(3, GL_FLOAT, 0, None)

        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self.texture_buffer_id)
            glTexCoordPointer(2, GL_FLOAT, 0, None)
        else:
            glDisable(GL_TEXTURE_2D)

        if self.smooth:
            glBindBuffer(GL_ARRAY_BUFFER, self.smooth_normals_buffer_id)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.flat_normals_buffer_id)
        glNormalPointer(GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, len(self.mesh.vboVertices) // 3)
        if self.annotate:
            self._draw_vertex_labels()

    def resizeGL(self, width, height):
        self._viewport = (0, 0, max(1, width), max(1, height))
        glViewport(0, 0, self._viewport[2], self._viewport[3])
        self._set_projection(width, height)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._orbiting = True
            self._mouse_x = event.position().x()
            self._mouse_y = event.position().y()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._orbiting = False
        event.accept()

    def mouseMoveEvent(self, event):
        if not self._orbiting:
            return
        x = event.position().x()
        y = event.position().y()
        self.eye_theta -= (x - self._mouse_x) * ORBIT_RADIANS_PER_PIXEL
        self.eye_phi -= (y - self._mouse_y) * ORBIT_RADIANS_PER_PIXEL
        if self.eye_phi >= pi:
            self.eye_phi = pi - ORBIT_PHI_EPSILON
        elif self.eye_phi <= 0:
            self.eye_phi = ORBIT_PHI_EPSILON
        self._mouse_x = x
        self._mouse_y = y
        self.update()
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return
        steps = delta / 120.0
        self.eye_radius *= ZOOM_IN_FACTOR ** steps
        self.update()
        event.accept()

    def _create_meshes(self):
        bunny_tex_coords = calculate_texture_coordinates(
            Bunny.bunnyVertices, Bunny.bunnyIndices)
        self.bunny = Mesh(Bunny.bunnyVertices, Bunny.bunnyIndices, bunny_tex_coords)

        new_vertices = self.bunny.copyOfVertices()
        new_indices = self.bunny.copyOfIndices()
        new_coords = calculate_texture_coordinates(new_vertices, new_indices)
        self.subdivided_bunny = Mesh(new_vertices, new_indices, new_coords)
        self.subdivided_bunny.butterflySubdivide()

        new_vertices = self.subdivided_bunny.copyOfVertices()
        new_indices = self.subdivided_bunny.copyOfIndices()
        new_coords = calculate_texture_coordinates(new_vertices, new_indices)
        self.subdivided_bunny2 = Mesh(new_vertices, new_indices, new_coords)
        self.subdivided_bunny2.butterflySubdivide()

        self.tetrahedron = Mesh.Tetrahedron(1)

        self.subdivided_tetrahedron = copy.deepcopy(self.tetrahedron)
        self.subdivided_tetrahedron.butterflySubdivide()
        self.subdivided_tetrahedron2 = copy.deepcopy(self.subdivided_tetrahedron)
        self.subdivided_tetrahedron2.butterflySubdivide()
        self.subdivided_tetrahedron3 = copy.deepcopy(self.subdivided_tetrahedron2)
        self.subdivided_tetrahedron3.butterflySubdivide()
        self.subdivided_tetrahedron4 = copy.deepcopy(self.subdivided_tetrahedron3)
        self.subdivided_tetrahedron4.butterflySubdivide()

        self.tri_cube = Mesh.Cube(1)
        self.tri_cube.triangulate()
        self.subdivided_tri_cube = copy.deepcopy(self.tri_cube)
        self.subdivided_tri_cube.butterflySubdivide()
        self.subdivided_tri_cube2 = copy.deepcopy(self.subdivided_tri_cube)
        self.subdivided_tri_cube2.butterflySubdivide()
        self.subdivided_tri_cube3 = copy.deepcopy(self.subdivided_tri_cube2)
        self.subdivided_tri_cube3.butterflySubdivide()

    def select_mesh(self, mesh, centroid):
        self.mesh = mesh
        self._buffers_dirty = True
        self.set_lookat(centroid)
        self.update()

    def set_display_option(self, name, enabled):
        setattr(self, name, enabled)
        self.update()

    def _upload_mesh(self):
        mesh = self.mesh
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboVertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.flat_normals_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboFlatNormals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.smooth_normals_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboSmoothNormals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboTexCoords, GL_STATIC_DRAW)

    def set_lookat(self, centroid):
        if centroid is not None:
            self.lookat = centroid

    def _apply_view(self):
        eye = spherical_to_cartesian(self.eye_radius, self.eye_theta, self.eye_phi)
        eye = eye + self.lookat
        self._view_matrix = look_at_matrix(eye, self.lookat, self.up)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(_as_gl_matrix(self._view_matrix))

    def _set_projection(self, width, height):
        aspect = max(1, width) / max(1, height)
        self._proj_matrix = perspective_matrix(40.0, aspect, 0.1, 30.0)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(_as_gl_matrix(self._proj_matrix))
        glMatrixMode(GL_MODELVIEW)

    def _restore_fixed_function(self):
        if bool(glUseProgram):
            glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_vertex_labels(self):
        fb_w = max(1, self._viewport[2])
        fb_h = max(1, self._viewport[3])
        image = QImage(fb_w, fb_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        font = QFont("Courier")
        font.setPixelSize(max(13, int(13 * self.devicePixelRatioF())))
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.white)
        for vertex in self.mesh.verts:
            if vertex is None:
                continue
            projected = project_to_window(
                vertex.position, self._view_matrix, self._proj_matrix,
                (0, 0, fb_w, fb_h))
            if projected is None:
                continue
            win_x, win_y, win_z = projected
            if win_z < 0.0 or win_z > 1.0:
                continue
            painter.drawText(int(round(win_x)), int(round(fb_h - win_y)),
                             "v%i" % vertex.index)
        painter.end()
        image = image.mirrored(False, True)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._label_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fb_w, fb_h, 0,
                     GL_BGRA, GL_UNSIGNED_BYTE, image.constBits())
        glColor4f(1, 1, 1, 1)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)

    def _init_lighting(self):
        glDisable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LIGHT_MODEL_AMBIENT)
        glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT0_AMBIENT)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT0_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_SPECULAR, LIGHT0_SPECULAR)
        glEnable(GL_LIGHT0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, AMBIENT)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR, SPECULAR)
        glMaterialfv(GL_FRONT, GL_SHININESS, SHININESS)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, LIGHT0_POSITION)
        glPopMatrix()

    def _init_texture(self):
        img = Image.open(TEXTURE_FILENAME)
        img_data = array(list(img.getdata()), uint8)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, TEXTURE_ENCODING, img.width, img.height,
                     0, TEXTURE_ENCODING, GL_UNSIGNED_BYTE, img_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glDisable(GL_TEXTURE_2D)

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
