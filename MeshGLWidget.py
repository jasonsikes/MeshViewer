# MeshGLWidget.py
# Created by: Jason Sikes
#

import copy
import os
from math import atan2, cos, pi, sin

from numpy import array, empty, float32, uint8, zeros
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective
from PIL import Image
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
        self.smooth = False
        self.texture = False

        self.eye_radius = 2.5
        self.eye_phi = pi / 4
        self.eye_theta = pi / 4
        self.lookat = array([0.0, 0.0, 0.0])
        self.up = array([0.0, 0.0, 1.0])

        self.vertices_buffer_id = 0
        self.smooth_normals_buffer_id = 0
        self.flat_normals_buffer_id = 0
        self.texture_buffer_id = 0
        self.texture_id = 0

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
        self._set_mesh(self.tetrahedron)
        self.tetrahedron_centroid = get_centroid(self.tetrahedron)
        self.cube_centroid = get_centroid(self.tri_cube)
        self.bunny_centroid = get_centroid(self.bunny)

        self._set_projection(self.width(), self.height())
        self._set_view(self.tetrahedron_centroid)
        self._init_lighting()
        self._init_texture()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glShadeModel(GL_SMOOTH)
        glCullFace(GL_BACK)
        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDisable(GL_LIGHTING)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
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

    def resizeGL(self, width, height):
        glViewport(0, 0, max(1, width), max(1, height))
        self._set_projection(width, height)

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

    def _set_mesh(self, mesh):
        self.mesh = mesh
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboVertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.flat_normals_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboFlatNormals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.smooth_normals_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboSmoothNormals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, mesh.vboTexCoords, GL_STATIC_DRAW)

    def _set_view(self, centroid):
        if centroid is not None:
            self.lookat = centroid
        eye = spherical_to_cartesian(self.eye_radius, self.eye_theta, self.eye_phi)
        eye = eye + self.lookat
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2],
                  self.lookat[0], self.lookat[1], self.lookat[2],
                  self.up[0], self.up[1], self.up[2])

    def _set_projection(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, max(1, width) / max(1, height), 0.1, 30.0)
        glMatrixMode(GL_MODELVIEW)

    def _init_lighting(self):
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LIGHT_MODEL_AMBIENT)
        glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT0_AMBIENT)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT0_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_SPECULAR, LIGHT0_SPECULAR)
        glLightfv(GL_LIGHT0, GL_POSITION, LIGHT0_POSITION)
        glEnable(GL_LIGHT0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, AMBIENT)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR, SPECULAR)
        glMaterialfv(GL_FRONT, GL_SHININESS, SHININESS)

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
