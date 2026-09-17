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
from PySide6.QtGui import QFontDatabase, QPainter, QSurfaceFormat
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
LIGHT0_POSITION = (30.0, 30.0, 30.0, 0.0)
LIGHT0_AMBIENT = (0.3, 0.3, 0.3, 1)
LIGHT0_DIFFUSE = (1, 1, 1, 1)
LIGHT0_SPECULAR = (1, 1, 1, 1)

ORBIT_RADIANS_PER_PIXEL = pi / 180
ORBIT_PHI_EPSILON = 1e-7
ZOOM_IN_FACTOR = 0.95

VERTEX_SHADER = """
#version 410 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

uniform mat4 mvp;
uniform mat3 normalMatrix;

out vec3 eyeNormal;
out vec2 vTexCoord;

void main()
{
    eyeNormal = normalMatrix * normal;
    vTexCoord = texCoord;
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 410 core

in vec3 eyeNormal;
in vec2 vTexCoord;

uniform int shade;
uniform int useTexture;
uniform sampler2D meshTexture;

uniform vec4 lightModelAmbient;
uniform vec4 lightPosition;
uniform vec4 lightAmbient;
uniform vec4 lightDiffuse;
uniform vec4 lightSpecular;
uniform vec4 materialAmbient;
uniform vec4 materialDiffuse;
uniform vec4 materialSpecular;
uniform float materialShininess;

out vec4 fragColor;

void main()
{
    vec3 color = vec3(1.0);
    if (shade != 0) {
        vec3 N = normalize(eyeNormal);
        vec3 L = normalize(lightPosition.xyz);
        vec3 V = vec3(0.0, 0.0, 1.0);
        vec3 H = normalize(L + V);
        float ndotl = max(dot(N, L), 0.0);
        vec3 ambient = materialAmbient.rgb
            * (lightModelAmbient.rgb + lightAmbient.rgb);
        vec3 diffuse = materialDiffuse.rgb * lightDiffuse.rgb * ndotl;
        vec3 specular = vec3(0.0);
        if (ndotl > 0.0) {
            specular = materialSpecular.rgb * lightSpecular.rgb
                * pow(max(dot(N, H), 0.0), materialShininess);
        }
        color = ambient + diffuse + specular;
    }
    vec4 outColor = vec4(color, 1.0);
    if (useTexture != 0) {
        outColor *= texture(meshTexture, vTexCoord);
    }
    fragColor = outColor;
}
"""


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


def _gl_info_log(log):
    if log is None:
        return ""
    if isinstance(log, bytes):
        return log.decode("utf-8", errors="replace")
    return str(log)


def _gl_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(value[0])


def _gl_succeeded(value):
    return int(ascontiguousarray(value).reshape(-1)[0]) != 0


def _compile_shader(source, shader_type, label):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not _gl_succeeded(glGetShaderiv(shader, GL_COMPILE_STATUS)):
        raise RuntimeError("%s shader compile failed:\n%s" % (
            label, _gl_info_log(glGetShaderInfoLog(shader))))
    return shader


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
        self.vao = 0
        self.program = 0
        self._uniforms = {}

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
        self._require_gl_41_core()
        glClearColor(0.12, 0.12, 0.14, 1.0)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)

        self.program = self._create_program()
        self.vao = _gl_id(glGenVertexArrays(1))
        (self.vertices_buffer_id,
         self.smooth_normals_buffer_id,
         self.flat_normals_buffer_id,
         self.texture_buffer_id) = [_gl_id(buffer_id) for buffer_id in glGenBuffers(4)]

        self._create_meshes()
        self.tetrahedron_centroid = get_centroid(self.tetrahedron)
        self.cube_centroid = get_centroid(self.tri_cube)
        self.bunny_centroid = get_centroid(self.bunny)
        self.mesh = self.tetrahedron
        self._upload_mesh()

        self.set_lookat(self.tetrahedron_centroid)
        self._init_texture()

    def _require_gl_41_core(self):
        ctx = self.context()
        fmt = ctx.format() if ctx is not None else self.format()
        version = (fmt.majorVersion(), fmt.minorVersion())
        profile = fmt.profile()
        gl_version = _gl_info_log(glGetString(GL_VERSION)).strip()
        if profile != QSurfaceFormat.CoreProfile or version < (4, 1):
            raise RuntimeError(
                "OpenGL 4.1 core is required. "
                "Got Qt format %d.%d profile=%s; GL_VERSION=%s" % (
                    version[0], version[1], profile, gl_version or "unknown"))

    def _create_program(self):
        vertex_shader = _compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER, "Vertex")
        fragment_shader = _compile_shader(
            FRAGMENT_SHADER, GL_FRAGMENT_SHADER, "Fragment")
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        if not _gl_succeeded(glGetProgramiv(program, GL_LINK_STATUS)):
            raise RuntimeError("Shader link failed:\n%s" % _gl_info_log(
                glGetProgramInfoLog(program)))

        self._uniforms = {}
        for name in (
            "mvp", "normalMatrix", "shade", "useTexture", "meshTexture",
            "lightModelAmbient", "lightPosition", "lightAmbient", "lightDiffuse",
            "lightSpecular", "materialAmbient", "materialDiffuse",
            "materialSpecular", "materialShininess",
        ):
            self._uniforms[name] = glGetUniformLocation(program, name)
        return int(program)

    def closeEvent(self, event):
        self.delete_gl_objects()
        super().closeEvent(event)

    def delete_gl_objects(self):
        if not self.isValid():
            return
        self.makeCurrent()
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        buffer_ids = [buffer_id for buffer_id in (
            self.vertices_buffer_id,
            self.smooth_normals_buffer_id,
            self.flat_normals_buffer_id,
            self.texture_buffer_id,
        ) if buffer_id]
        if buffer_ids:
            glDeleteBuffers(len(buffer_ids), buffer_ids)
        self.vertices_buffer_id = 0
        self.smooth_normals_buffer_id = 0
        self.flat_normals_buffer_id = 0
        self.texture_buffer_id = 0

        if self.texture_id:
            glDeleteTextures(1, [self.texture_id])
        self.texture_id = 0
        if self.program:
            glDeleteProgram(self.program)
            self.program = 0
        self.doneCurrent()

    def paintGL(self):
        if self._buffers_dirty:
            self._upload_mesh()
            self._buffers_dirty = False

        self._apply_view()
        self._set_projection(self._viewport[2], self._viewport[3])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glUseProgram(self.program)
        self._set_shader_uniforms()

        if self.shade:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.cull:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)

        self._bind_mesh_attributes()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glDrawArrays(GL_TRIANGLES, 0, len(self.mesh.vboVertices) // 3)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

        if self.annotate:
            self._draw_vertex_labels()

    def _set_shader_uniforms(self):
        u = self._uniforms
        mvp = self._proj_matrix @ self._view_matrix
        glUniformMatrix4fv(u["mvp"], 1, GL_FALSE, _as_gl_matrix(mvp))
        glUniformMatrix3fv(
            u["normalMatrix"], 1, GL_FALSE,
            _as_gl_matrix(self._view_matrix[0:3, 0:3]))
        glUniform1i(u["shade"], 1 if self.shade else 0)
        glUniform1i(u["useTexture"], 1 if self.texture else 0)
        glUniform1i(u["meshTexture"], 0)
        glUniform4f(u["lightModelAmbient"], *LIGHT_MODEL_AMBIENT)
        glUniform4f(u["lightPosition"], *LIGHT0_POSITION)
        glUniform4f(u["lightAmbient"], *LIGHT0_AMBIENT)
        glUniform4f(u["lightDiffuse"], *LIGHT0_DIFFUSE)
        glUniform4f(u["lightSpecular"], *LIGHT0_SPECULAR)
        glUniform4f(u["materialAmbient"], *AMBIENT)
        glUniform4f(u["materialDiffuse"], *DIFFUSE)
        glUniform4f(u["materialSpecular"], *SPECULAR)
        glUniform1f(u["materialShininess"], SHININESS)

    def _bind_mesh_attributes(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer_id)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        normal_id = (self.smooth_normals_buffer_id if self.smooth
                     else self.flat_normals_buffer_id)
        glBindBuffer(GL_ARRAY_BUFFER, normal_id)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.texture_buffer_id)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

    def resizeGL(self, width, height):
        fb_w = max(1, width)
        fb_h = max(1, height)
        self._viewport = (0, 0, fb_w, fb_h)
        glViewport(0, 0, fb_w, fb_h)
        self._set_projection(fb_w, fb_h)

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
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_lookat(self, centroid):
        if centroid is not None:
            self.lookat = centroid

    def _apply_view(self):
        eye = spherical_to_cartesian(self.eye_radius, self.eye_theta, self.eye_phi)
        eye = eye + self.lookat
        self._view_matrix = look_at_matrix(eye, self.lookat, self.up)

    def _set_projection(self, width, height):
        aspect = max(1, width) / max(1, height)
        self._proj_matrix = perspective_matrix(40.0, aspect, 0.1, 30.0)

    def _draw_vertex_labels(self):
        # QPainter's OpenGL engine inherits the current polygon mode, so
        # GL_LINE (wireframe / shading off) would stroke glyph quads as
        # outlines instead of filling them.
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glBindVertexArray(0)
        glUseProgram(0)

        painter = QPainter(self)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPixelSize(18)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.yellow)
        dpr = self.devicePixelRatioF()
        height = self.height()
        for vertex in self.mesh.verts:
            if vertex is None:
                continue
            projected = project_to_window(
                vertex.position, self._view_matrix, self._proj_matrix,
                self._viewport)
            if projected is None:
                continue
            win_x, win_y, win_z = projected
            if win_z < 0.0 or win_z > 1.0:
                continue
            painter.drawText(int(round(win_x / dpr)),
                             int(round(height - win_y / dpr)),
                             "v%i" % vertex.index)
        painter.end()

    def _init_texture(self):
        img = Image.open(TEXTURE_FILENAME)
        img_data = array(list(img.getdata()), uint8)
        self.texture_id = _gl_id(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, TEXTURE_ENCODING, img.width, img.height,
                     0, TEXTURE_ENCODING, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glBindTexture(GL_TEXTURE_2D, 0)
