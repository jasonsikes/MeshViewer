#! /usr/bin/env python3
# 
# ViewMesh.py
# Created by: Jason Sikes
# 

import sys

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image

from math import *
from numpy import *

from Mesh import *
from Bunny import *


HELP_TEXT = """
A viewer and demonstration of efficient subdivision using the Half-Edge (or 
"Winged Edge") data structure.

USAGE: Just run it! This program doesn't use command-line parameters except
whatever you want to pass on to GLUT.

This program is a simple viewer for a mesh. It allows you to view a mesh in 3D
space and, optionally, texture it. Several mesh models are included. In 
addition, all of the mesh objects are subdivided several times into more
detailed meshes.

Built-in meshes include:
    The Stanford Bunny
    Tetrahedron
    Cube (subdivided into triangles)

Use your keyboard to zoom in, zoom out, and quit.

KEYS:
    Z: Zoom in
    X: Zoom out
    Q: Quit

For all other functions and options, right-click and select from the pop-up
menu.
"""

TEXTURE_FILENAME = 'block_texture.png'
TEXTURE_ENCODING = GL_RGBA

# Mesh objects.
bunny = 0
subdividedBunny = 0
subdividedBunny2 = 0
subdividedBunny3 = 0
tetrahedron = 0
subdividedTetrahedron = 0
subdividedTetrahedron2 = 0
subdividedTetrahedron3 = 0
subdividedTetrahedron4 = 0
triCube = 0
subdividedTriCube = 0
subdividedTriCube2 = 0
subdividedTriCube3 = 0
mesh = 0

tetrahedronCentroid = 0
cubeCentroid = 0
bunnyCentroid = 0

eyeRadius = 2.5
eyePhi = pi/4
eyeTheta = pi/4
eye = array([0,0,0])
lookat = array([0,0,0])
up = array([0,0,1])



# Creates a 2D numpy array of texture coordinates
# For use with mesh objects.
def calculateTextureCoordinates(vertices, indices):
    ssi  = 0 # Vertex component that is source for texture s component
    tsi1 = 1 # Vertex component that is part 1 of source for texture t component (for atan2)
    tsi2 = 2 # Vertex component that is part 2 of source for texture t component (for atan2)

    retval = empty((len(indices), 3, 2), dtype = float32)
    # This is a modified copy of getCentroid(). Bad, I know.
    centroid = zeros(3)
    for vert in vertices:
        centroid += vert
    centroid /= len(vertices)

    maxssi = max( [vert[ssi] for vert in vertices] ) * 1.01
    minssi = min( [vert[ssi] for vert in vertices] ) * 1.01

    retvalIndex = 0
    for f in range(len(indices)):
        isInQ1 = False
        isInQ4 = False
        for v in range(len(indices[f])):
            vertex = vertices[indices[f][v]]
            # s is simply the normalized component
            retval[f,v,0] = (vertex[ssi] - minssi) / (maxssi - minssi)
            # t is derived from atan2
            retval[f,v,1] = atan2(vertex[tsi1] - centroid[tsi1], vertex[tsi2] - centroid[tsi2]) / pi / 2.0 + 0.5
            if retval[f,v,1] < 0.25:
                isInQ1 = True
            if retval[f,v,1] > 0.75:
                isInQ4 = True

        # Check for texture wrap-around
        if isInQ1 and isInQ4:
            for v in range(len(indices[f])):
                if retval[f,v,1] < 0.25:
                    retval[f,v,1] += 1
    return retval



def sphericalToCartesian(r, theta, phi):
    return array([r * cos(theta) * sin(phi),
                r * sin(theta) * sin(phi),
                r * cos(phi)])

def getCentroid(m):
    centroid = zeros(3)
    for vert in m.verts:
        centroid += vert.position
    centroid /= len(m.verts)
    return centroid

def setView(centroid):
    global lookat, eyeRadius, eyeTheta, eyePhi
    if (centroid is not None):
        lookat = centroid
    eye = sphericalToCartesian(eyeRadius, eyeTheta, eyePhi) + lookat
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2],
              lookat[0], lookat[1], lookat[2],
              up[0], up[1], up[2])

ambient = (0.6, 0.6, 0.6, 1)
diffuse = (0.5, 0.5, 0.5, 1)
specular = (0.5, 0.5, 0.5, 1)
shininess = 51.2

lModelAmbient = (0.1, 0.1, 0.1, 1)

light0Position = (30, 30, 30, 0)
light0Ambient = (0.3, 0.3, 0.3, 1)
light0Diffuse = (1, 1, 1, 1)
light0Specular = (1, 1, 1, 1)
openGLVertexBufferIDs = []

verticesBufferID = 0
smoothNormalsBufferID = 0
flatNormalsBufferID = 0
textureBufferID = 0
textureID = 0

def setMesh(aMesh):
    global mesh
    mesh = aMesh

    glBindBuffer(GL_ARRAY_BUFFER, verticesBufferID)
    glBufferData(GL_ARRAY_BUFFER,
                mesh.vboVertices,
                GL_STATIC_DRAW)


    glBindBuffer(GL_ARRAY_BUFFER, flatNormalsBufferID)
    glBufferData(GL_ARRAY_BUFFER,
                mesh.vboFlatNormals,
                GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, smoothNormalsBufferID)
    glBufferData(GL_ARRAY_BUFFER,
                mesh.vboSmoothNormals,
                GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, textureBufferID)
    glBufferData(GL_ARRAY_BUFFER,
                mesh.vboTexCoords,
                GL_STATIC_DRAW)

def initTexture():
    global textureID
    img = Image.open(TEXTURE_FILENAME)
    if img is None:
        print("Failed to load texture!")
        return
    img_data = array(list(img.getdata()), uint8)

    glEnable(GL_TEXTURE_2D)
    textureID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureID)
    glTexImage2D(GL_TEXTURE_2D, 0, TEXTURE_ENCODING, img.width, img.height,
                0, TEXTURE_ENCODING, GL_UNSIGNED_BYTE, img_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)



def initGL():
    global bunny, subdividedBunny, subdividedBunny2, subdividedBunny3, bunnyCentroid
    global mesh, tetrahedron, subdividedTetrahedron, subdividedTetrahedron2
    global subdividedTetrahedron3, subdividedTetrahedron4, triCube, subdividedTriCube
    global subdividedTriCube2, subdividedTriCube3, tetrahedronCentroid, cubeCentroid
    global verticesBufferID, smoothNormalsBufferID, flatNormalsBufferID, textureBufferID

    [verticesBufferID, smoothNormalsBufferID, flatNormalsBufferID, textureBufferID] = glGenBuffers(4)
    
    bunnyTexCoords = calculateTextureCoordinates(Bunny.bunnyVertices, Bunny.bunnyIndices)
    bunny = Mesh(Bunny.bunnyVertices,Bunny.bunnyIndices, bunnyTexCoords)

    newVertices = bunny.copyOfVertices()
    newIndices = bunny.copyOfIndices()
    newCoords = calculateTextureCoordinates(newVertices, newIndices)

    subdividedBunny = Mesh(newVertices, newIndices, newCoords)
    subdividedBunny.butterflySubdivide()

    newVertices = subdividedBunny.copyOfVertices()
    newIndices = subdividedBunny.copyOfIndices()
    newCoords = calculateTextureCoordinates(newVertices, newIndices)

    subdividedBunny2 = Mesh(newVertices, newIndices, newCoords)
    subdividedBunny2.butterflySubdivide()

    tetrahedron = Mesh.Tetrahedron(1)

    subdividedTetrahedron = copy.deepcopy(tetrahedron)
    subdividedTetrahedron.butterflySubdivide()

    subdividedTetrahedron2 = copy.deepcopy(subdividedTetrahedron)
    subdividedTetrahedron2.butterflySubdivide()

    subdividedTetrahedron3 = copy.deepcopy(subdividedTetrahedron2)
    subdividedTetrahedron3.butterflySubdivide()

    subdividedTetrahedron4 = copy.deepcopy(subdividedTetrahedron3)
    subdividedTetrahedron4.butterflySubdivide()

    triCube = Mesh.Cube(1)
    triCube.triangulate()

    subdividedTriCube = copy.deepcopy(triCube)
    subdividedTriCube.butterflySubdivide()

    subdividedTriCube2 = copy.deepcopy(subdividedTriCube)
    subdividedTriCube2.butterflySubdivide()

    subdividedTriCube3 = copy.deepcopy(subdividedTriCube2)
    subdividedTriCube3.butterflySubdivide()

    setMesh(tetrahedron)

    tetrahedronCentroid = getCentroid(tetrahedron)
    cubeCentroid = getCentroid(triCube)
    bunnyCentroid = getCentroid(bunny)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40, 1, 0.1, 30)

    setView(tetrahedronCentroid)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lModelAmbient)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light0Ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0Diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0Specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light0Position)
    glEnable(GL_LIGHT0)

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)

    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)

    glCullFace(GL_BACK)
    glDisable(GL_CULL_FACE)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glDisable(GL_LIGHTING)
    initTexture()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)

    glColor3f(1,1,1)

    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, verticesBufferID)
    glVertexPointer(3, GL_FLOAT, 0, None)

    if texture:
        glBindTexture(GL_TEXTURE_2D, textureID)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, textureBufferID)

        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, textureBufferID)
        glTexCoordPointer(2, GL_FLOAT, 0, None)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, textureID)
    else:
        glDisable(GL_TEXTURE_2D)

    if smooth:
        glBindBuffer(GL_ARRAY_BUFFER, smoothNormalsBufferID)
    else:
        glBindBuffer(GL_ARRAY_BUFFER, flatNormalsBufferID)
    glNormalPointer(GL_FLOAT, 0, None)

    glDrawArrays(GL_TRIANGLES, 0, len(mesh.vboVertices) // 3)
    glDisable(GL_DEPTH_TEST)

    if annotate:
        verts = mesh.verts
        glDisable(GL_LIGHTING)
        for v in verts:
            buf = "v%i" % (v.index)
            glRasterPos3fv(v.position)
            for cp in buf:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(cp))
        if shade:
            glEnable(GL_LIGHTING)

    # End testTextureSetup
    glutSwapBuffers()


mouseRotate = False
mousex = 0
mousey = 0
epsilon = 0.0000001

def mouse(button, state, x, y):
    global mouseRotate, mousex, mousey
    if state == GLUT_DOWN:
        if button == GLUT_LEFT_BUTTON:
            mouseRotate = True
            mousex = x
            mousey = y
    elif state == GLUT_UP:
        mouseRotate = False


def mouseMotion(x, y):
    global eyeTheta, eyePhi, mousex, mousey
    if mouseRotate:
        radiansPerPixel = pi / 180
        dx = x - mousex
        dy = y - mousey
        eyeTheta -= dx * radiansPerPixel
        eyePhi -= dy * radiansPerPixel
        if eyePhi >= pi:
            eyePhi = pi - epsilon
        elif eyePhi <= 0:
            eyePhi = epsilon

        eye = sphericalToCartesian(eyeRadius, eyeTheta, eyePhi)
        eye += lookat
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2],
                lookat[0], lookat[1], lookat[2],
                up[0], up[1], up[2])
        mousex = x
        mousey = y
        glutPostRedisplay()


# Menu Enumerations
MENU_BUNNY = 1
MENU_SUBDIVIDED_BUNNY = 2
MENU_SUBDIVIDED_BUNNY2 = 3
MENU_TETRAHEDRON = 4
MENU_SUBDIVIDED_TETRAHEDRON = 5
MENU_SUBDIVIDED_TETRAHEDRON2 = 6
MENU_SUBDIVIDED_TETRAHEDRON3 = 7
MENU_SUBDIVIDED_TETRAHEDRON4 = 8
MENU_TRI_CUBE = 9
MENU_SUBDIVIDED_TRI_CUBE = 10
MENU_SUBDIVIDED_TRI_CUBE2 = 11
MENU_SUBDIVIDED_TRI_CUBE3 = 12
MENU_SHADE = 13
MENU_CULL_BACKFACES = 14
MENU_ANNOTATE = 15
MENU_SMOOTH_SHADING = 16
MENU_TEXTURE = 17
MENU_DIVIDER = 888
MENU_QUIT = 999

shade = False
cull = False
annotate = False
smooth = False
texture = False

def menu(value):
    global mesh, shade, cull, annotate, smooth, window, texture
    if value == MENU_BUNNY:
        setMesh(bunny)
        setView(bunnyCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_BUNNY:
        setMesh(subdividedBunny)
        setView(bunnyCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_BUNNY2:
        setMesh(subdividedBunny2)
        setView(bunnyCentroid)
        glutPostRedisplay()
    if value == MENU_TETRAHEDRON:
        setMesh(tetrahedron)
        setView(tetrahedronCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TETRAHEDRON:
        setMesh(subdividedTetrahedron)
        setView(tetrahedronCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TETRAHEDRON2:
        setMesh(subdividedTetrahedron2)
        setView(tetrahedronCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TETRAHEDRON3:
        setMesh(subdividedTetrahedron3)
        setView(tetrahedronCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TETRAHEDRON4:
        setMesh(subdividedTetrahedron4)
        setView(tetrahedronCentroid)
        glutPostRedisplay()
    if value == MENU_TRI_CUBE:
        setMesh(triCube)
        setView(cubeCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TRI_CUBE:
        setMesh(subdividedTriCube)
        setView(cubeCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TRI_CUBE2:
        setMesh(subdividedTriCube2)
        setView(cubeCentroid)
        glutPostRedisplay()
    if value == MENU_SUBDIVIDED_TRI_CUBE3:
        setMesh(subdividedTriCube3)
        setView(cubeCentroid)
        glutPostRedisplay()
    if value == MENU_SMOOTH_SHADING:
        smooth = not smooth
        glutPostRedisplay()
    if value == MENU_SHADE:
        shade = not shade
        if shade:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
        glutPostRedisplay()
    if value == MENU_CULL_BACKFACES:
        cull = not cull
        if cull:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)
        glutPostRedisplay()
    if value == MENU_ANNOTATE:
        annotate = not annotate
        glutPostRedisplay()
    if value == MENU_TEXTURE:
        texture = not texture
        glutPostRedisplay()
    if value == MENU_QUIT:
        if window:
            glutDestroyWindow(window)
    return 0

def keyboard(key, x, y):
    global window, eyeRadius
    if (key == as_8_bit('z')) or (key == as_8_bit('Z')):
        eyeRadius *= 0.95
        setView(None)
        glutPostRedisplay()
    if (key == as_8_bit('x')) or (key == as_8_bit('X')):
        eyeRadius *= 1.05
        setView(None)
        glutPostRedisplay()
    if (key == as_8_bit('q')) or (key == as_8_bit('Q')):
        if window:
            glutDestroyWindow(window)

def main():
    global window

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(700,700)
    glutInitWindowPosition(10,10)
    window = glutCreateWindow("Winged-Edge Mesh in Python")
    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(mouseMotion)
    glutCreateMenu(menu)
    glutAddMenuEntry("Low-Res Bunny", MENU_BUNNY)
    glutAddMenuEntry("Low-Res Bunny Subdivided 1 iteration", MENU_SUBDIVIDED_BUNNY)
    glutAddMenuEntry("Low-Res Bunny Subdivided 2 iterations", MENU_SUBDIVIDED_BUNNY2)
    glutAddMenuEntry("----------------------------", MENU_DIVIDER)
    glutAddMenuEntry("Tetrahedron", MENU_TETRAHEDRON)
    glutAddMenuEntry("Tetrahedron Subdivided 1 iteration", MENU_SUBDIVIDED_TETRAHEDRON)
    glutAddMenuEntry("Tetrahedron Subdivided 2 iterations", MENU_SUBDIVIDED_TETRAHEDRON2)
    glutAddMenuEntry("Tetrahedron Subdivided 3 iterations", MENU_SUBDIVIDED_TETRAHEDRON3)
    glutAddMenuEntry("Tetrahedron Subdivided 4 iterations", MENU_SUBDIVIDED_TETRAHEDRON4)
    glutAddMenuEntry("----------------------------", MENU_DIVIDER)
    glutAddMenuEntry("Triangulated Cube", MENU_TRI_CUBE)
    glutAddMenuEntry("Triangulated Cube Subdivided 1 iteration", MENU_SUBDIVIDED_TRI_CUBE)
    glutAddMenuEntry("Triangulated Cube Subdivided 2 iterations", MENU_SUBDIVIDED_TRI_CUBE2)
    glutAddMenuEntry("Triangulated Cube Subdivided 3 iterations", MENU_SUBDIVIDED_TRI_CUBE3)
    glutAddMenuEntry("----------------------------", MENU_DIVIDER)
    glutAddMenuEntry("Shading on/off", MENU_SHADE)
    glutAddMenuEntry("Cull backfaces on/off", MENU_CULL_BACKFACES)
    glutAddMenuEntry("Vertex annotation on/off", MENU_ANNOTATE)
    glutAddMenuEntry("Smooth shading on/off", MENU_SMOOTH_SHADING)
    glutAddMenuEntry("Texture on/off", MENU_TEXTURE)
    glutAddMenuEntry("----------------------------", MENU_DIVIDER)
    glutAddMenuEntry("Quit", MENU_QUIT)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    glutKeyboardFunc(keyboard)
    initGL()
    print(HELP_TEXT)
    glutMainLoop()

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    glutInit(sys.argv)
    main()

