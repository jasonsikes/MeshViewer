# Mesh.py
# Created by: Jason Sikes
# 
# A Python implementation of the Winged-Edge (or Half-Edge) data structure.

from math import *
from numpy import *
import sys
import copy
from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray

class Edge:
    pass

class Vertex:
    position: NDArray[np.float32]
    eminatingEdge: Optional['Edge']
    index: int

class Face:
    pass


class Mesh:
    # vertices is the coordinates of vertices in 3D space, given as a N x 3 Numpy array 
    # faces is a Python array of arrays (not numpy).
    # Each subarray describes a face by its vertex index.
    # For example:faces = [[0,1,2], [1,3,2]] describes two adjacent triangles
    # textureCoordinates is a N x 3 x 2 Numpy array

    def __init__(self, verts, faces, texCoords):
        self.faces = [Face() for x in range(len(faces))]
        self.edges = [Edge() for x in range((2*(len(faces) + len(verts) - 2)))]
        self.verts = [None for x in range(len(verts))]
        edgeMap = {}

        firstEdgeOfFace = 0

        for faceIndex in range(len(faces)):
            self.faces[faceIndex].edge = self.edges[firstEdgeOfFace]
            countOfVertices = len(faces[faceIndex])                      # typing shortcut
            for vertexIndex in range(countOfVertices):
                v0 = faces[faceIndex][vertexIndex]                     # edge (v0, v1)  
                v1 = faces[faceIndex][(vertexIndex+1) % countOfVertices]
                e = self.edges[firstEdgeOfFace + vertexIndex] # typing and comprehension shortcut
                if (self.verts[v0]):
                    # Vertex is already created
                    e.vertex = self.verts[v0]
                else:
                    # Vertex is not found. Create one.
                    e.vertex = Vertex()
                    e.vertex.position = verts[v0,:]
                    e.vertex.eminatingEdge = e
                    self.verts[v0] = e.vertex
                e.face = self.faces[faceIndex]
                nextIndex = (vertexIndex + 1) % countOfVertices
                previousIndex = (vertexIndex - 1 + countOfVertices) % countOfVertices
                e.nextEdge = self.edges[firstEdgeOfFace + nextIndex]
                e.previousEdge = self.edges[firstEdgeOfFace + previousIndex]
                e.texCoord = texCoords[faceIndex,vertexIndex,:]

                symmetricEdge = edgeMap.get("%i,%i" % (v1, v0))
                if (symmetricEdge):
                    e.symmetricEdge = symmetricEdge # Symmetric edge already processed
                    symmetricEdge.symmetricEdge = e
                else:
                    edgeMap["%i,%i" % (v0, v1)] = e # not processed. Put in map

            firstEdgeOfFace += countOfVertices

        i = 0
        for v in self.verts:
            v.index = i
            i += 1

        self.triangulate()
        self.computeNormals()
        self.createOpenGLArrays()


    # Create arrays of vertices and faces appropriate for creating new meshes
    def copyOfVertices(self):
        retval = empty((len(self.verts), 3), dtype=float32)
        for i in range(len(self.verts)):
            retval[i,:] = self.verts[i].position.copy()
        return retval

    def copyOfIndices(self):
        retval = []
        for f in self.faces:
            fi = []
            s = e = f.edge
            while True:
                fi.append(e.vertex.index)
                e = e.nextEdge
                if e == s:
                    break
            retval.append(fi)
        return retval

    def normalize(self, d):
        dd = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if (dd > 0):
            d /= dd


    def triangulateFace(self, f):
        while(True):
            e = f.edge
            e1 = e.nextEdge
            e2 = e1.nextEdge
            e3 = e2.nextEdge
            if (e == e3):
                break


            v1 = e1.vertex
            v3 = e3.vertex

            enew = Edge()
            esym = Edge()
            fnew = Face()

            enew.vertex = v1
            enew.face = f
            enew.nextEdge = e3
            enew.previousEdge = e
            enew.symmetricEdge = esym
            enew.texCoord = e1.texCoord

            esym.vertex = v3
            esym.face = fnew
            esym.nextEdge = e1
            esym.previousEdge = e2
            esym.symmetricEdge = enew
            esym.texCoord = e3.texCoord

            fnew.edge = e1

            e1.previousEdge = e2.nextEdge = esym
            e3.previousEdge = e.nextEdge = enew
            e1.face = e2.face = fnew

            self.faces.append(fnew)
            self.edges.append(enew)
            self.edges.append(esym)
            f.edge = enew


    def triangulate(self):
        for face in self.faces:
            self.triangulateFace(face)
        self.computeNormals()


    def Tetrahedron(a):
        vertices = array([[sqrt(3)/3 * a,   0,       0              ],
                          [-sqrt(3)/6 * a,  0.5 * a, 0              ],
                          [-sqrt(3)/6 * a, -0.5 * a, 0              ],
                          [0,               0,       sqrt(6) / 3 * a]])
        faces = [[0,3,2],[1,2,3],[1,3,0],[1,0,2]]
        b = 0.788675
        c = 0.211325
        texCoord = array([ [ [0.5, 0.5],
                             [c,1],
                             [b,1] ],
                           [ [0.5, 0.5],
                             [1,b],
                             [1,c] ],
                           [ [0.5, 0.5],
                             [b,0],
                             [c,0] ],
                           [ [0.5, 0.5],
                             [0,c],
                             [0,b] ] ], dtype = float32)
        return Mesh(vertices, faces, texCoord)
    Tetrahedron = staticmethod(Tetrahedron)


    def Cube(a):
        vertices = array([[0,0,0],
                          [a,0,0],
                          [a,a,0],
                          [0,a,0],
                          [0,0,a],
                          [a,0,a],
                          [a,a,a],
                          [0,a,a]])
        faces = [[0,1,5,4],
                 [1,2,6,5],
                 [2,3,7,6],
                 [3,0,4,7],
                 [3,2,1,0],
                 [4,5,6,7]]
        d2 = 1.0 / 2.0
        d3 = 1.0 / 3.0
        texCoord = array([ [ [0, 0],
                             [d3, 0],
                             [d3, d2],
                             [0, d2] ],
                           [ [d3, 0],
                             [2 * d3, 0],
                             [2 * d3, d2],
                             [d3, d2] ],
                           [ [2 * d3, 0],
                             [0.99, 0],
                             [0.99, d2],
                             [2 * d3, d2] ],
                           [ [0, d2],
                             [d3, d2],
                             [d3, 0.99],
                             [0, 0.99] ],
                           [ [d3, d2],
                             [2 * d3, d2],
                             [2 * d3, 0.99],
                             [d3, 0.99] ],
                           [ [2 * d3, d2],
                             [0.99, d2],
                             [0.99, 0.99],
                             [2 * d3, 0.99] ] ], dtype = float32)

        return Mesh(vertices, faces, texCoord)
    Cube = staticmethod(Cube)


    def oddLoopVertices(self):
        vMap = {}
        for e in self.edges:
            if (e > e.symmetricEdge): # Associate vertex with the smaller edge pointer
                continue
            pos = (e.vertex.position + e.symmetricEdge.vertex.position) * (3.0/8) +\
                (e.previousEdge.vertex.position + e.symmetricEdge.previousEdge.vertex.position) * (1.0/8)
            vMap[e] = pos
        return vMap

    def evenLoopVertices(self):
        vMap = {}
        for v in self.verts:
            n = 0
            pos = zeros((3))
            e = s = v.eminatingEdge
            while True:
                pos += e.symmetricEdge.vertex.position
                n += 1
                e = e.symmetricEdge.nextEdge
                if (e == s):
                    break
            if n == 3:
                beta = 3.0/16
            else:
                beta = 3.0/(8*n)
            pos = pos * beta + v.position * (1 - n * beta)
            vMap[v] = pos
        return vMap

    def splitEdge(self, e, pos):
        e0 = Edge()
        e1 = Edge()
        v0 = Vertex()

        v0.index = len(self.verts)
        v0.position = pos
        v0.eminatingEdge = e0

        e0.vertex = v0
        e0.face = e.face
        e0.nextEdge = e.nextEdge
        e0.previousEdge = e
        e0.symmetricEdge = e.symmetricEdge
        e0.texCoord = (e.texCoord + e.nextEdge.texCoord) / 2.0

        e1.vertex = v0
        e1.face = e.symmetricEdge.face
        e1.nextEdge = e.symmetricEdge.nextEdge
        e1.previousEdge = e.symmetricEdge
        e1.symmetricEdge = e
        e1.texCoord = (e.symmetricEdge.texCoord + e.symmetricEdge.nextEdge.texCoord) / 2.0

        e.nextEdge.previousEdge = e0
        e.symmetricEdge.nextEdge.previousEdge = e1
        e.nextEdge = e0
        e.symmetricEdge.nextEdge = e1
        e.symmetricEdge.symmetricEdge = e0
        e.symmetricEdge = e1

        self.edges.append(e0)
        self.edges.append(e1)
        self.verts.append(v0)

    def splitAllEdges(self, vmap):
        for e in vmap.keys():
            self.splitEdge(e,vmap[e])

    def loopSubdivide(self):
        oddVertMap = self.oddLoopVertices()
        evenVertMap = self.evenLoopVertices()
        for v in self.verts:
            v.position = evenVertMap[v]
        self.splitAllEdges(oddVertMap)
        self.triangulate()
        self.computeNormals()
        self.createOpenGLArrays()

    def butterflySubdivide(self):
        vMap = {}
        for e in self.edges:
            if (id(e) > id(e.symmetricEdge)): # Associate vertex with smaller edge pointer
                continue
            p1 = e.vertex.position
            p2 = e.symmetricEdge.vertex.position
            p3 = e.nextEdge.symmetricEdge.vertex.position
            p4 = e.symmetricEdge.nextEdge.symmetricEdge.vertex.position
            q1 = e.symmetricEdge.nextEdge.symmetricEdge.nextEdge.symmetricEdge.vertex.position
            q2 = e.symmetricEdge.previousEdge.symmetricEdge.previousEdge.vertex.position
            q3 = e.nextEdge.symmetricEdge.nextEdge.symmetricEdge.vertex.position
            q4 = e.previousEdge.symmetricEdge.nextEdge.symmetricEdge.vertex.position

            pos = (8.0 * (p1 + p2) + 2.0 * (p3 + p4) - (q1 + q2 + q3 + q4)) / 16.0

            vMap[e] = pos

        self.splitAllEdges(vMap)
        self.triangulate()
        self.computeNormals()
        self.createOpenGLArrays()


    def computeNormals(self):
        # Calculate the flat shading normals
        for f in self.faces:
            # First, calculate the normal
            normal = zeros((3))
            e = s = f.edge

            while(True):
                v = e.vertex.position
                vNext = e.nextEdge.vertex.position
                normal += array([(v[1] - vNext[1]) * (v[2] + vNext[2]),
                                 (v[2] - vNext[2]) * (v[0] + vNext[0]),
                                 (v[0] - vNext[0]) * (v[1] + vNext[1])])
                e = e.nextEdge
                if (e == s):
                    break
            self.normalize(normal)

            # Now, apply the normal to all of the edges
            e = s = f.edge
            while(True):
                e.flatNormal = normal
                e = e.nextEdge
                if (e == s):
                    break

        # Calculate the smooth shading normals
        for v in self.verts:
            normal = zeros((3))
            e = s = v.eminatingEdge
            while(True):
                normal += e.flatNormal
                e = e.previousEdge.symmetricEdge
                if (e == s):
                    break
            self.normalize(normal)

            # Apply to all of the edges
            while (True):
                e.smoothNormal = normal
                e = e.previousEdge.symmetricEdge
                if (e == s):
                    break


    # Fetch all vertex coordinates and store them in
    # numpy arrays (for OpenGL VBOs)
    # Note that we are creating triangles so
    # triangulate() must be called shortly before this.
    def createOpenGLArrays(self):
        self.vboVertices = empty((3 * 3 * len(self.faces)), dtype = float32)
        self.vboSmoothNormals = empty((3 * 3 * len(self.faces)), dtype = float32)
        self.vboFlatNormals = empty((3 * 3 * len(self.faces)), dtype = float32)
        self.vboTexCoords = empty((3 * 2 * len(self.faces)), dtype = float32)

        i = 0

        for f in self.faces:
            s = e = f.edge
            while(True):
                self.vboVertices[i * 3 : i * 3 + 3] = e.vertex.position
                self.vboSmoothNormals[i * 3 : i * 3 + 3] = e.smoothNormal
                self.vboFlatNormals[i * 3 : i * 3 + 3] = e.flatNormal
                self.vboTexCoords[i * 2 : i * 2 + 2] = e.texCoord
                i += 1
                e = e.nextEdge
                if (e == s):
                    break
