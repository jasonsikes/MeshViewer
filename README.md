# Mesh Viewer

A viewer for winged-edge (half-edge) meshes with butterfly subdivision. The
primary application is a PySide6 window that draws with OpenGL 4.1.

## Requirements

* Python 3.x
* PySide6
* PyOpenGL 3.x
* NumPy
* Pillow (PIL)
* OpenGL 4.1 core profile

## Files

* `MeshViewer.py`: Entry point, application window, and menus.
* `MeshGLWidget.py`: OpenGL widget (shaders, camera, mesh drawing, vertex labels).
* `Mesh.py`: Winged-edge mesh, tetrahedron and cube, butterfly subdivision.
* `Bunny.py`: Stanford Bunny vertex and index data.
* `block_texture.png`: Optional mesh texture.

## Setup

Install the Python packages:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

From this directory:

```bash
python3 MeshViewer.py
```

The window opens on the tetrahedron (wireframe). Use the menu bar:

* **File → Quit** (`Q`)
* **Bunny**, **Tetrahedron**, **Cube**: built-in meshes and subdivision levels
* **Display**: shading, backface cull, vertex annotation, smooth shading, texture

Controls:

* Left-drag: rotate
* Mouse wheel: zoom
* `Q`: quit

## About the Bunny

The Stanford Bunny data originated at the
[Stanford University Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/). I used the low-polygon data set since the purpose of this application is to
demonstrate surface smoothing using butterfly subdivision. And that is why this bunny is so ugly.