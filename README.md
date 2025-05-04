# README for Mesh Viewer

## Requirements
* Python 3.x
* PyOpenGL 3.x
* PyOpenGL-accelerate 3.x
* GLU
* GLUT
* Python Imaging Library (PIL)

## Files
* `ViewMesh.py`: The viewer.
* `Mesh.py`: The mesh data structure for Tetrahedron and Cube. Includes winged-edge data structure and butterfly subdivision algorithm.
* `FixedBunny.py`: The Stanford Bunny mesh.
* `block_texture.png`: The texture.

## Usage
Run `python ViewMesh.py` to view the meshes. Use the pop-up menu to select mesh
objects and options. Use the keyboard to:
* `Z`: zoom in
* `X`: zoom out
* `Q`: quit.

