import igl
import numpy as np
import scipy.sparse as sp
import meshplot as mp

V, F = igl.read_triangle_mesh("gaudi.ply")

G = igl.grad(V, F)
f = V[:, 2]  # scalar function on vertices
grad_f = G @ f

p = mp.plot(V, F, c=f, shading={"wireframe": False})