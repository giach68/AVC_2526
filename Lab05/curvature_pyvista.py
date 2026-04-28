import pyvista as pv
import numpy as np

mesh = pv.read("gaudi.ply")
print(mesh)
print(mesh.n_open_edges)
print(mesh.is_manifold)


cqual = mesh.compute_cell_quality('min_angle')

cqual.plot(show_edges=True)

curv = mesh.curvature(curv_type='mean')
print(curv)

max_val = np.percentile(np.abs(curv), 95) # Using 95th percentile to avoid outliers
clim = [-max_val, max_val]

mesh.plot(scalars=curv,
    cmap='bwr',   
    clim=clim,
    below_color='blue',
    above_color='red',),