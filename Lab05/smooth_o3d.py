import open3d as o3d
# filter_smooth_simple
# filter_smooth_laplacian
# filter_smooth_taubin
# Load a mesh (e.g., from ply or obj)
mesh = o3d.io.read_triangle_mesh("gaudi_noise.ply")
mesh.compute_vertex_normals()

print(mesh)
print("Vertices:", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))
# Apply Laplacian smoothing
smoothed_mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
smoothed_mesh.compute_vertex_normals()

# Visualize
o3d.visualization.draw_geometries([smoothed_mesh])


mesh_laplacian = mesh.filter_smooth_laplacian(
    number_of_iterations=20,
    lambda_filter=0.5
)
mesh_laplacian.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [mesh_laplacian],
    window_name="Laplacian smoothing"
)

mesh_taubin = mesh.filter_smooth_taubin(
    number_of_iterations=20,
    lambda_filter=0.5,
    mu=-0.53
)
mesh_taubin.compute_vertex_normals()
o3d.visualization.draw_geometries(
    [mesh_taubin],
    window_name="Taubin smoothing"
)