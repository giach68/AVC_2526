# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
  
    pcd = o3d.io.read_point_cloud("registered_merged.ply")
    print("Displaying input pointcloud ...")
    pcd.estimate_normals()

    o3d.visualization.draw_geometries([pcd])
    alpha = 40
    print(f"alpha={alpha:.3f}")
    print('Running alpha shapes surface reconstruction ...')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha)
    mesh.compute_triangle_normals(normalized=True)

    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw_geometries([mesh,pcd], mesh_show_back_face=True)

    o3d.io.write_triangle_mesh("alpha_shape_mesh.ply", mesh)

    radii = [5, 10, 40, 80]
    print('Running ball pivoting surface reconstruction ...')
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    print(rec_mesh.is_watertight())
    print(rec_mesh.is_self_intersecting())
    print(rec_mesh.is_edge_manifold())

    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw_geometries([rec_mesh,pcd], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("ball_pivoting_mesh.ply", rec_mesh)

    print('Running Poisson surface reconstruction ...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=4)
    
    print(mesh.is_watertight())

    print(mesh.is_edge_manifold())

    print('Displaying reconstructed mesh ...')
    o3d.visualization.draw_geometries([mesh,pcd], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("poisson_mesh.ply", mesh)
