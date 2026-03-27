import open3d as o3d
import numpy as np


def create_mesh_from_rgbd(rgb_image, depth_image, intrinsic,
                          depth_scale=1000.0,
                          depth_trunc=3.0,
                          max_edge_diff=0.03):
    color = np.asarray(rgb_image)
    depth_raw = np.asarray(depth_image)
    depth = depth_raw.astype(np.float32) / depth_scale

    rows, cols = depth.shape[:2]

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    vertices = []
    colors = []
    index_map = -np.ones((rows, cols), dtype=np.int32)

    k = 0
    for i in range(rows):
        for j in range(cols):
            z = depth[i, j]
            if z <= 0 or z >= depth_trunc:
                continue

            x = (j - cx) * z / fx
            y = (i - cy) * z / fy

            vertices.append([x, y, z])

            if color.ndim == 3:
                colors.append(color[i, j] / 255.0)
            else:
                g = color[i, j] / 255.0
                colors.append([g, g, g])

            index_map[i, j] = k
            k += 1

    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = index_map[i, j]
            v1 = index_map[i, j + 1]
            v2 = index_map[i + 1, j]
            v3 = index_map[i + 1, j + 1]

            z0 = depth[i, j]
            z1 = depth[i, j + 1]
            z2 = depth[i + 1, j]
            z3 = depth[i + 1, j + 1]

            if v0 >= 0 and v1 >= 0 and v2 >= 0 and max(z0, z1, z2) - min(z0, z1, z2) < max_edge_diff:
                triangles.append([v2, v1, v0])

            if v2 >= 0 and v1 >= 0 and v3 >= 0 and max(z2, z1, z3) - min(z2, z1, z3) < max_edge_diff:
                triangles.append([v3, v1, v2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    return mesh

if __name__ == '__main__':
    # read the color and the depth image:
    rgb_image = o3d.io.read_image("rgb.jpg")
    depth_image = o3d.io.read_image("depth.png")

    # create an rgbd image object:
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False)


    # use the rgbd image to create point cloud:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # visualize:
    o3d.visualization.draw_geometries([pcd])

    mesh = create_mesh_from_rgbd(
        rgb_image,
        depth_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        depth_scale=1000.0,
        depth_trunc=3.0,
        max_edge_diff=0.03
    )
    
    
    o3d.visualization.draw_geometries([mesh], window_name="Mesh")
    o3d.io.write_triangle_mesh("mesh.ply", mesh)