import copy
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation=np.eye(4)):
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src.paint_uniform_color([1, 0.706, 0])
    tgt.paint_uniform_color([0, 0.651, 0.929])
    src.transform(transformation)
    o3d.visualization.draw_geometries([src, tgt])


def preprocess_point_cloud(pcd, voxel_size):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0,
            max_nn=30
        )
    )

    # Compute FPFH features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0,
            max_nn=100
        )
    )
    return pcd_down, pcd_fpfh



# ---- Load your partial scans ----
source = o3d.io.read_point_cloud("part1.ply")  # source scan
target = o3d.io.read_point_cloud("part2.ply")  # target scan


draw_registration_result(source, target)

