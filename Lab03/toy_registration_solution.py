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


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result


def refine_registration(source, target, init_transformation, voxel_size):
    # ICP needs normals for point-to-plane
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )

    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=distance_threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


# ---- Load your partial scans ----
source = o3d.io.read_point_cloud("part1.ply")  # source scan
target = o3d.io.read_point_cloud("part2.ply")  # target scan


draw_registration_result(source, target)

# Optional cleanup
source, _ = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
target, _ = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

voxel_size = 5  # tune to your object scale (e.g. 5 mm if units are meters)

source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Rough alignment from features
result_ransac = execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
)

print("Global registration:")
print("Fitness:", result_ransac.fitness)
print("Inlier RMSE:", result_ransac.inlier_rmse)
print(result_ransac.transformation)

draw_registration_result(source_down, target_down, result_ransac.transformation)

# Refine with ICP
result_icp = refine_registration(
    source, target, result_ransac.transformation, voxel_size
)

print("\nICP refinement:")
print("Fitness:", result_icp.fitness)
print("Inlier RMSE:", result_icp.inlier_rmse)
print(result_icp.transformation)

draw_registration_result(source, target, result_icp.transformation)

# Merge
source_aligned = copy.deepcopy(source)
source_aligned.transform(result_icp.transformation)
merged = source_aligned + target
merged = merged.voxel_down_sample(voxel_size / 2.0)

o3d.io.write_point_cloud("registered_merged.ply", merged)
print("\nSaved merged cloud to registered_merged.ply")