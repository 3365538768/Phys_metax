import sys

sys.path.append("gaussian-splatting")
import os
import sys
import json
import torch
import numpy as np
import open3d as o3d

sys.path.append(os.getcwd())

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.transformation_utils import *
from utils.decode_param import decode_param_json


# -------------------------------------------------------
# load gaussian checkpoint
# -------------------------------------------------------

def load_checkpoint(model_path, sh_degree=3):

    checkpt_dir = os.path.join(model_path, "point_cloud")

    iteration = searchForMaxIteration(checkpt_dir)

    checkpt_path = os.path.join(
        checkpt_dir,
        f"iteration_{iteration}",
        "point_cloud.ply"
    )

    # ----------------------------
    # 自动检测 SH degree
    # ----------------------------

    with open(checkpt_path, "rb") as f:

        header = ""

        while True:

            line = f.readline().decode("utf-8")

            header += line

            if line.strip() == "end_header":
                break

    if "f_rest_" not in header:

        print("Detected PhysGaussian format")

        sh_degree = 0

    # ----------------------------
    # load gaussian
    # ----------------------------

    gaussians = GaussianModel(sh_degree)

    gaussians.load_ply(checkpt_path)

    return gaussians


# -------------------------------------------------------
# extract particles (same preprocess as gs_simulation)
# -------------------------------------------------------

def extract_particles(gaussians, preprocessing_params):

    pos = gaussians.get_xyz
    opacity = gaussians.get_opacity

    mask = opacity[:, 0] > preprocessing_params["opacity_threshold"]
    pos = pos[mask]

    # rotation
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"]
    )

    pos = apply_rotations(pos, rotation_matrices)

    # normalize
    pos, scale_origin, original_mean_pos = transform2origin(
        pos,
        preprocessing_params["scale"]
    )

    pos = shift2center111(pos)

    return pos.detach().cpu().numpy()


# -------------------------------------------------------
# detect press plate
# -------------------------------------------------------

def generate_press_bc(pts,
                      velocity=-0.1,
                      thickness=0.05,
                      margin=0.02,
                      scale_xy=1.2):

    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)

    center = [
        float((xmin + xmax) / 2),
        float((ymin + ymax) / 2),
        float(zmax + margin + thickness / 2)
    ]

    size = [
        float((xmax - xmin) * scale_xy),
        float((ymax - ymin) * scale_xy),
        float(thickness)
    ]

    bc = {

        "type": "set_velocity_on_cuboid",

        "point": center,

        "size": size,

        "velocity": [0, 0, velocity],

        "start_time": 0,

        "end_time": 2.5,

        "reset": 0
    }

    return bc, center, size


# -------------------------------------------------------
# visualization
# -------------------------------------------------------

def visualize(pts, plate_center, plate_size):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        pcd.points
    )
    bbox.color = (1, 0, 0)

    plate = o3d.geometry.AxisAlignedBoundingBox(

        min_bound=[
            plate_center[0] - plate_size[0] / 2,
            plate_center[1] - plate_size[1] / 2,
            plate_center[2] - plate_size[2] / 2
        ],

        max_bound=[
            plate_center[0] + plate_size[0] / 2,
            plate_center[1] + plate_size[1] / 2,
            plate_center[2] + plate_size[2] / 2
        ]

    )

    plate.color = (0, 1, 0)

    o3d.visualization.draw_geometries([pcd, bbox, plate])


# -------------------------------------------------------
# save debug ply
# -------------------------------------------------------

def save_debug_ply(points, path):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(path, pcd)


# -------------------------------------------------------
# main
# -------------------------------------------------------

def main():

    model_path = "model/cabin"
    base_config = "config/press_cube_jelly.json"

    print("\nLoading config...")

    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param_json(base_config)

    print("Loading gaussian model...")

    gaussians = load_checkpoint(model_path)

    print("Extracting particles...")

    pts = extract_particles(gaussians, preprocessing_params)

    print("Particle count:", pts.shape[0])

    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)

    print("bbox min:", xmin, ymin, zmin)
    print("bbox max:", xmax, ymax, zmax)

    print("\nGenerating press BC...")

    bc, center, size = generate_press_bc(pts)

    print(json.dumps(bc, indent=4))

    config = {

        "boundary_conditions": [

            {"type": "bounding_box"},

            bc

        ]
    }
    print(config)
    with open("press_auto.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nSaved config → press_auto.json")

    save_debug_ply(pts, "particles_debug.ply")

    print("Saved debug ply → particles_debug.ply")

    visualize(pts, center, size)


# -------------------------------------------------------

if __name__ == "__main__":
    main()