import os
import glob
import shutil
import argparse
import subprocess

import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
from open3d.visualization import rendering


# =====================================
# IO
# =====================================

def load_deformation_frames(folder):

    files = sorted(glob.glob(os.path.join(folder,"deformation_frame_*.npz")))

    frames = []

    for f in files:
        frames.append(np.load(f))

    return frames


def load_stress_frames(folder):

    files = sorted(glob.glob(os.path.join(folder,"stress_frame_*.npz")))

    frames = []

    for f in files:
        frames.append(np.load(f))

    return frames


# =====================================
# physics
# =====================================

def von_mises(stress):

    s = 0.5 * (stress + stress.transpose(0,2,1))

    mean = np.trace(s,axis1=1,axis2=2) / 3

    dev = s - mean[:,None,None] * np.eye(3)

    vm = np.sqrt(1.5 * np.sum(dev*dev,(1,2)))

    return vm


# =====================================
# geometry
# =====================================

def create_point_cloud(points,colors):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_trajectory_lines(all_pos):

    T,N,_ = all_pos.shape

    pts=[]
    lines=[]
    colors=[]

    idx=0

    for p in range(N):

        traj = all_pos[:,p]

        for i in range(T):
            pts.append(traj[i])

        for i in range(T-1):

            lines.append([idx+i,idx+i+1])

            t=i/(T-1)

            colors.append([t,0,1-t])

        idx+=T

    ls=o3d.geometry.LineSet()

    ls.points=o3d.utility.Vector3dVector(np.array(pts))

    ls.lines=o3d.utility.Vector2iVector(np.array(lines))

    ls.colors=o3d.utility.Vector3dVector(np.array(colors))

    return ls


# =====================================
# stress video
# =====================================

def render_stress_video(deformation_frames, stress_frames, output_path, width, height):

    tmp = os.path.join(output_path,"tmp_stress")

    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    os.makedirs(tmp)

    T = min(len(deformation_frames),len(stress_frames))

    # global stress range
    all_vm=[]

    for i in range(T):

        vm = von_mises(stress_frames[i]["stress_cauchy"])

        all_vm.append(vm)

    all_vm=np.concatenate(all_vm)

    all_vm=np.log(all_vm+1e-6)

    vmin=np.percentile(all_vm,1)
    vmax=np.percentile(all_vm,99)

    renderer = rendering.OffscreenRenderer(width,height)

    scene = renderer.scene

    scene.set_background([1,1,1,1])

    bounds=np.concatenate([f["position"] for f in deformation_frames],axis=0)

    center=bounds.mean(0)

    extent=bounds.max(0)-bounds.min(0)

    eye=center+np.array([1.5,1.5,1.5])*extent.max()

    scene.camera.look_at(center,eye,[0,0,1])

    mat=rendering.MaterialRecord()
    mat.shader="defaultUnlit"
    mat.point_size=6

    for i in tqdm(range(T)):

        if scene.has_geometry("pcd"):
            scene.remove_geometry("pcd")

        pos=deformation_frames[i]["position"]

        stress=stress_frames[i]["stress_cauchy"]

        vm=von_mises(stress)

        vm=np.log(vm+1e-6)

        col=(vm-vmin)/(vmax-vmin+1e-8)

        col=np.clip(col,0,1)

        colors=np.stack([col,np.zeros_like(col),1-col],axis=1)

        pcd=create_point_cloud(pos,colors)

        scene.add_geometry("pcd",pcd,mat)

        img = renderer.render_to_image()

        img_np = np.asarray(img)

        if img_np.size == 0:
            print("WARNING empty frame",i)
            continue

        img_np = cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(tmp,f"{i:04d}.png")

        cv2.imwrite(frame_path,img_np)

    subprocess.run([
        "ffmpeg","-y",
        "-framerate","20",
        "-i",os.path.join(tmp,"%04d.png"),
        "-c:v","libx264",
        "-pix_fmt","yuv420p",
        os.path.join(output_path,"stress_heatmap.mp4")
    ])


# =====================================
# trajectory video
# =====================================

def render_trajectory_video(frames, output_path, width, height):

    tmp=os.path.join(output_path,"tmp_traj")

    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    os.makedirs(tmp)

    T=len(frames)

    N=len(frames[0]["position"])

    all_pos=np.zeros((T,N,3))

    for i,f in enumerate(frames):
        all_pos[i]=f["position"]

    traj=create_trajectory_lines(all_pos)

    renderer=rendering.OffscreenRenderer(width,height)

    scene=renderer.scene

    scene.set_background([1,1,1,1])

    center=np.array([1,1,1])

    eye=center+np.array([2.5,2.5,2.5])

    scene.camera.look_at(center,eye,[0,0,1])

    mat_line=rendering.MaterialRecord()
    mat_line.shader="unlitLine"
    mat_line.line_width=2

    scene.add_geometry("traj",traj,mat_line)

    for i,f in enumerate(tqdm(frames)):

        if scene.has_geometry("pcd"):
            scene.remove_geometry("pcd")

        pos=f["position"]

        colors=np.ones_like(pos)*0.8

        pcd=create_point_cloud(pos,colors)

        mat=rendering.MaterialRecord()
        mat.shader="defaultUnlit"
        mat.point_size=3

        scene.add_geometry("pcd",pcd,mat)

        img=renderer.render_to_image()

        img_np=np.asarray(img)

        if img_np.size==0:
            print("WARNING empty frame",i)
            continue

        img_np=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(tmp,f"{i:04d}.png"),img_np)

    subprocess.run([
        "ffmpeg","-y",
        "-framerate","20",
        "-i",os.path.join(tmp,"%04d.png"),
        "-c:v","libx264",
        "-pix_fmt","yuv420p",
        os.path.join(output_path,"deformation_trajectory.mp4")
    ])


# =====================================
# main
# =====================================

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--simulation_dir",required=True)

    parser.add_argument("--output_path",required=True)

    parser.add_argument("--test",action="store_true")

    parser.add_argument("--test_frames",type=int,default=10)

    parser.add_argument("--width",type=int,default=1280)

    parser.add_argument("--height",type=int,default=960)

    args=parser.parse_args()

    os.makedirs(args.output_path,exist_ok=True)

    deformation_folder=os.path.join(args.simulation_dir,"deformation_field")

    stress_folder=os.path.join(args.simulation_dir,"stress_field")

    deformation_frames=load_deformation_frames(deformation_folder)

    stress_frames=load_stress_frames(stress_folder)

    if args.test:

        deformation_frames=deformation_frames[:args.test_frames]

        stress_frames=stress_frames[:args.test_frames]

    render_stress_video(
        deformation_frames,
        stress_frames,
        args.output_path,
        args.width,
        args.height
    )

    render_trajectory_video(
        deformation_frames,
        args.output_path,
        args.width,
        args.height
    )