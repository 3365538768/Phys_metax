import os
import re
import gc
import glob
import json
import shutil
import tempfile
import argparse
import subprocess

import numpy as np
import cv2
from tqdm import tqdm

# 无显示/headless 时避免 Filament 抱怨 XDG_RUNTIME_DIR（须在 import open3d 之前）
if not os.environ.get("XDG_RUNTIME_DIR"):
    _xdg = os.path.join(tempfile.gettempdir(), f"o3d_xdg_{os.getpid()}")
    os.makedirs(_xdg, exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = _xdg

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


def load_deformation_frame_indices(folder):
    """
    与 load_deformation_frames 相同排序，解析文件名中的仿真帧号（与 Gaussian current_frame 一致）。
    """
    files = sorted(glob.glob(os.path.join(folder, "deformation_frame_*.npz")))
    ids = []
    for fp in files:
        m = re.search(r"deformation_frame_(\d+)\.npz$", os.path.basename(fp))
        if m:
            ids.append(int(m.group(1)))
        else:
            ids.append(len(ids))
    return ids


def load_stress_frames(folder):

    files = sorted(glob.glob(os.path.join(folder,"stress_frame_*.npz")))

    frames = []

    for f in files:
        frames.append(np.load(f))

    return frames


def mpm_positions_to_world_numpy(pos, spec):
    """
    与 modified_simulation 中 pos_world = undo_all_transforms(particle_x) 一致。
    deformation_field / stress_field 里存的是 MPM 空间坐标；3DGS 渲染用世界坐标。
    """
    if spec is None:
        return np.asarray(pos, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64).reshape(-1, 3).copy()
    x -= np.array([1.0, 1.0, 1.0], dtype=np.float64)
    scale = float(spec["scale_origin"])
    om = np.asarray(spec["original_mean_pos"], dtype=np.float64).reshape(3)
    x = om + x / scale
    mats = spec.get("rotation_matrices") or []
    for mi in range(len(mats) - 1, -1, -1):
        R = np.asarray(mats[mi], dtype=np.float64).reshape(3, 3)
        x = x @ R
    return x


def deformation_frames_to_world_positions(frames, spec):
    """返回新帧列表（dict），position 为世界系；其它键原样复制。"""
    if spec is None:
        return frames
    out = []
    for fr in frames:
        d = {k: np.asarray(fr[k]) for k in fr.files}
        d["position"] = mpm_positions_to_world_numpy(d["position"], spec)
        out.append(d)
    return out


def observant_matrix_from_world_up(world_up):
    """与 utils.generate_local_coord 一致，由世界「上」方向得到 3x3 observant_coordinates。"""
    wu = np.asarray(world_up, dtype=np.float64).reshape(3)
    wu = wu / (np.linalg.norm(wu) + 1e-12)
    h1 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if abs(np.dot(h1, wu)) < 0.01:
        h1 = np.array([0.72, 0.37, -0.67], dtype=np.float64)
    h1 = h1 - np.dot(h1, wu) * wu
    h1 = h1 / (np.linalg.norm(h1) + 1e-12)
    h2 = np.cross(h1, wu)
    return np.column_stack((h1, h2, wu))


def try_load_stress_pcd_meta(simulation_dir):
    p = os.path.join(simulation_dir, "meta", "stress_pcd_cameras.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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

def render_stress_video(
    deformation_frames,
    stress_frames,
    output_path,
    width,
    height,
    compose_video=True,
    world_up=None,
    fov_vertical_deg=60.0,
):

    if compose_video:

        tmp = os.path.join(output_path, "tmp_stress")

        if os.path.exists(tmp):

            shutil.rmtree(tmp)

        os.makedirs(tmp)

    else:

        tmp = os.path.join(output_path, "stress_heatmap_frames")

        os.makedirs(tmp, exist_ok=True)

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

    emax=float(np.max(extent))

    if emax < 1e-12:

        emax = 1.0

    eye=center+np.array([1.5,1.5,1.5],dtype=np.float64)*emax

    if world_up is None:

        world_up=np.array([0.0,0.0,1.0],dtype=np.float64)

    _apply_o3d_camera_from_meta(
        scene, center, eye, world_up, width, height, fov_vertical_deg
    )

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

        cv2.imwrite(frame_path, img_np)

    del renderer

    gc.collect()

    if compose_video:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "20",
                "-i",
                os.path.join(tmp, "%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                os.path.join(output_path, "stress_heatmap.mp4"),
            ]
        )

    else:

        print("单视角应力 PNG 已写入:", tmp, "（未调用 ffmpeg）")


def load_stress_pcd_camera_meta(path):

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


def _apply_o3d_camera_from_meta(scene, look_at, eye, up, width, height, fov_vertical_deg):

    cam = scene.camera

    center = np.asarray(look_at, dtype=np.float64).reshape(3)

    e = np.asarray(eye, dtype=np.float64).reshape(3)

    u = np.asarray(up, dtype=np.float64).reshape(3)

    un = np.linalg.norm(u)

    if un < 1e-12:

        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    else:

        u = u / un

    cam.look_at(center, e, u)

    aspect = float(width) / float(max(int(height), 1))

    fov = float(fov_vertical_deg)

    near_v, far_v = 0.1, 100.0

    for attr in ("Fov", "FovType"):

        if hasattr(rendering.Camera, attr):

            F = getattr(rendering.Camera, attr)

            if hasattr(F, "Vertical"):

                try:

                    cam.set_projection(fov, aspect, near_v, far_v, F.Vertical)

                    return

                except Exception:

                    break

    try:

        cam.set_projection(fov, aspect, near_v, far_v, rendering.Camera.Fov.Vertical)

    except Exception:

        pass


def _eye_for_view_sim_frame(camera_meta, vinfo, sim_frame, look_at_vc, obs_np):
    """
    与 get_camera_view(..., move_camera=True, current_frame=sim_frame) 一致。
    meta v1 无 camera_motion 时退回 views[].eye。
    """
    motion = camera_meta.get("camera_motion") or {}
    if not bool(motion.get("move_camera")):
        return np.asarray(vinfo["eye"], dtype=np.float64).reshape(3)
    try:
        from utils.camera_view_utils import get_camera_position_and_rotation
    except ImportError:
        print(
            "[stress_pcd] 无法 import utils.camera_view_utils，move_camera 时仍用静态 eye"
        )
        return np.asarray(vinfo["eye"], dtype=np.float64).reshape(3)
    da = float(motion.get("delta_a", 0.0))
    de = float(motion.get("delta_e", 0.0))
    dr = float(motion.get("delta_r", 0.0))
    rad0 = float(camera_meta.get("init_radius", 2.0))
    az0 = float(vinfo["azimuth"])
    el0 = float(vinfo["elevation"])
    sf = int(sim_frame)
    az = az0 + sf * da
    el = el0 + sf * de
    rad = rad0 + sf * dr
    eye, _R = get_camera_position_and_rotation(az, el, rad, look_at_vc, obs_np)
    return np.asarray(eye, dtype=np.float64).reshape(3)


def render_stress_video_multiview(
    deformation_frames,
    stress_frames,
    output_path,
    width,
    height,
    camera_meta,
    compose_video=True,
    deformation_folder=None,
    mpm_fallback_camera=False,
):

    """
    与 Gaussian synthetic 轨道相机一致的多视角点云应力图（von Mises 着色逻辑同 render_stress_video）。
    camera_meta: load_stress_pcd_camera_meta 的 dict（通常来自 meta/stress_pcd_cameras.json）。
    compose_video: True 时在**销毁 Open3D 之后**再调 ffmpeg（降低段错误概率）。
    deformation_folder: 用于解析 deformation_frame_XXXX 中的帧号以对齐 move_camera。
    mpm_fallback_camera: True 时 npz 仍为 MPM 坐标，用 mpm_space_viewpoint_center（或质心）+
        与 config 相同的 az/el/radius 在 MPM 系下重建相机（旧 meta 无 mpm_to_world 时用）。
    """

    for w in camera_meta.get("warnings") or []:

        print("[stress_pcd_cameras] warning:", w)

    look_at = np.asarray(camera_meta["look_at"], dtype=np.float64).reshape(3)

    world_up = np.asarray(camera_meta["world_up"], dtype=np.float64)

    fov_v = float(camera_meta.get("fov_vertical_deg", 60.0))

    views = camera_meta["views"]

    views_work = [dict(v) for v in views]

    if mpm_fallback_camera:

        vpc = camera_meta.get("mpm_space_viewpoint_center")
        if vpc is not None:
            look_at = np.asarray(vpc, dtype=np.float64).reshape(3)
        else:
            look_at = (
                np.concatenate(
                    [f["position"] for f in deformation_frames], axis=0
                )
                .mean(0)
                .reshape(3)
            )
        obs_np_fb = observant_matrix_from_world_up([0.0, 0.0, 1.0])
        world_up = obs_np_fb[:, 2].astype(np.float64).copy()
        try:
            from utils.camera_view_utils import get_camera_position_and_rotation

            rad0 = float(camera_meta.get("init_radius", 2.0))
            for i, v in enumerate(views_work):
                eye, _R = get_camera_position_and_rotation(
                    float(v["azimuth"]),
                    float(v["elevation"]),
                    rad0,
                    look_at,
                    obs_np_fb,
                )
                views_work[i]["eye"] = (
                    np.asarray(eye, dtype=np.float64).reshape(3).tolist()
                )
        except ImportError:
            print(
                "[stress_pcd] 无法 import camera_view_utils，多视角 MPM 回退可能不准"
            )

    T = min(len(deformation_frames), len(stress_frames))

    all_vm = []

    for i in range(T):

        vm = von_mises(stress_frames[i]["stress_cauchy"])

        all_vm.append(vm)

    all_vm = np.concatenate(all_vm)

    all_vm = np.log(all_vm + 1e-6)

    vmin = np.percentile(all_vm, 1)

    vmax = np.percentile(all_vm, 99)

    if deformation_folder and os.path.isdir(deformation_folder):

        frame_ids = load_deformation_frame_indices(deformation_folder)[:T]

    else:

        frame_ids = list(range(T))

    if len(frame_ids) < T:

        print(
            f"[stress_pcd] 警告: 帧号解析数量 {len(frame_ids)} < T={T}，不足部分用序号代替"
        )

        frame_ids = frame_ids + [j for j in range(len(frame_ids), T)]

    if mpm_fallback_camera:

        obs_np = observant_matrix_from_world_up([0.0, 0.0, 1.0])

    elif camera_meta.get("observant_coordinates") is not None:

        obs_np = np.asarray(
            camera_meta["observant_coordinates"], dtype=np.float64
        ).reshape(3, 3)

    else:

        wu = np.asarray(camera_meta["world_up"], dtype=np.float64).reshape(3)
        obs_np = observant_matrix_from_world_up(wu)

    out_root = os.path.join(output_path, "stress_pcd_multiview")

    os.makedirs(out_root, exist_ok=True)

    view_output_dirs = []

    mat = rendering.MaterialRecord()

    mat.shader = "defaultUnlit"

    mat.point_size = 6

    renderer = rendering.OffscreenRenderer(width, height)

    scene = renderer.scene

    scene.set_background([1, 1, 1, 1])

    try:

        for vinfo in views_work:

            name = vinfo["name"]

            if compose_video:

                tmp = os.path.join(out_root, f"tmp_{name}")

                if os.path.exists(tmp):

                    shutil.rmtree(tmp)

                os.makedirs(tmp)

                out_dir = tmp

            else:

                out_dir = os.path.join(out_root, name)

                os.makedirs(out_dir, exist_ok=True)

            view_output_dirs.append((name, out_dir))

            for i in tqdm(range(T), desc=f"stress_pcd {name}"):

                sim_f = frame_ids[i]

                eye = _eye_for_view_sim_frame(
                    camera_meta, vinfo, sim_f, look_at, obs_np
                )

                _apply_o3d_camera_from_meta(
                    scene, look_at, eye, world_up, width, height, fov_v
                )

                if scene.has_geometry("pcd"):

                    scene.remove_geometry("pcd")

                pos = deformation_frames[i]["position"]

                stress = stress_frames[i]["stress_cauchy"]

                vm = von_mises(stress)

                vm = np.log(vm + 1e-6)

                col = (vm - vmin) / (vmax - vmin + 1e-8)

                col = np.clip(col, 0, 1)

                colors = np.stack(
                    [col, np.zeros_like(col), 1 - col],
                    axis=1,
                )

                pcd = create_point_cloud(pos, colors)

                scene.add_geometry("pcd", pcd, mat)

                img = renderer.render_to_image()

                img_np = np.asarray(img)

                if img_np.size == 0:

                    print("WARNING empty frame", name, i)

                    continue

                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img_np)

    finally:

        del renderer

        gc.collect()

    for name, out_dir in view_output_dirs:

        if compose_video:

            mp4_path = os.path.join(out_root, f"stress_heatmap_{name}.mp4")

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "20",
                    "-i",
                    os.path.join(out_dir, "%04d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    mp4_path,
                ]
            )

            shutil.rmtree(out_dir)

        else:

            print("视角", name, "PNG 序列:", out_dir)

    print(
        "多视角应力点云已写入:",
        out_root,
        "（mp4）" if compose_video else "（仅 PNG，未合成视频）",
    )


# =====================================
# trajectory video
# =====================================

def render_trajectory_video(
    frames,
    output_path,
    width,
    height,
    compose_video=True,
    world_up=None,
    fov_vertical_deg=60.0,
):

    if compose_video:

        tmp = os.path.join(output_path, "tmp_traj")

        if os.path.exists(tmp):

            shutil.rmtree(tmp)

        os.makedirs(tmp)

    else:

        tmp = os.path.join(output_path, "deformation_trajectory_frames")

        os.makedirs(tmp, exist_ok=True)

    T=len(frames)

    N=len(frames[0]["position"])

    all_pos=np.zeros((T,N,3))

    for i,f in enumerate(frames):
        all_pos[i]=f["position"]

    traj=create_trajectory_lines(all_pos)

    renderer=rendering.OffscreenRenderer(width,height)

    scene=renderer.scene

    scene.set_background([1,1,1,1])

    flat=all_pos.reshape(-1,3)

    center=flat.mean(0)

    extent=flat.max(0)-flat.min(0)

    emax=float(np.max(extent))

    if emax < 1e-12:

        emax = 1.0

    eye=center+np.array([2.5,2.5,2.5],dtype=np.float64)*emax

    if world_up is None:

        world_up=np.array([0.0,0.0,1.0],dtype=np.float64)

    _apply_o3d_camera_from_meta(
        scene, center, eye, world_up, width, height, fov_vertical_deg
    )

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

        cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"), img_np)

    del renderer

    gc.collect()

    if compose_video:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "20",
                "-i",
                os.path.join(tmp, "%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                os.path.join(output_path, "deformation_trajectory.mp4"),
            ]
        )

    else:

        print("轨迹 PNG 已写入:", tmp, "（未调用 ffmpeg）")


# =====================================
# main
# =====================================

if __name__=="__main__":

    parser=argparse.ArgumentParser(
        epilog=(
            "多视角示例: python my_utils/visualize_fields.py --simulation_dir RUN --output_path OUT --multiview"
            "（默认只写 PNG；要 mp4 在同一行末尾加 --ffmpeg）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--simulation_dir",required=True)

    parser.add_argument("--output_path",required=True)

    parser.add_argument("--test",action="store_true")

    parser.add_argument("--test_frames",type=int,default=10)

    parser.add_argument("--width",type=int,default=1280)

    parser.add_argument("--height",type=int,default=960)

    parser.add_argument(
        "--multiview",
        action="store_true",
        help="使用 meta/stress_pcd_cameras.json 与 Gaussian 一致的环绕相机渲染多视角点云应力图",
    )

    parser.add_argument(
        "--camera_meta",
        default=None,
        help="应力点云相机 meta JSON；默认 <simulation_dir>/meta/stress_pcd_cameras.json",
    )

    parser.add_argument(
        "--no_ffmpeg",
        action="store_true",
        help="禁止一切 ffmpeg（单视角应力、多视角应力、轨迹）",
    )

    parser.add_argument(
        "--ffmpeg",
        action="store_true",
        help="多视角应力：额外合成 mp4（默认多视角只写 PNG，避免段错误；须与 --multiview 同用）",
    )

    args = parser.parse_args()

    if args.multiview:

        compose_multiview = bool(args.ffmpeg) and (not args.no_ffmpeg)

        if not compose_multiview and not args.no_ffmpeg:

            print(
                "[stress_pcd] 多视角默认仅写 PNG；需要 mp4 请加 --ffmpeg（须与 --multiview 在同一行，勿换行单独执行）。"
            )

    else:

        compose_multiview = not args.no_ffmpeg

    compose_single_and_traj = not args.no_ffmpeg

    os.makedirs(args.output_path,exist_ok=True)

    deformation_folder=os.path.join(args.simulation_dir,"deformation_field")

    stress_folder=os.path.join(args.simulation_dir,"stress_field")

    meta_default = os.path.join(
        args.simulation_dir, "meta", "stress_pcd_cameras.json"
    )
    meta_path = args.camera_meta if args.camera_meta else meta_default

    if args.multiview and not os.path.isfile(meta_path):

        raise FileNotFoundError(
            f"未找到相机 meta: {meta_path}\n"
            "请先运行带 --output_deformation / --output_stress 的 modified_simulation 生成该文件，"
            "或用 --camera_meta 指定。"
        )

    camera_meta = None
    if os.path.isfile(meta_path):
        camera_meta = load_stress_pcd_camera_meta(meta_path)

    mpm_tw = camera_meta.get("mpm_to_world") if camera_meta else None

    deformation_frames=load_deformation_frames(deformation_folder)

    stress_frames=load_stress_frames(stress_folder)

    if args.test:

        deformation_frames=deformation_frames[:args.test_frames]

        stress_frames=stress_frames[:args.test_frames]

    if mpm_tw:

        deformation_frames = deformation_frames_to_world_positions(
            deformation_frames, mpm_tw
        )
        print(
            "[visualize_fields] 已对 deformation 点云应用 mpm_to_world，"
            "与 3DGS / stress_pcd_cameras 世界系一致。"
        )

    fov_gs = 60.0
    world_up_vis = None
    if camera_meta is not None:
        fov_gs = float(camera_meta.get("fov_vertical_deg", 60.0))
        if mpm_tw:
            world_up_vis = np.asarray(
                camera_meta.get("world_up", [0.0, 0.0, 1.0]), dtype=np.float64
            )

    mpm_fallback_mv = bool(
        args.multiview and camera_meta is not None and mpm_tw is None
    )
    if mpm_fallback_mv:
        print(
            "[visualize_fields] 警告: meta 无 mpm_to_world（请重新跑仿真生成 v3），"
            "多视角将使用 MPM 系相机回退。"
        )

    if args.multiview:

        render_stress_video_multiview(
            deformation_frames,
            stress_frames,
            args.output_path,
            args.width,
            args.height,
            camera_meta,
            compose_video=compose_multiview,
            deformation_folder=deformation_folder,
            mpm_fallback_camera=mpm_fallback_mv,
        )

    else:

        render_stress_video(
            deformation_frames,
            stress_frames,
            args.output_path,
            args.width,
            args.height,
            compose_video=compose_single_and_traj,
            world_up=world_up_vis,
            fov_vertical_deg=fov_gs,
        )

    render_trajectory_video(
        deformation_frames,
        args.output_path,
        args.width,
        args.height,
        compose_video=compose_single_and_traj,
        world_up=world_up_vis,
        fov_vertical_deg=fov_gs,
    )