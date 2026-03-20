import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _list_ply_files(model_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(model_dir)):
        if name.lower().endswith(".ply"):
            files.append(os.path.join(model_dir, name))
    if not files:
        raise FileNotFoundError(f"未在 model_path 找到 .ply: {model_dir}")
    return files


def _sample_from_space(spec: Any, rng: random.Random) -> Any:
    """
    支持 3 种参数空间写法：
    1) list: 从列表均匀采样一个
    2) {"type": "uniform", "min": a, "max": b}: 连续均匀
    3) {"type": "log_uniform", "min": a, "max": b}: 对数均匀（a,b>0）
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("空间列表为空")
        return rng.choice(spec)
    if isinstance(spec, dict):
        t = spec.get("type", "uniform")
        if t == "uniform":
            a = float(spec["min"])
            b = float(spec["max"])
            return a + (b - a) * rng.random()
        if t == "log_uniform":
            import math

            a = float(spec["min"])
            b = float(spec["max"])
            if a <= 0 or b <= 0:
                raise ValueError("log_uniform 需要 min/max > 0")
            la = math.log(a)
            lb = math.log(b)
            return float(math.exp(la + (lb - la) * rng.random()))
    # 直接常量
    return spec


def _grid_values_from_space(spec: Any, num: int) -> List[Any]:
    """
    将参数空间转换成“遍历用”的取值列表：
    - list: 直接返回该列表（保持顺序）
    - {"type":"uniform","min":a,"max":b}: 等间距 num 点（含端点，num=1 时取中点）
    - {"type":"log_uniform","min":a,"max":b}: 对数等间距 num 点（含端点，num=1 时取几何均值）
    - 常量: 返回 [常量]
    """
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        t = spec.get("type", "uniform")
        if t == "uniform":
            a = float(spec["min"])
            b = float(spec["max"])
            if num <= 1:
                return [0.5 * (a + b)]
            step = (b - a) / float(num - 1)
            return [a + i * step for i in range(num)]
        if t == "log_uniform":
            import math

            a = float(spec["min"])
            b = float(spec["max"])
            if a <= 0 or b <= 0:
                raise ValueError("log_uniform 需要 min/max > 0")
            if num <= 1:
                return [float(math.sqrt(a * b))]
            la = math.log(a)
            lb = math.log(b)
            step = (lb - la) / float(num - 1)
            return [float(math.exp(la + i * step)) for i in range(num)]
    return [spec]


def _filter_material_params_for_type(material_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留对该 material 有意义/被 solver 接受的参数。
    solver 接受的键来自 mpm_solver_warp.MPM_Simulator_WARP.set_parameters_dict
    """
    allowed_common = {
        "material",
        "grid_lim",
        "n_grid",
        "E",
        "nu",
        "density",
        "g",
        "rpic_damping",
        "plastic_viscosity",
        "softening",
        "grid_v_damping_scale",
        "hardening",
        "xi",
        "friction_angle",
        "yield_stress",
    }
    out = {k: v for k, v in params.items() if k in allowed_common}

    # material 专属参数约束（经验规则，避免无意义键）
    if material_type != "metal":
        out.pop("yield_stress", None)
        out.pop("hardening", None)
    if material_type != "sand":
        out.pop("friction_angle", None)
    return out


@dataclass(frozen=True)
class Job:
    ply_path: str
    sim_type: str
    material_params: Dict[str, Any]
    output_root: str
    num_views: int
    render_img: bool
    compile_video: bool
    output_deformation: bool
    output_stress: bool
    field_output_interval: int
    output_bc_info: bool
    output_force_info: bool
    white_bg: bool
    debug: bool


def _build_job_list(
    ply_files: List[str],
    material_space: Dict[str, Any],
    train_cfg: Dict[str, Any],
    rng: random.Random,
) -> List[Job]:
    sim_types = train_cfg.get("sim_types", ["bend", "drop", "press", "shear", "stretch"])
    n_total = int(train_cfg.get("num_simulations", 10))

    # 输出根目录默认固定为 PhysGaussian/auto_output（相对本脚本目录）
    output_root = train_cfg.get("output_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_output"))
    num_views = int(train_cfg.get("num_views", 1))
    render_img = bool(train_cfg.get("render_img", True))
    compile_video = bool(train_cfg.get("compile_video", True))
    output_deformation = bool(train_cfg.get("output_deformation", False))
    output_stress = bool(train_cfg.get("output_stress", False))
    field_output_interval = int(train_cfg.get("field_output_interval", 1))
    output_bc_info = bool(train_cfg.get("output_bc_info", True))
    output_force_info = bool(train_cfg.get("output_force_info", True))
    white_bg = bool(train_cfg.get("white_bg", False))
    debug = bool(train_cfg.get("debug", False))

    jobs: List[Job] = []

    mat_spaces = material_space.get("material_spaces", {})
    if not mat_spaces:
        raise ValueError("material_space_config 需要提供 material_spaces，按材质拆分参数空间。")

    material_values = sorted(mat_spaces.keys())

    # 完全随机采样：每次仿真独立随机选 model / sim_type / 材质类型 / 该材质的参数组合
    for _ in range(n_total):
        ply_path = rng.choice(ply_files)
        sim_type = rng.choice(sim_types)

        material_type = rng.choice(material_values)
        space = mat_spaces[material_type]

        sampled: Dict[str, Any] = {"material": material_type}
        for k, spec in space.items():
            sampled[k] = _sample_from_space(spec, rng)

        # 仍然跑一遍过滤器，防止 future 扩展时混入不支持的键
        sampled = _filter_material_params_for_type(material_type, sampled)

        jobs.append(
            Job(
                ply_path=ply_path,
                sim_type=sim_type,
                material_params=sampled,
                output_root=output_root,
                num_views=num_views,
                render_img=render_img,
                compile_video=compile_video,
                output_deformation=output_deformation,
                output_stress=output_stress,
                field_output_interval=field_output_interval,
                output_bc_info=output_bc_info,
                output_force_info=output_force_info,
                white_bg=white_bg,
                debug=debug,
            )
        )

    return jobs


def _make_run_config(
    base_cfg: Dict[str, Any],
    sim_type: str,
    material_params: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    # 直接覆盖 material 参数
    for k, v in material_params.items():
        # g 只对 drop 生效：非 drop 时不覆盖模板里的重力设置
        if k == "g" and sim_type != "drop":
            continue
        cfg[k] = v
    # boundary_conditions 会在 modified_simulation.py 内按 sim_type 自动生成
    cfg["boundary_conditions"] = cfg.get("boundary_conditions", [])
    return cfg


def _job_output_dir(job: Job) -> str:
    ply_stem = os.path.splitext(os.path.basename(job.ply_path))[0]
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            # 统一科学计数法，避免超长
            return f"{v:.3g}"
        return str(v)

    # 物理参数串：按 key 排序，便于稳定命名
    parts = []
    for k in sorted(job.material_params.keys()):
        parts.append(f"{k}={_fmt(job.material_params[k])}")
    phys_tag = "_".join(parts)
    phys_tag = phys_tag.replace(" ", "").replace("/", "_")

    # 目录结构：auto_output/<sim_type>/<obj__phys__sim_type>/
    folder_name = f"{ply_stem}__{phys_tag}__{job.sim_type}"
    return os.path.join(job.output_root, job.sim_type, folder_name)


def _run_one_job(
    idx: int,
    job: Job,
    cmd: List[str],
    out_dir: str,
    script_dir: str,
    run_env: Dict[str, str],
    gpu_id: int,
) -> Tuple[int, int, str]:
    """在指定 GPU 上跑一个任务，返回 (idx, returncode, out_dir)。"""
    env = dict(run_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.run(cmd, cwd=script_dir, env=env)
    return (idx, proc.returncode, out_dir)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="包含多个 .ply 的文件夹")
    parser.add_argument("--material_space_config", type=str, required=True, help="材质空间采样配置 JSON")
    parser.add_argument("--train_config", type=str, required=True, help="训练/批量运行配置 JSON")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="并行使用的 GPU 数量，每个任务绑定一张卡。默认 1（单卡顺序执行）。0 表示自动检测并使用全部可见 GPU。",
    )
    args = parser.parse_args()

    model_dir = args.model_path
    material_space = _read_json(args.material_space_config)
    train_cfg = _read_json(args.train_config)

    seed = int(train_cfg.get("seed", 0))
    rng = random.Random(seed)

    ply_files = _list_ply_files(model_dir)

    base_cfg_by_sim: Dict[str, str] = train_cfg.get("base_config_by_sim_type", {})
    if not base_cfg_by_sim:
        raise ValueError(
            "train_config 需要提供 base_config_by_sim_type，指向各 sim_type 的模板 config json（例如 press/drop/...）。"
        )

    jobs = _build_job_list(ply_files, material_space, train_cfg, rng)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    modified_sim = os.path.join(script_dir, "modified_simulation.py")

    # 子进程需能加载 torch 的 libc10.so 等，否则 simple_knn/diff_gaussian_rasterization 会报错
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    _run_env = os.environ.copy()
    _run_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [_torch_lib] + (_run_env.get("LD_LIBRARY_PATH") or "").split(os.pathsep)
    ).rstrip(os.pathsep)

    tmp_cfg_dir = os.path.join(train_cfg.get("output_root", os.path.join(script_dir, "auto_output")), "_tmp_configs")
    os.makedirs(tmp_cfg_dir, exist_ok=True)

    # 确定并行 GPU 数
    num_gpus = args.num_gpus
    if num_gpus == 0:
        num_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
        print(f"自动检测到 {num_gpus} 张 GPU，将并行运行。")
    if num_gpus < 1:
        num_gpus = 1

    # 为所有 job 生成配置并构建 cmd
    tasks: List[Tuple[int, Job, List[str], str]] = []
    for idx, job in enumerate(jobs):
        if job.sim_type not in base_cfg_by_sim:
            raise ValueError(f"base_config_by_sim_type 缺少 sim_type={job.sim_type}")
        base_cfg = _read_json(base_cfg_by_sim[job.sim_type])
        run_cfg = _make_run_config(base_cfg, job.sim_type, job.material_params)

        out_dir = _job_output_dir(job)
        cfg_path = os.path.join(tmp_cfg_dir, f"{idx:06d}_{job.sim_type}.json")
        _write_json(cfg_path, run_cfg)

        cmd = [
            sys.executable,
            modified_sim,
            "--ply_path",
            job.ply_path,
            "--config",
            cfg_path,
            "--output_path",
            out_dir,
            "--sim_type",
            job.sim_type,
            "--num_views",
            str(job.num_views),
            "--field_output_interval",
            str(job.field_output_interval),
        ]
        if job.render_img:
            cmd.append("--render_img")
        if job.compile_video:
            cmd.append("--compile_video")
        if job.output_deformation:
            cmd.append("--output_deformation")
        if job.output_stress:
            cmd.append("--output_stress")
        if job.output_bc_info:
            cmd.append("--output_bc_info")
        if job.output_force_info:
            cmd.append("--output_force_info")
        if job.white_bg:
            cmd.append("--white_bg")
        if job.debug:
            cmd.append("--debug")

        tasks.append((idx, job, cmd, out_dir))

    if num_gpus <= 1:
        # 单卡顺序执行（与原逻辑一致）
        for idx, job, cmd, out_dir in tasks:
            print(f"\n[{idx+1}/{len(jobs)}] Running: {os.path.basename(job.ply_path)} | {job.sim_type} | {job.material_params}")
            print(f"  out: {out_dir}")
            proc = subprocess.run(cmd, cwd=script_dir, env=_run_env)
            if proc.returncode != 0:
                raise RuntimeError(f"Job failed (exit={proc.returncode}): {cmd}")
    else:
        # 多卡并行：主进程直接用 Popen 启动子进程，每个子进程显式设置 CUDA_VISIBLE_DEVICES，避免 fork/进程池导致只跑在一张卡上
        gpu_task_lists: List[List[Tuple[int, Job, List[str], str]]] = [[] for _ in range(num_gpus)]
        for idx, job, cmd, out_dir in tasks:
            gpu_task_lists[idx % num_gpus].append((idx, job, cmd, out_dir))

        print(f"\n多卡并行: 使用 {num_gpus} 张 GPU，共 {len(tasks)} 个任务。")
        for gpu_id in range(num_gpus):
            n = len(gpu_task_lists[gpu_id])
            print(f"  GPU {gpu_id}: {n} 个任务")

        # active[gpu_id] = (popen, idx, out_dir)
        active: Dict[int, Tuple[subprocess.Popen, int, str]] = {}
        gpu_next: List[int] = [0] * num_gpus  # 每个 GPU 下一个要执行的任务在 gpu_task_lists[gpu_id] 中的下标
        failed: List[Tuple[int, int, str]] = []

        def start_one(gpu_id: int) -> bool:
            """若该 GPU 还有任务则启动一个，返回是否已启动。"""
            if gpu_id in active:
                return False
            lst = gpu_task_lists[gpu_id]
            i = gpu_next[gpu_id]
            if i >= len(lst):
                return False
            idx, job, cmd, out_dir = lst[i]
            gpu_next[gpu_id] = i + 1
            # 用系统 env 在命令行强制设置 CUDA_VISIBLE_DEVICES，避免子进程继承或未生效
            full_cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu_id}", sys.executable] + cmd[1:]
            popen = subprocess.Popen(full_cmd, cwd=script_dir, env=_run_env)
            active[gpu_id] = (popen, idx, out_dir)
            print(f"  [GPU {gpu_id}] 启动任务 {idx+1}/{len(tasks)}: {os.path.basename(job.ply_path)} | {job.sim_type}")
            return True

        # 先为每张卡各启动一个任务
        for gpu_id in range(num_gpus):
            start_one(gpu_id)

        while active:
            for gpu_id in list(active.keys()):
                popen, idx, out_dir = active[gpu_id]
                ret = popen.poll()
                if ret is not None:
                    del active[gpu_id]
                    if ret != 0:
                        failed.append((idx, ret, out_dir))
                    if not start_one(gpu_id):
                        pass  # 该 GPU 无更多任务
            time.sleep(0.2)

        if failed:
            for idx, ret, out_dir in failed:
                print(f"失败: job_idx={idx} returncode={ret} out={out_dir}", file=sys.stderr)
            raise RuntimeError(f"有 {len(failed)} 个任务失败，见上方 stderr。")


if __name__ == "__main__":
    run()

