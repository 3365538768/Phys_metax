#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按顺序跑两类实验：
1) 实验1：小样本 overfit（只看 train）
2) 实验2：输入消融（只看 train）

默认：
- overfit_num_samples=32
- eval 只取 1 个样本做快速检查（sample_limit=1）

用法示例：
  python3 run_overfit_32_ablation.py --epochs 50 --nproc_per_node 1
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import socket
import os
import signal
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_if_exists(d: Dict[str, Any], key_path: List[str], value: Any) -> None:
    cur: Any = d
    for k in key_path[:-1]:
        if not isinstance(cur, dict):
            return
        if k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur[key_path[-1]] = value


def _ensure_path(d: Dict[str, Any], key_path: List[str]) -> None:
    cur: Any = d
    for k in key_path:
        if not isinstance(cur, dict):
            return
        if k not in cur:
            cur[k] = {}
        cur = cur[k]


def _set_path(d: Dict[str, Any], key_path: List[str], value: Any) -> None:
    _ensure_path(d, key_path[:-1])
    cur: Any = d
    for k in key_path[:-1]:
        cur = cur[k]
    if isinstance(cur, dict):
        cur[key_path[-1]] = value


def build_train_cmd(*, nproc_per_node: int, config_path: Path) -> List[str]:
    # torchrun 会自动提供 WORLD_SIZE/RANK 等环境变量
    return [
        "torchrun",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        "29500",
        "-m",
        "my_model.train",
        "--config",
        str(config_path),
    ]


def _get_free_port() -> int:
    """在本机找一个尽量空闲的 TCP 端口号。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _terminate_process_group(proc: subprocess.Popen, *, grace_sigint_s: float = 3.0, grace_sigterm_s: float = 3.0) -> None:
    """
    关闭整个子进程组，避免 torchrun/worker 残留：
    1) SIGINT（模拟 Ctrl+C）
    2) 超时后 SIGTERM
    3) 再超时 SIGKILL
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        return

    def _wait(timeout_s: float) -> bool:
        end_t = time.time() + timeout_s
        while time.time() < end_t:
            if proc.poll() is not None:
                return True
            time.sleep(0.1)
        return proc.poll() is not None

    try:
        os.killpg(pgid, signal.SIGINT)
    except Exception:
        pass
    if _wait(grace_sigint_s):
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        pass
    if _wait(grace_sigterm_s):
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass


def _is_port_listening(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        return s.connect_ex(("127.0.0.1", int(port))) == 0
    except Exception:
        return False
    finally:
        s.close()


def _wait_port_released(port: int, timeout_s: float = 8.0) -> bool:
    end_t = time.time() + float(timeout_s)
    while time.time() < end_t:
        if not _is_port_listening(port):
            return True
        time.sleep(0.2)
    return not _is_port_listening(port)


def _expected_in_channels(input_mode: str) -> int:
    im = str(input_mode).strip().lower()
    if im in ("rgb", "rgb_only", "rgbonly"):
        return 3
    if im in ("rgb+flow", "rgb_flow", "rgb+stress", "rgb_stress", "rgb+force_mask", "rgb_force_mask", "rgb+mask"):
        return 6
    if im in ("rgb+stress+flow", "rgb_stress_flow", "rgbstressflow"):
        return 9
    # 兜底：保持 6（与旧默认一致）
    return 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50, help="每个实验训练 epoch 数")
    ap.add_argument("--nproc_per_node", type=int, default=1, help="单卡=1，多卡>1")
    ap.add_argument("--overfit_num_samples", type=int, default=32, help="overfit 样本数（从 train_ids 抽取）")
    # -1 表示继承 configs.json 里的 quick_eval 配置，不要强行覆盖
    ap.add_argument("--quick_eval_every", type=int, default=-1, help="覆盖 quick_eval.every_epochs；-1 表示继承 configs.json")
    ap.add_argument("--quick_eval_sample_limit", type=int, default=-1, help="覆盖 quick_eval.sample_limit；-1 表示继承 configs.json")
    ap.add_argument("--master_port", type=int, default=0, help="torchrun 的 master_port；0=自动选择空闲端口")
    ap.add_argument(
        "--eval_test",
        action="store_true",
        help="是否同时在 test_ids 做 quick eval（默认只跑 train）",
    )
    ap.add_argument(
        "--disable_quick_eval",
        action="store_true",
        help="完全关闭 quick_eval（只看 train loss 不做 eval）",
    )
    args = ap.parse_args()

    phys_root = Path(__file__).resolve().parent  # PhysGaussian/
    base_cfg_path = phys_root / "my_model" / "configs.json"
    base_cfg = _load_json(base_cfg_path)

    out_dir = phys_root / "output_configs_overfit_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = base_cfg.get("model") or {}
    default_input_mode = str(base_model.get("input_mode", "rgb+force_mask"))
    default_in_channels = int(base_model.get("in_channels", 6))
    # baseline 兜底：若 base configs 的 in_channels 与 input_mode 不一致，则以 input_mode 推导的为准
    exp_ch = _expected_in_channels(default_input_mode)
    if default_in_channels != exp_ch:
        print(
            f"[warn] base configs mismatch: input_mode={default_input_mode!r} in_channels={default_in_channels} -> use {exp_ch}",
            flush=True,
        )
        default_in_channels = exp_ch

    experiments = []

    # 实验1：baseline overfit（沿用当前 configs.json 的 input_mode/in_channels）
    experiments.append(
        {
            "name": f"exp1_overfit_baseline_{default_input_mode.replace('+', 'plus')}",
            "input_mode": default_input_mode,
            "in_channels": default_in_channels,
        }
    )

    # 实验2：输入消融
    experiments.extend(
        [
            {"name": "exp2_rgb_plus_flow", "input_mode": "rgb+flow", "in_channels": 6},
            {"name": "exp2_rgb_plus_stress", "input_mode": "rgb+stress", "in_channels": 6},
            {"name": "exp2_rgb_plus_stress_plus_flow", "input_mode": "rgb+stress+flow", "in_channels": 9},
        ]
    )

    for exp in experiments:
        exp_name: str = exp["name"]
        print(f"\n========== Running: {exp_name} ==========")

        cfg = copy.deepcopy(base_cfg)

        # 基础训练设置：overfit & 训练时长
        _set_path(cfg, ["train", "overfit_num_samples"], int(args.overfit_num_samples))
        _set_path(cfg, ["train", "epochs"], int(args.epochs))

        # 降低 checkpoint 磁盘压力：只保留 last.pt（quick eval 用它）
        _set_path(cfg, ["train", "checkpoint", "save_every_epochs"], 0)
        _set_path(cfg, ["train", "checkpoint", "save_last"], True)
        _set_path(cfg, ["train", "checkpoint", "save_best"], False)

        # quick eval 配置：默认只跑 train；可选跑 test
        if args.disable_quick_eval:
            _set_path(cfg, ["train", "quick_eval", "enabled"], False)
        else:
            # 继承 base configs.json：只有在显式传参时才覆盖
            if args.quick_eval_every >= 0:
                _set_path(cfg, ["train", "quick_eval", "enabled"], True)
                _set_path(cfg, ["train", "quick_eval", "every_epochs"], int(args.quick_eval_every))
            if args.quick_eval_sample_limit >= 0:
                _set_path(cfg, ["train", "quick_eval", "enabled"], True)
                _set_path(cfg, ["train", "quick_eval", "sample_limit"], int(args.quick_eval_sample_limit))

            # 若你希望额外跑 test，而 base configs.json 里 test_ids_json 为空，则补上
            if args.eval_test:
                cur_test_ids = None
                if isinstance(cfg.get("train"), dict) and isinstance(cfg["train"].get("quick_eval"), dict):
                    cur_test_ids = cfg["train"]["quick_eval"].get("test_ids_json")
                if cur_test_ids is None:
                    _set_path(cfg, ["train", "quick_eval", "test_ids_json"], "train_test_split.json")

            # 确保 train_ids_json 至少存在（否则 train_ids / test_ids 取样会为空）
            cur_train_ids = None
            if isinstance(cfg.get("train"), dict) and isinstance(cfg["train"].get("quick_eval"), dict):
                cur_train_ids = cfg["train"]["quick_eval"].get("train_ids_json")
            if cur_train_ids is None:
                _set_path(cfg, ["train", "quick_eval", "train_ids_json"], "train_test_split.json")

        # 输入消融：只改变 model.input_mode 和 model.in_channels
        _set_path(cfg, ["model", "input_mode"], exp["input_mode"])
        # 再做一次一致性校验（防止手动改 base configs 造成矛盾）
        want_mode = str(exp["input_mode"])
        want_ch = int(exp["in_channels"])
        exp_ch = _expected_in_channels(want_mode)
        if want_ch != exp_ch:
            print(
                f"[warn] exp mismatch: {exp_name} input_mode={want_mode!r} in_channels={want_ch} -> use {exp_ch}",
                flush=True,
            )
            want_ch = exp_ch
        _set_path(cfg, ["model", "in_channels"], int(want_ch))

        exp_cfg_path = out_dir / f"configs_{exp_name}.json"
        _save_json(exp_cfg_path, cfg)

        master_port = int(args.master_port)
        if master_port <= 0:
            master_port = _get_free_port()
        # 为每个 exp 选择不同 port，避免串行/残留进程干扰
        # 使用 python -m torch.distributed.run（比依赖 PATH 里的 torchrun 更稳）
        # 单机场景下用 --standalone，避免 hostname 解析导致的 c10d 初始化卡顿。
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(int(args.nproc_per_node)),
            "--master_port",
            str(master_port),
            "-m",
            "my_model.train",
            "--config",
            str(exp_cfg_path),
        ]
        print("CMD:", " ".join(cmd))

        # 训练过程完全交给 torchrun；这里串行跑，避免显卡互抢
        # 使用独立进程组，确保 Ctrl+C 时能清掉 torchrun + 全部 worker
        proc = subprocess.Popen(
            cmd,
            cwd=str(phys_root),
            preexec_fn=os.setsid,
            env={
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                # 若用户未手动指定网卡，则默认走 loopback，避免 hostname 解析失败导致的卡住
                "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "lo"),
            },
        )
        rc: Optional[int] = None
        try:
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
        except KeyboardInterrupt:
            print("\n[runner] KeyboardInterrupt received, stopping current experiment...", flush=True)
            _terminate_process_group(proc)
            raise SystemExit(130)
        except Exception:
            _terminate_process_group(proc)
            raise
        finally:
            # 成功/失败都做一次兜底清理（已退出时 no-op）
            _terminate_process_group(proc)

        # 前一个实验结束后确认资源状态（固定端口模式）
        if int(args.master_port) > 0:
            ok = _wait_port_released(int(args.master_port), timeout_s=8.0)
            if not ok:
                print(
                    f"[runner][warn] master_port {int(args.master_port)} may still be in use before next experiment",
                    flush=True,
                )

        print(f"[runner] finished {exp_name} (rc={rc}), resources released.", flush=True)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()

