#!/usr/bin/env python3
"""
扫描 auto_output 下所有仿真结果目录，解析文件夹名中的参数或 gt_parameters.json，
并输出 E / nu / density / yield_stress / hardening / softening 的分布统计图。

输出：
- auto_output/stats/params_hist_E.png
- auto_output/stats/params_hist_nu.png
- auto_output/stats/params_hist_density.png
- auto_output/stats/params_hist_yield_stress.png
- auto_output/stats/params_hist_hardening.png
- auto_output/stats/params_hist_softening.png
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt


# 只考虑这几种动作的子目录
ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")

# 需要统计的参数
TARGET_KEYS = ("E", "nu", "density", "yield_stress", "hardening", "softening")


def _parse_params_to_dict(params_str: str) -> dict:
    """将 E=1.62e+04_density=1.28e+03_... 解析为 {key: value}，兼容 key 中含下划线（如 yield_stress）。"""
    if not params_str.strip():
        return {}
    tokens = params_str.split("_")
    result = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" in t:
            # 形如 key=value
            k, _, v = t.partition("=")
            result[k] = v
            i += 1
        else:
            # t 本身不含 '='，有两种情况：
            # 1) 下一个也不含 '='：当成 key/value 一对（旧格式）
            # 2) 下一个含 '='：认为这是带下划线的 key 前半段，例如 'yield' + 'stress=1.2e3'
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if "=" not in nxt:
                    # 情况 1：t, nxt 组成一对
                    result[t] = nxt
                    i += 2
                else:
                    # 情况 2：拼成复合 key
                    k2, _, v2 = nxt.partition("=")
                    full_key = f"{t}_{k2}"
                    result[full_key] = v2
                    i += 2
            else:
                i += 1
    return result


def _accumulate_from_gt_json(gt_path: Path, counts_by_action: Counter, values: defaultdict) -> None:
    """从 gt_parameters.json 读取 sim_type 与 material_params。"""
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    action = gt.get("sim_type", "")
    if action not in ACTION_NAMES:
        return
    counts_by_action[action] += 1
    mp = gt.get("material_params") or {}
    for key in TARGET_KEYS:
        if key not in mp:
            continue
        try:
            v = float(mp[key])
        except (TypeError, ValueError):
            continue
        values[key].append(v)


def _parse_one_run_folder(name: str, counts_by_action: Counter, values: defaultdict) -> None:
    """解析名为 OBJECT__PARAMS__action 的目录名，累加 counts 与 values。"""
    if "__" not in name:
        return
    segs = name.split("__")
    if len(segs) < 3:
        return
    action = segs[-1]
    if action not in ACTION_NAMES:
        return
    params_str = "__".join(segs[1:-1])
    params = _parse_params_to_dict(params_str)
    counts_by_action[action] += 1
    for key in TARGET_KEYS:
        if key not in params:
            continue
        try:
            v = float(params[key])
        except ValueError:
            continue
        values[key].append(v)


def _collect_params(auto_output_root: Path):
    """
    从 auto_output 下仿真目录解析参数。支持：
    - auto_output/<action>/<obj__params__action>/
    - auto_output/<model>/<obj__params__action>/  （按物体一级目录）
    """
    values = defaultdict(list)
    counts_by_action = Counter()

    for action in ACTION_NAMES:
        action_dir = auto_output_root / action
        if not action_dir.is_dir():
            continue
        for run_dir in action_dir.iterdir():
            if not run_dir.is_dir():
                continue
            _parse_one_run_folder(run_dir.name, counts_by_action, values)
        # 新：<action>/<model>/<NNNN>/gt_parameters.json
        for model_sub in sorted(action_dir.iterdir(), key=lambda p: p.name):
            if not model_sub.is_dir():
                continue
            if model_sub.name in ACTION_NAMES:
                continue
            for run_dir in sorted(model_sub.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue
                gp = run_dir / "gt_parameters.json"
                if gp.is_file():
                    _accumulate_from_gt_json(gp, counts_by_action, values)

    skip_top = set(ACTION_NAMES) | {"_tmp_configs", "stats", "combined"}
    for model_top in sorted(auto_output_root.iterdir()):
        if not model_top.is_dir() or model_top.name in skip_top:
            continue
        for run_dir in sorted(model_top.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            if run_dir.name.isdigit():
                gp = run_dir / "gt_parameters.json"
                if gp.is_file():
                    _accumulate_from_gt_json(gp, counts_by_action, values)
                continue
            _parse_one_run_folder(run_dir.name, counts_by_action, values)

    return values, counts_by_action


def _plot_hist(values: dict, stats_dir: Path):
    stats_dir.mkdir(parents=True, exist_ok=True)

    for key in TARGET_KEYS:
        data = values.get(key, [])
        if not data:
            print(f"[INFO] 参数 {key} 没有任何数据，跳过绘图。")
            continue

        plt.figure(figsize=(6, 4))
        # E 和 yield_stress 可能跨好几个数量级，使用 log-x 直方图更直观
        use_log = key in ("E", "yield_stress")
        if use_log:
            # 过滤掉非正值，避免 log 时报错
            data_pos = [v for v in data if v > 0]
            if not data_pos:
                print(f"[INFO] 参数 {key} 没有正值数据，跳过 log 直方图。")
                plt.close()
                continue
            plt.hist(data_pos, bins=40, edgecolor="black", alpha=0.7, log=False)
            plt.xscale("log")
            plt.title(f"Log-scale distribution of {key}")
        else:
            plt.hist(data, bins=40, edgecolor="black", alpha=0.7)
            plt.title(f"Distribution of {key}")

        plt.xlabel(key)
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = stats_dir / f"params_hist_{key}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] 保存直方图: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    auto_output = script_dir / "auto_output"
    if not auto_output.is_dir():
        print(f"未找到目录: {auto_output}")
        return

    values, counts_by_action = _collect_params(auto_output)
    total = sum(counts_by_action.values())
    print(f"共解析到 {total} 个仿真目录。按动作计数: {dict(counts_by_action)}")

    stats_dir = auto_output / "stats"
    _plot_hist(values, stats_dir)
    print("完成。")


if __name__ == "__main__":
    main()

