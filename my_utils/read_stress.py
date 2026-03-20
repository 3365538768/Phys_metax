import os
import re
import argparse
import numpy as np
import plotly.graph_objects as go


def von_mises(stress):
    """
    stress: [N, 3, 3]
    return: [N]
    """
    stress = np.asarray(stress, dtype=np.float64)
    s = 0.5 * (stress + np.transpose(stress, (0, 2, 1)))
    mean = np.trace(s, axis1=1, axis2=2) / 3.0
    dev = s - mean[:, None, None] * np.eye(3)[None, :, :]
    vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    return vm


def try_get_frame_id(path):
    m = re.search(r"frame_(\d+)\.npz$", os.path.basename(path))
    if m:
        return m.group(1)
    return None


def load_position_from_matching_deformation(stress_file, deformation_dir):
    frame_id = try_get_frame_id(stress_file)
    if frame_id is None:
        raise ValueError(f"无法从文件名解析 frame id: {stress_file}")

    deformation_file = os.path.join(
        deformation_dir,
        f"deformation_frame_{frame_id}.npz"
    )
    if not os.path.exists(deformation_file):
        raise FileNotFoundError(f"未找到对应的 deformation 文件: {deformation_file}")

    data = np.load(deformation_file)
    if "position" not in data:
        raise KeyError(f"{deformation_file} 中没有 'position' 键")
    return data["position"]


def load_stress_and_position(stress_file, deformation_dir=None, stress_key="stress_cauchy"):
    data = np.load(stress_file)

    if stress_key not in data:
        raise KeyError(
            f"{stress_file} 中没有键 '{stress_key}'，可用键为: {list(data.keys())}"
        )

    stress = data[stress_key]

    if "position" in data:
        position = data["position"]
    else:
        if deformation_dir is None:
            raise KeyError(
                f"{stress_file} 中没有 'position'，请通过 --deformation_dir 提供对应形变目录"
            )
        position = load_position_from_matching_deformation(stress_file, deformation_dir)

    return position, stress, list(data.keys())


def pick_stress_scalar(stress, mode="von_mises"):
    """
    stress: [N,3,3]
    mode:
      - von_mises
      - xx, yy, zz, xy, yz, xz
      - mean
    """
    if mode == "von_mises":
        return von_mises(stress)

    if mode == "xx":
        return stress[:, 0, 0]
    if mode == "yy":
        return stress[:, 1, 1]
    if mode == "zz":
        return stress[:, 2, 2]
    if mode == "xy":
        return stress[:, 0, 1]
    if mode == "yz":
        return stress[:, 1, 2]
    if mode == "xz":
        return stress[:, 0, 2]
    if mode == "mean":
        return np.trace(stress, axis1=1, axis2=2) / 3.0

    raise ValueError(f"不支持的 mode: {mode}")


def percentile_clip(values, low=2, high=98):
    vmin = np.percentile(values, low)
    vmax = np.percentile(values, high)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    clipped = np.clip(values, vmin, vmax)
    return clipped, vmin, vmax


def sample_points(position, values, sample_count):
    n = len(position)
    if sample_count <= 0 or sample_count >= n:
        return position, values

    idx = np.linspace(0, n - 1, sample_count)
    idx = np.round(idx).astype(np.int32)
    idx = np.unique(idx)

    return position[idx], values[idx]


def build_hover_text(values, mode):
    return [f"{mode}: {v:.6f}" for v in values]


def plot_stress_3d(position, values, mode="von_mises", point_size=4, title=None, out_html=None):
    x, y, z = position[:, 0], position[:, 1], position[:, 2]

    hover_text = build_hover_text(values, mode)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=values,
                    colorscale="Turbo",
                    colorbar=dict(title=mode),
                    opacity=0.9,
                ),
                text=hover_text,
                hovertemplate=(
                    "x=%{x:.4f}<br>"
                    "y=%{y:.4f}<br>"
                    "z=%{z:.4f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title or f"3D Stress Visualization ({mode})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if out_html is not None:
        fig.write_html(out_html)
        print(f"[INFO] 已保存交互式结果到: {out_html}")

    fig.show()


def main():
    parser = argparse.ArgumentParser(description="3D stress visualization for stress_frame_xxxx.npz")
    parser.add_argument("--stress_file", type=str, required=True,
                        help="stress_frame_xxxx.npz 文件路径")
    parser.add_argument("--deformation_dir", type=str, default=None,
                        help="若 stress 文件内无 position，则从该目录读取对应 deformation_frame_xxxx.npz")
    parser.add_argument("--stress_key", type=str, default="stress_cauchy",
                        help="应力张量键名，默认 stress_cauchy")
    parser.add_argument("--mode", type=str, default="von_mises",
                        choices=["von_mises", "xx", "yy", "zz", "xy", "yz", "xz", "mean"],
                        help="显示哪种应力标量")
    parser.add_argument("--sample_count", type=int, default=300000,
                        help="最多显示多少个点，防止点太多浏览器卡顿")
    parser.add_argument("--point_size", type=int, default=1,
                        help="点大小")
    parser.add_argument("--clip_percentile", action="store_true",
                        help="按百分位裁剪，增强颜色对比")
    parser.add_argument("--out_html", type=str, default="stress_3d.html",
                        help="输出 html 文件名")

    args = parser.parse_args()

    position, stress, keys = load_stress_and_position(
        args.stress_file,
        deformation_dir=args.deformation_dir,
        stress_key=args.stress_key
    )

    values = pick_stress_scalar(stress, mode=args.mode)

    print("[INFO] stress file keys:", keys)
    print("[INFO] position shape:", position.shape)
    print("[INFO] stress shape:", stress.shape)
    print("[INFO] scalar mode:", args.mode)
    print("[INFO] scalar min/max:", float(values.min()), float(values.max()))
    print("[INFO] scalar mean/std:", float(values.mean()), float(values.std()))

    if args.clip_percentile:
        values, vmin, vmax = percentile_clip(values, low=2, high=98)
        print(f"[INFO] percentile clip range: [{vmin:.6f}, {vmax:.6f}]")

    position, values = sample_points(position, values, args.sample_count)
    print("[INFO] displayed points:", len(position))

    plot_stress_3d(
        position=position,
        values=values,
        mode=args.mode,
        point_size=args.point_size,
        title=f"Stress 3D Visualization - {os.path.basename(args.stress_file)}",
        out_html=args.out_html,
    )


if __name__ == "__main__":
    main()