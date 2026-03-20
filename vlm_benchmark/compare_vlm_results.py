import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_COLS = ["E", "nu", "density", "yield_stress"]


def _load_and_align(gt_path: Path, pred_path: Path) -> pd.DataFrame:
    """
    当前的预测文件 already 包含 GT 列和 Pred 列：
      rel_video_path, action, material_gt,
      E_gt, nu_gt, density_gt, yield_stress_gt,
      E_pred, nu_pred, density_pred, yield_stress_pred,
      material_pred, motion_pred

    因此这里只需要读取预测文件即可，gt_path 仅作存在性检查。
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT 文件不存在: {gt_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    df = pd.read_csv(pred_path)
    if df.empty:
        raise RuntimeError("预测文件为空，请先运行 VLM benchmark 生成结果。")
    return df


def _compute_regression_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUM_COLS:
        g = f"{col}_gt"
        p = f"{col}_pred"
        if g not in df or p not in df:
            continue

        gt_vals = df[g].astype(float).to_numpy()
        pred_vals = df[p].astype(float).to_numpy()

        abs_err = np.abs(pred_vals - gt_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(gt_vals != 0, abs_err / np.abs(gt_vals), np.nan)

        mae = float(np.nanmean(abs_err))
        rmse = float(np.sqrt(np.nanmean((pred_vals - gt_vals) ** 2)))
        mape = float(np.nanmean(rel_err) * 100.0)

        rows.append(
            {
                "metric": col,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE(%)": mape,
            }
        )

    return pd.DataFrame(rows)


def _compute_classification_accuracy(df: pd.DataFrame) -> Tuple[float, float]:
    # 材质分类准确率
    mat_gt = df["material_gt"].astype(str).str.lower()
    mat_pred = df.get("material_pred", pd.Series([""] * len(df))).astype(str).str.lower()
    mat_acc = float((mat_gt == mat_pred).mean())

    # 运动类型分类准确率
    motion_gt = df["action"].astype(str).str.lower()
    motion_pred = df.get("motion_pred", pd.Series([""] * len(df))).astype(str).str.lower()
    motion_acc = float((motion_gt == motion_pred).mean())

    return mat_acc, motion_acc


def _compute_per_category_accuracy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按材质、按 motion 分别计算分类准确率。
    返回 (material_acc_df, motion_acc_df)。
    """
    mat_gt = df["material_gt"].astype(str).str.lower()
    mat_pred = df.get("material_pred", pd.Series([""] * len(df))).astype(str).str.lower()
    motion_gt = df["action"].astype(str).str.lower()
    motion_pred = df.get("motion_pred", pd.Series([""] * len(df))).astype(str).str.lower()

    mat_correct = mat_gt == mat_pred
    motion_correct = motion_gt == motion_pred

    mat_rows = []
    for mat in mat_gt.unique():
        if pd.isna(mat) or not str(mat).strip():
            continue
        mask = mat_gt == mat
        n = mask.sum()
        if n == 0:
            continue
        acc = float(mat_correct[mask].mean())
        mat_rows.append({"material": mat, "count": int(n), "accuracy": acc})

    motion_rows = []
    for motion in motion_gt.unique():
        if pd.isna(motion) or not str(motion).strip():
            continue
        mask = motion_gt == motion
        n = mask.sum()
        if n == 0:
            continue
        acc = float(motion_correct[mask].mean())
        motion_rows.append({"motion": motion, "count": int(n), "accuracy": acc})

    mat_df = pd.DataFrame(mat_rows)
    motion_df = pd.DataFrame(motion_rows)
    return mat_df, motion_df


def _scatter_gt_vs_pred(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in NUM_COLS:
        g = f"{col}_gt"
        p = f"{col}_pred"
        if g not in df or p not in df:
            continue

        x = df[g].astype(float).to_numpy()
        y = df[p].astype(float).to_numpy()

        plt.figure(figsize=(5, 5))

        if col in ("E", "yield_stress"):
            # 对 E 采用对数坐标，缓解跨好几个数量级导致“全挤在一起”的问题
            # 仅保留正值，避免 log10 报错
            mask = (x > 0) & (y > 0)
            x_pos = x[mask]
            y_pos = y[mask]
            if x_pos.size == 0:
                plt.close()
                continue

            plt.scatter(x_pos, y_pos, s=8, alpha=0.5)
            # 使用 log-log 轴，同时根据数据 min/max 自适应范围
            plt.xscale("log")
            plt.yscale("log")
            lim_min = float(min(x_pos.min(), y_pos.min()))
            lim_max = float(max(x_pos.max(), y_pos.max()))
            # 给一点 padding，避免点贴边
            if lim_min > 0 and lim_max > 0 and lim_max > lim_min:
                pad_low = lim_min / 1.5
                pad_high = lim_max * 1.5
                plt.xlim(pad_low, pad_high)
                plt.ylim(pad_low, pad_high)
            plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="y = x")
        else:
            plt.scatter(x, y, s=8, alpha=0.5)
            lim_min = min(x.min(), y.min())
            lim_max = max(x.max(), y.max())
            plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="y = x")

        plt.xlabel(f"{col} GT")
        plt.ylabel(f"{col} Pred")
        plt.title(f"{col}: GT vs Pred")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"scatter_{col}_gt_vs_pred.png"
        plt.savefig(out_path)
        plt.close()


def _hist_relative_error(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in NUM_COLS:
        g = f"{col}_gt"
        p = f"{col}_pred"
        if g not in df or p not in df:
            continue

        gt_vals = df[g].astype(float).to_numpy()
        pred_vals = df[p].astype(float).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(gt_vals != 0, (pred_vals - gt_vals) / np.abs(gt_vals), np.nan)

        rel_err = rel_err[np.isfinite(rel_err)]
        if rel_err.size == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(rel_err, bins=40, edgecolor="black", alpha=0.7)
        plt.xlabel(f"Relative error of {col} ( (pred-gt)/|gt| )")
        plt.ylabel("Count")
        plt.title(f"Relative error distribution of {col}")
        plt.tight_layout()
        out_path = out_dir / f"hist_relerr_{col}.png"
        plt.savefig(out_path)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 VLM 预测与 GT，并输出数值指标和可视化。"
    )
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="PhysGaussian 根目录（包含 auto_output 和 vlm_benchmark）",
    )
    parser.add_argument(
        "--vlm_tag",
        type=str,
        default="qwen2.5-vl",
        help="要评估的模型标签（与 vlm_model_registry.py 中一致）",
    )
    args = parser.parse_args()
    root = Path(args.physgaussian_root).resolve()
    # 与 run_vlm_benchmark.py 保持一致的输出结构
    output_root = root / "vlm_benchmark" / "output" / args.vlm_tag
    gt_path = output_root / "vlm_benchmark_gt.csv"
    pred_path = output_root / "vlm_benchmark_pred.csv"
    out_dir = output_root / "plots"
    df = _load_and_align(gt_path, pred_path)
    # 数值回归指标
    metrics_df = _compute_regression_metrics(df)
    print("=== Regression metrics (overall) ===")
    print(metrics_df.to_string(index=False))
    # 分类准确率
    mat_acc, motion_acc = _compute_classification_accuracy(df)
    print("\n=== Classification accuracy ===")
    print(f"Material accuracy: {mat_acc * 100:.2f}%")
    print(f"Motion   accuracy: {motion_acc * 100:.2f}%")

    # 按材质、按 motion 的准确率
    mat_acc_df, motion_acc_df = _compute_per_category_accuracy(df)
    print("\n=== Material accuracy (per category) ===")
    if not mat_acc_df.empty:
        mat_acc_df["accuracy(%)"] = (mat_acc_df["accuracy"] * 100).round(2)
        print(mat_acc_df[["material", "count", "accuracy(%)"]].to_string(index=False))
    else:
        print("(no data)")
    print("\n=== Motion accuracy (per category) ===")
    if not motion_acc_df.empty:
        motion_acc_df["accuracy(%)"] = (motion_acc_df["accuracy"] * 100).round(2)
        print(motion_acc_df[["motion", "count", "accuracy(%)"]].to_string(index=False))
    else:
        print("(no data)")
    # 可视化
    _scatter_gt_vs_pred(df, out_dir)
    _hist_relative_error(df, out_dir)
    print(f"\n可视化结果已保存到: {out_dir}")

if __name__ == "__main__":
    main()