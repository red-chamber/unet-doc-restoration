"""绘图：训练曲线与「输入 / 目标 / 预测 / 误差」对比图。

matplotlib 用 Agg 后端：训练常在无图形界面的环境里跑，默认后端会因为找不到
显示设备而报错。图统一用 fig.savefig 保存并显式建目录，画完关掉 figure 释放内存。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 跨图统一的配色
TRAIN_COLOR = "#1f77b4"  # 蓝
VAL_COLOR = "#d62728"  # 红

# 误差图的放大倍数。预测与目标的差通常在 0.02 这个量级，
# 按原值显示几乎全黑，放大 5 倍才能看清笔迹残留留在哪里。
ERROR_GAIN = 5.0


def save_figure(fig, out_path: str | Path) -> None:
    """存图并释放。用 fig.savefig 而不是 plt.savefig —— 后者存的是「当前活跃的
    figure」，有多个 figure 时容易存错。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"图已保存：{out_path}")


def plot_training_curves(history: dict, out_path: str | Path, title: str) -> None:
    """Loss / PSNR / SSIM 三条演化曲线。

    三张图并排而不是画在同一个坐标系里：Loss 和 PSNR 的量纲差得很远
    （一个是 0.0x，一个是 30 上下），叠在一起谁也看不清。
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # 左：损失
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color=TRAIN_COLOR)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validation", color=VAL_COLOR)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 中：PSNR（验证集，单位 dB）
    axes[1].plot(epochs, history["val_psnr"], "o-", label="Validation", color=VAL_COLOR)
    best_epoch = int(np.argmax(history["val_psnr"])) + 1
    best_value = max(history["val_psnr"])
    axes[1].scatter([best_epoch], [best_value], marker="*", s=140, color="#2ca02c", zorder=5,
                    label=f"Best {best_value:.2f} dB (epoch {best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 右：SSIM（验证集，越接近 1 越好）
    axes[2].plot(epochs, history["val_ssim"], "o-", label="Validation", color=VAL_COLOR)
    best_epoch = int(np.argmax(history["val_ssim"])) + 1
    best_value = max(history["val_ssim"])
    axes[2].scatter([best_epoch], [best_value], marker="*", s=140, color="#2ca02c", zorder=5,
                    label=f"Best {best_value:.4f} (epoch {best_epoch})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_comparison(samples: list[dict], out_path: str | Path, title: str, max_rows: int = 4) -> None:
    """四列对比图：带手写输入 / 干净目标 / 模型预测 / 误差（放大）。

    列的顺序是阅读顺序：先看要解决什么问题（输入），再看标准答案（目标），
    然后是模型的回答（预测），最后一列专门看「差在哪里」——
    残留的手写笔迹会在误差图上直接显形，比肉眼看预测图更容易发现问题。
    """
    if not samples:
        print("没有可画的样本，跳过对比图")
        return

    rows = min(max_rows, len(samples))
    fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 3.5), squeeze=False)

    for row in range(rows):
        sample = samples[row]
        panels = (
            (sample["input"], "gray", "Input (with handwriting)"),
            (sample["target"], "gray", "Target (clean)"),
            (sample["pred"], "gray", "Prediction"),
            (np.abs(sample["pred"] - sample["target"]) * ERROR_GAIN, "magma",
             f"Error x{ERROR_GAIN:g}"),
        )
        for col, (image, cmap, label) in enumerate(panels):
            ax = axes[row][col]
            if cmap == "gray":
                ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                # 误差图固定上限，不同样本之间才能横向比较
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=10)

        axes[row][0].set_ylabel(sample["name"], fontsize=7)
        axes[row][2].set_xlabel(f"PSNR {sample['psnr']:.2f} dB / SSIM {sample['ssim']:.4f}",
                                fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_path)
