"""测试集评估

在留出的那个日期上跑一遍，给出最终 PSNR / SSIM，并额外做两项质量检查：

    1. PSNR / SSIM 是否达到可接受水平（并给逐图的最差、最好）
    2. 输出有没有退化成全白 / 全黑

    python src/evaluate.py --checkpoint checkpoints/unet_l1_best.pt --keep 4

产出：
    reports/metrics/<tag>_test.json           测试集全部指标
    reports/figures/<tag>_test_worst.png      最差 N 张的对比图
    reports/figures/<tag>_test_best.png       最好 N 张的对比图

为什么要专门看最差的几张
------------------------
平均值会把问题藏起来。如果模型在大多数图上都不错、但在「手写特别密」或
「表格线密集」的图上把印刷内容一起擦掉了，平均值只掉一点，看最差的那几张
才能立刻看出失效模式在哪一类图上。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from checkpoint import load_weights  # noqa: E402
from utils import PROJECT_ROOT, count_parameters, save_metrics  # noqa: E402
from data import PairedDocDataset, build_loaders, list_groups, split_pairs  # noqa: E402
from model import build_model_from_config  # noqa: E402
from predict import evaluate_dataset  # noqa: E402
from viz import plot_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在验证集或测试集上评估 U-Net 的擦除效果")
    p.add_argument("--checkpoint", type=str, required=True, help="权重路径")
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--split", type=str, default="test", choices=["test", "val"],
                   help="评估哪个划分；默认 test")
    p.add_argument("--val-ratio", type=float, default=0.1, help="与训练时保持一致")
    p.add_argument("--train-dates", type=str, nargs="+", default=None,
                   help="与训练时保持一致；不传则自动推断（只用于定位数据目录，评估不碰训练集）")
    p.add_argument("--holdout-date", type=str, default=None,
                   help="与训练时保持一致，评估就在这个分组上跑；不传则取最后一组")
    p.add_argument("--max-images", type=int, default=0,
                   help="只评估前 N 张，0 表示全部（测试集整图评估较慢）")
    p.add_argument("--tile", type=int, default=0, help="分块推理块边长，0 表示整图直推")
    p.add_argument("--keep", type=int, default=4, help="保存最差/最好各多少张的对比图")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default=None,
                   help="输出文件名前缀，默认从权重文件名推断")
    return p.parse_args()


def min_max_mean(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype="float64")
    return float(array.min()), float(array.max()), float(array.mean())


def main() -> None:
    args = parse_args()

    tag = args.tag or Path(args.checkpoint).stem.replace("_best", "")

    print("=" * 72)
    print(f"在 {args.split} 集上评估")
    print("=" * 72)

    model, ckpt = load_weights(Path(args.checkpoint), args.device, build_model_from_config)
    print(f"权重      : {args.checkpoint}")
    print(f"训练轮次  : {ckpt.get('epoch', 'unknown')}")
    print(f"参数量    : {count_parameters(model):,}")
    print(f"设备      : {args.device}")
    if args.tile > 0:
        print(f"分块推理  : 块边长 {args.tile}")

    # ---------- 数据 ----------
    groups = list_groups(args.data_root)
    holdout_date = args.holdout_date or groups[-1]
    splits = split_pairs(
        args.data_root,
        val_ratio=args.val_ratio,
        train_dates=tuple(args.train_dates) if args.train_dates else None,
        holdout_date=holdout_date,
    )
    loaders = build_loaders(
        splits,
        crop_size=0,  # 评估用整图，不裁剪
        batch_size=1,
        num_workers=args.num_workers,
        device=args.device,
        augment=False,
        verbose=True,
    )

    max_images = None if args.max_images <= 0 else args.max_images
    print(f"\n开始评估（整图，{'全部' if max_images is None else f'前 {max_images} 张'}）...")

    summary, per_image, _ = evaluate_dataset(
        model, loaders[args.split], device=args.device, tile=args.tile, max_images=max_images
    )

    if not summary:
        raise SystemExit("没有评估到任何图片，检查 --data-root 与 --split")

    # ---------- 结果 ----------
    psnr_min, psnr_max, psnr_mean = min_max_mean([item["psnr"] for item in per_image])
    ssim_min, ssim_max, ssim_mean = min_max_mean([item["ssim"] for item in per_image])
    std_min, std_max, std_mean = min_max_mean([item["pred_std"] for item in per_image])
    mean_min, mean_max, mean_mean = min_max_mean([item["pred_mean"] for item in per_image])

    print("\n" + "=" * 72)
    print(f"{args.split} 集结果（{summary['n_images']} 张整图）")
    print("=" * 72)
    print(f"PSNR    平均 {psnr_mean:>6.2f} dB   最差 {psnr_min:>6.2f}   最好 {psnr_max:>6.2f}")
    print(f"SSIM    平均 {ssim_mean:>6.4f}      最差 {ssim_min:>6.4f}   最好 {ssim_max:>6.4f}")
    print(f"预测亮度 平均 {mean_mean:.4f}（范围 {mean_min:.4f} ~ {mean_max:.4f}）")
    print(f"预测标准差 平均 {std_mean:.4f}（最小 {std_min:.4f}）")

    # ---------- 质量检查 ----------
    print("\n" + "-" * 72)
    print("质量检查")
    print("-" * 72)
    n_flat = summary["n_flat"]
    flat_ok = n_flat == 0
    print(f"1. PSNR / SSIM 演化曲线      : 由 train.py 生成 "
          f"reports/figures/{tag}_curves.png")
    print(f"2. 不出现全白/全黑           : {'通过' if flat_ok else '未通过'} "
          f"（{n_flat}/{summary['n_images']} 张预测近似纯色，最小标准差 {std_min:.4f}）")
    print(f"3. 训练时长与费用            : 记录在 reports/metrics/{tag}.json"
          f"（train_time_hour / estimated_cost 两个字段）")
    if not flat_ok:
        worst_flat = min(per_image, key=lambda item: item["pred_std"])
        print(f"   最平的一张：{worst_flat['name']}  标准差 {worst_flat['pred_std']:.5f}  "
              f"亮度 {worst_flat['pred_mean']:.4f}")

    # ---------- 保存最差 / 最好的对比图 ----------
    #
    # 画图需要像素，但整轮评估时我们没有把每张预测都留在内存里（整图太大）。
    # 所以这里只对挑出来的那几张重新跑一次推理 —— 几 kb 的开销，
    # 而不是把整个测试集再跑一遍。
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    if args.keep > 0:
        # 文件名（不含扩展名）-> 配对路径，用于按名字取回想要的几张图
        name_to_pair = {pair[0].stem: pair for pair in splits[args.split]}
        ranked = sorted(per_image, key=lambda item: item["psnr"])

        for label, chosen in (("worst", ranked[: args.keep]), ("best", ranked[-args.keep:][::-1])):
            selected = [name_to_pair[item["name"]] for item in chosen if item["name"] in name_to_pair]
            if not selected:
                continue
            sub_loader = DataLoader(
                PairedDocDataset(selected, crop_size=0),
                batch_size=1, shuffle=False, num_workers=0,
            )
            _, _, picked = evaluate_dataset(
                model, sub_loader, device=args.device, tile=args.tile,
                keep_samples=len(selected),
            )
            if picked:
                plot_comparison(
                    picked, figure_dir / f"{tag}_test_{label}.png",
                    f"{args.split} {label} {len(picked)} examples ({tag})",
                    max_rows=len(picked),
                )

    # ---------- 存指标 ----------
    metrics = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_images": summary["n_images"],
        "psnr_mean": psnr_mean,
        "psnr_min": psnr_min,
        "psnr_max": psnr_max,
        "ssim_mean": ssim_mean,
        "ssim_min": ssim_min,
        "ssim_max": ssim_max,
        "pred_mean_mean": mean_mean,
        "pred_std_mean": std_mean,
        "pred_std_min": std_min,
        "n_flat": n_flat,
        "meets_flat_criterion": flat_ok,
        "tile": args.tile,
        "per_image": per_image,
    }

    # 训练时的验证指标也带上，方便看泛化差距
    train_metrics_path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    if train_metrics_path.exists():
        with open(train_metrics_path, encoding="utf-8") as f:
            trained = json.load(f)
        metrics["train_val_psnr"] = trained.get("best_val_psnr")
        metrics["train_val_ssim"] = trained.get("best_val_ssim")
        metrics["train_time_hour"] = trained.get("train_time_hour")
        metrics["estimated_cost"] = trained.get("estimated_cost")
        if trained.get("best_val_psnr"):
            gap = trained["best_val_psnr"] - psnr_mean
            print(f"\n泛化差距：验证 {trained['best_val_psnr']:.2f} dB -> "
                  f"{args.split} {psnr_mean:.2f} dB（差 {gap:+.2f} dB）")

    metrics_path = save_metrics(metrics, f"{tag}_test")
    print(f"\n评估指标已保存：{metrics_path}")


if __name__ == "__main__":
    main()
