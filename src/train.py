"""训练脚本

数据加载、模型、损失、指标、绘图各自在独立模块里，本文件只负责：
命令行接口、组装、调用训练循环、导出指标。

默认参数按 24 GB 显存级别设定（原分辨率随机裁剪 384×384、batch size 16）。
显存不够时把 --crop 降到 256、--batch-size 降到 8 可以跑通，但超参变了，
得到的数值不会和记录里的一致，只能用来验证流程。

outputs:
    checkpoints/<tag>_best.pt         验证 PSNR 最高的权重（推理用）
    checkpoints/<tag>_last.pt         最近一轮 + 优化器状态（断点续训用）
    reports/figures/<tag>_curves.png  Loss / PSNR / SSIM 演化曲线
    reports/figures/<tag>_samples.png 输入/目标/预测/误差 四列对比图
    reports/metrics/<tag>.json        全部超参、指标与逐轮 history

用法：
    # 单次完整训练
    python src/train.py --data-root /path/to/dataset --epochs 60 --crop 384 --batch-size 16 --loss l1

    # 被打断后续训（会从 <tag>_last.pt 恢复模型、优化器与 epoch）
    python src/train.py --data-root /path/to/dataset --epochs 60 --tag unet_l1 --resume
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from utils import (  # noqa: E402
    PROJECT_ROOT,
    count_parameters,
    env_info,
    get_git_commit,
    save_metrics,
    set_seed,
)
from data import build_loaders, list_groups, split_pairs  # noqa: E402
from engine import build_scheduler, run_training  # noqa: E402
from losses import LOSS_NAMES, build_loss  # noqa: E402
from model import UNet  # noqa: E402
from viz import plot_comparison, plot_training_curves  # noqa: E402

# 实测数据集里的日期目录名，写在这里是为了让 --train-dates 的默认值可读
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 U-Net 做手写内容擦除")
    # --- 数据 ---
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                   help="数据集根目录，期望 <root>/<日期>/dataset/{input,output}/")
    p.add_argument("--crop", type=int, default=384,
                   help="训练时的随机裁剪边长（原分辨率裁剪，不缩放）")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="留出日期里用作验证集的比例，其余作测试集")
    p.add_argument("--train-dates", type=str, nargs="+", default=None,
                   help="用哪几个分组训练；不传则自动取除留出组以外的所有分组")
    p.add_argument("--holdout-date", type=str, default=None,
                   help="留出作验证/测试的分组；不传则取名称排序后的最后一组")
    p.add_argument("--augment", action="store_true", help="开启随机缩放抖动（0.85~1.15）")
    # --- 训练 ---
    p.add_argument("--epochs", type=int, default=60, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=16, help="每个 batch 的裁剪块数")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"],
                   help="学习率调度")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值，0 表示不裁剪")
    p.add_argument("--loss", type=str, default="l1", choices=list(LOSS_NAMES), help="损失函数")
    p.add_argument("--grad-weight", type=float, default=0.5, help="l1_grad 损失里梯度项的权重")
    # --- 模型 ---
    p.add_argument("--base-channels", type=int, default=64,
                   help="U-Net 第一层通道数，每级翻倍。显存紧张可降到 32")
    p.add_argument("--depth", type=int, default=4, help="下采样次数，输入边长需能被 2**depth 整除")
    p.add_argument("--bilinear", action="store_true",
                   help="上采样用双线性插值 + 卷积，而不是转置卷积（参数更少，伪影更少）")
    # --- 验证与产出 ---
    p.add_argument("--val-images", type=int, default=20,
                   help="每轮验证用多少张整图，0 表示全部（整图评估很慢，训练中途适当抽样）")
    p.add_argument("--samples", type=int, default=4, help="训练结束生成多少张对比图")
    p.add_argument("--tile", type=int, default=0,
                   help="推理分块边长；0 表示整图直推。整图 OOM 时设 512")
    # --- 其他 ---
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader 进程数")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="unet_l1", help="产物文件名前缀")
    p.add_argument("--resume", action="store_true", help="从 <tag>_last.pt 续训")
    p.add_argument("--gpu-hourly-cost", type=float, default=0.0,
                   help="GPU 每小时费用（元），用于估算总花费；0 表示不估算")
    # 0 表示全部验证图，argparse 无法区分「没传」和「传了 0」，所以约定 0 = 全部
    return p.parse_args()


def build_experiment_metrics(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    n_params: int,
    split_sizes: dict,
    result,
    figure_path: Path,
    ckpt_path: Path,
) -> dict:
    """组装本次实验的指标。

    除了超参和逐轮 history，还带上模型结构、参数量、显存峰值与运行环境，
    任何一次训练的结果都能从这份 json 单独还原出来。
    """
    hours = result.train_time_sec / 3600.0
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "model": f"U-Net base={args.base_channels} depth={args.depth}"
                 f"{' bilinear' if args.bilinear else ''}",
        "params": n_params,
        "task": "handwriting removal (paired image-to-image)",
        "dataset": (
            f"paired documents: train {'+'.join(args.train_dates)} ({split_sizes['train']} 对) / "
            f"val+test {args.holdout_date} ({split_sizes['val']} + {split_sizes['test']} 对)"
        ),
        "input_size": f"native-resolution random crop {args.crop}x{args.crop}",
        "seed": args.seed,
        "crop_size": args.crop,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "grad_clip": args.grad_clip,
        "loss_fn": args.loss,
        "grad_weight": args.grad_weight if args.loss == "l1_grad" else None,
        "epochs": args.epochs,
        "augment": args.augment,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "bilinear": args.bilinear,
        "best_epoch": result.best_epoch,
        "best_val_psnr": result.best_val_psnr,
        "best_val_ssim": result.best_val_ssim,
        "final_val_psnr": result.final_val_psnr,
        "final_val_ssim": result.final_val_ssim,
        "train_time_sec": round(result.train_time_sec, 2),
        "train_time_hour": round(hours, 3),
        "gpu_peak_mem_mb": round(result.gpu_peak_mb, 1),
        "gpu_hourly_cost": args.gpu_hourly_cost,
        "estimated_cost": round(result.cost, 2),
        "history": result.history,
        "figure": str(figure_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "model_config": model.config(),
        "env": env_info(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 72)
    print("文档手写擦除 · U-Net 训练")
    print("=" * 72)
    print(f"设备      : {args.device}")
    if args.device == "cuda":
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"随机种子  : {args.seed}")
    print(f"数据集    : {args.data_root}")

    # ---------- 数据 ----------
    # 先把分组定下来，再交给 split_pairs，这样指标里能记下实际用了哪几组
    groups = list_groups(args.data_root)
    holdout_date = args.holdout_date or groups[-1]
    train_dates = (
        tuple(args.train_dates)
        if args.train_dates
        else tuple(g for g in groups if g != holdout_date)
    )
    if not train_dates:
        raise SystemExit(f"除了留出组 {holdout_date} 之外没有别的分组可训练")

    splits = split_pairs(
        args.data_root,
        val_ratio=args.val_ratio,
        train_dates=train_dates,
        holdout_date=holdout_date,
    )
    # 回填实际用到的分组，指标 json 里记的是这个而不是命令行原值
    args.train_dates = list(train_dates)
    args.holdout_date = holdout_date
    loaders = build_loaders(
        splits,
        crop_size=args.crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        augment=args.augment,
        seed=args.seed,
    )
    split_sizes = {name: len(pairs) for name, pairs in splits.items()}

    # ---------- 模型 ----------
    model = UNet(
        base_channels=args.base_channels,
        depth=args.depth,
        bilinear=args.bilinear,
    ).to(args.device)
    n_params = count_parameters(model)
    print(f"\n模型      : U-Net base_channels={args.base_channels} depth={args.depth}"
          f"{' bilinear' if args.bilinear else ''}")
    print(f"参数量    : {n_params:,}")

    # ---------- 损失与优化器 ----------
    criterion = build_loss(args.loss, grad_weight=args.grad_weight)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)
    print(f"损失      : {args.loss}"
          + (f"（梯度项权重 {args.grad_weight}）" if args.loss == "l1_grad" else ""))
    print(f"优化器    : {args.optimizer.upper()}，lr={args.lr}，weight_decay={args.weight_decay}"
          f"，scheduler={args.scheduler}")

    # ---------- 训练 ----------
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.tag}_best.pt"
    last_ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.tag}_last.pt"
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    sample_dir = PROJECT_ROOT / "reports" / "samples"

    # val_images 约定 0 = 全部
    val_images = None if args.val_images <= 0 else args.val_images

    result = run_training(
        model,
        loaders,
        criterion,
        optimizer,
        epochs=args.epochs,
        device=args.device,
        ckpt_path=ckpt_path,
        last_ckpt_path=last_ckpt_path,
        model_config=model.config(),
        args_dict=vars(args),
        scheduler=scheduler,
        tile=args.tile,
        val_images=val_images,
        samples=args.samples,
        sample_dir=sample_dir,
        plot_fn=plot_comparison,
        grad_clip=args.grad_clip,
        resume=args.resume,
        hourly_cost=args.gpu_hourly_cost,
    )

    # ---------- 曲线与指标 ----------
    figure_path = figure_dir / f"{args.tag}_curves.png"
    plot_training_curves(
        result.history, figure_path,
        f"U-Net handwriting removal (crop {args.crop}, loss {args.loss})",
    )

    metrics = build_experiment_metrics(
        args,
        model=model,
        n_params=n_params,
        split_sizes=split_sizes,
        result=result,
        figure_path=figure_path,
        ckpt_path=ckpt_path,
    )
    metrics_path = save_metrics(metrics, args.tag)
    print(f"实验指标已保存：{metrics_path}")

    # ---------- 实验摘要 ----------
    n_flat = sum(result.history.get("val_flat", []))
    print("\n" + "=" * 72)
    print("实验摘要")
    print("=" * 72)
    print(f"实验编号与日期  : {args.tag} / {metrics['date']}")
    print(f"Git commit      : {metrics['git_commit']}")
    print(f"模型            : {metrics['model']}（{n_params:,} 参数）")
    print(f"数据集与划分    : {metrics['dataset']}")
    print(f"输入尺寸        : 原分辨率随机裁剪 {args.crop}x{args.crop}")
    print(f"batch size      : {args.batch_size}")
    print(f"优化器/学习率   : {args.optimizer} / {args.lr}，scheduler={args.scheduler}")
    print(f"损失函数        : {args.loss}")
    print(f"训练轮数        : {args.epochs}（最优在第 {result.best_epoch} 轮）")
    print(f"验证 PSNR (最优): {result.best_val_psnr:.2f} dB")
    print(f"验证 SSIM (最优): {result.best_val_ssim:.4f}")
    print(f"训练时长        : {result.train_time_sec / 60:.1f} 分钟"
          f"（{result.train_time_sec / 3600:.2f} 小时）")
    print(f"GPU 显存峰值    : {result.gpu_peak_mb:.1f} MB")
    if args.gpu_hourly_cost > 0:
        print(f"预估费用        : {result.cost:.2f} 元（{args.gpu_hourly_cost} 元/小时）")
    else:
        print(f"预估费用        : 未估算（传 --gpu-hourly-cost 可按小时费用自动算）")
    print(f"退化输出检查    : 验证集累计 {n_flat} 次接近纯色"
          f"（{'正常' if n_flat == 0 else '⚠ 需要检查'}）")
    print(f"曲线            : {metrics['figure']}")
    print("=" * 72)
    print("\n下一步：用 evaluate.py 在测试集上跑一遍，得到最终 PSNR / SSIM")


if __name__ == "__main__":
    main()
