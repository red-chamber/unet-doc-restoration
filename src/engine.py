"""训练循环（图像到图像恢复）。

这个任务每轮的记录和分类任务不同：分类看 loss 和 accuracy、按验证准确率保存
最优权重；恢复任务看 loss 和 PSNR / SSIM、按验证 PSNR 保存最优权重，而且每隔
若干轮要存一张对比图，用眼睛确认笔迹到底擦干净没有 —— 指标涨了但输出还是一团
灰的情况并不少见。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

from checkpoint import save_checkpoint
from utils import gpu_peak_memory_mb
from predict import evaluate_dataset


class TrainResult(NamedTuple):
    """一次训练的产出。"""

    history: dict
    best_epoch: int
    best_val_psnr: float
    best_val_ssim: float
    final_val_psnr: float
    final_val_ssim: float
    train_time_sec: float
    gpu_peak_mb: float
    cost: float


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    grad_clip: float = 0.0,
    log_every: int = 50,
) -> float:
    """跑一个训练 epoch，返回平均损失。

    与分类任务的三步一样（zero_grad / backward / step），差别在于：
    输入和目标都是图，算的是逐像素的恢复误差。
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        prediction = model(inputs)
        loss = criterion(prediction, targets)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪对 U-Net 这种深层编解码结构比较有用：
        # 转置卷积和 BatchNorm 组合下偶发的大梯度会让某一轮突然把权重带偏
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        if log_every and step % log_every == 0:
            print(f"    step {step:>4}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    return total_loss / max(1, total_samples)


def run_training(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: str,
    ckpt_path: str | Path,
    last_ckpt_path: str | Path,
    model_config: dict,
    args_dict: dict,
    scheduler=None,
    tile: int = 0,
    val_images: int | None = None,
    samples: int = 0,
    sample_dir: str | Path | None = None,
    plot_fn=None,
    grad_clip: float = 0.0,
    resume: bool = False,
    hourly_cost: float = 0.0,
    verbose: bool = True,
) -> TrainResult:
    """完整训练流程。

    步骤：
        1. 逐轮训练，每隔若干轮（默认每轮）在验证集上算 PSNR / SSIM
        2. 同时保存两种权重：
             best.pt  验证 PSNR 最高的那一轮，推理时用
             last.pt  最近一轮，含优化器状态，断点续训用
        3. 训练结束后画对比图，看清笔迹到底擦干净没有

    为什么按 PSNR 而不是按 loss 挑最优权重：
        训练损失是在随机裁剪的小块上算的，验证指标是在整图上算的，两者的
        口径不同。用验证指标挑才对得上最终评测的方式。

    续训（--resume）：
        加载 last.pt 里的模型权重、优化器状态和 epoch 号继续跑。
        云 GPU 按小时计费，中途被打断是常态，不续训就得从头再来一遍。
    """
    ckpt_path = Path(ckpt_path)
    last_ckpt_path = Path(last_ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "val_flat": []}
    best_val_psnr = -float("inf")
    best_val_ssim = 0.0
    best_epoch = 0
    start_epoch = 1

    # ---------- 断点续训 ----------
    if resume and last_ckpt_path.exists():
        state = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and "scheduler_state" in state and state["scheduler_state"]:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        history = state.get("history", history)
        best_val_psnr = float(state.get("best_val_psnr", best_val_psnr))
        best_val_ssim = float(state.get("best_val_ssim", best_val_ssim))
        best_epoch = int(state.get("best_epoch", 0))
        print(f"从 {last_ckpt_path} 续训：已跑 {start_epoch - 1} 轮，最好验证 PSNR {best_val_psnr:.2f} dB")
    elif resume:
        print(f"--resume 指定了但没有找到 {last_ckpt_path}，从头开始训练")

    if start_epoch > epochs:
        print(f"已经跑满 {epochs} 轮，无需训练（如需继续请调大 --epochs）")

    # ---------- 训练循环 ----------
    print("\n" + "-" * 72)
    print(f"{'Epoch':>7} {'Train Loss':>11} {'Val Loss':>10} {'Val PSNR':>10} {'Val SSIM':>10} {'Flat':>6} {'Time':>8}")
    print("-" * 72)

    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, grad_clip=grad_clip
        )
        val_summary, _, _ = evaluate_dataset(
            model, loaders["val"], device=device, tile=tile, max_images=val_images, criterion=criterion
        )
        dt = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        val_loss = val_summary.get("loss", float("nan"))
        val_psnr = val_summary.get("psnr", float("nan"))
        val_ssim = val_summary.get("ssim", float("nan"))
        n_flat = val_summary.get("n_flat", 0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["val_flat"].append(n_flat)

        marker = ""
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                model=model,
                model_config=model_config,
                epoch=epoch,
                val_psnr=val_psnr,
                val_ssim=val_ssim,
                args=args_dict,
                history=history,
            )
            marker = "  <- best"

        # 每轮都覆盖存一份，供续训用（含优化器状态，文件比 best 大）
        save_checkpoint(
            last_ckpt_path,
            model=model,
            model_config=model_config,
            epoch=epoch,
            val_psnr=val_psnr,
            val_ssim=val_ssim,
            args=args_dict,
            history=history,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            best_val_psnr=best_val_psnr,
            best_val_ssim=best_val_ssim,
            best_epoch=best_epoch,
        )

        if verbose:
            print(
                f"{epoch:>7} {train_loss:>11.4f} {val_loss:>10.4f} "
                f"{val_psnr:>9.2f}dB {val_ssim:>10.4f} {n_flat:>6} {dt:>7.1f}s{marker}"
            )
            if n_flat:
                print(f"    ⚠ 有 {n_flat} 张输出接近纯色（标准差 < 阈值），模型可能塌了")

    total_time = time.time() - start_time
    cost = total_time / 3600.0 * hourly_cost

    if verbose:
        print("-" * 72)
        print(
            f"训练结束，本次耗时 {total_time / 60:.1f} 分钟，"
            f"最好的 epoch 是第 {best_epoch} 轮（验证 PSNR {best_val_psnr:.2f} dB / "
            f"SSIM {best_val_ssim:.4f}）"
        )
        if hourly_cost > 0:
            print(f"预估费用：{total_time / 3600:.2f} 小时 x {hourly_cost:.2f} 元/小时 = {cost:.2f} 元")

    # ---------- 训练结束后出对比图（用最优权重）----------
    if samples > 0 and sample_dir is not None and plot_fn is not None:
        print("\n加载最优权重，生成对比图...")
        if ckpt_path.exists():
            best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_state["model_state"])
        _, _, sample_images = evaluate_dataset(
            model, loaders["val"], device=device, tile=tile, max_images=samples, keep_samples=samples
        )
        sample_path = Path(sample_dir) / f"{args_dict.get('tag', 'unet')}_samples.png"
        plot_fn(sample_images, sample_path, "U-Net: handwriting removal", max_rows=min(4, samples))

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_psnr=best_val_psnr,
        best_val_ssim=best_val_ssim,
        final_val_psnr=history["val_psnr"][-1] if history["val_psnr"] else float("nan"),
        final_val_ssim=history["val_ssim"][-1] if history["val_ssim"] else float("nan"),
        train_time_sec=total_time,
        gpu_peak_mb=gpu_peak_memory_mb(device),
        cost=cost,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int):
    """按名字建学习率调度器。

    文档恢复任务常用余弦退火：前期大学习率快速到位，后期小学习率精修边缘。
    """
    if not name or name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    raise ValueError(f"未知调度器 {name!r}，可选：['none', 'cosine', 'step']")
