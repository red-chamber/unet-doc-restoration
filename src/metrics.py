"""PSNR / SSIM 与退化输出检查。

为什么这两个指标要一起看
------------------------

PSNR 只看逐像素的均方误差，对「整体亮度对但结构糊了」这种失败很宽容 ——
把结果轻微模糊一下，PSNR 往往还很高，但人眼看得出印刷字变软了。

SSIM 从亮度、对比度、结构三个角度比较，对结构性的改变更敏感。
所以这个任务里两个一起报：PSNR 衡量像素误差，SSIM 衡量结构是否保住。

参考值与经验：PSNR 30 dB 以上肉眼已不容易看出差异，40 dB 以上基本无感。
本任务要在擦掉手写的同时保住印刷字，若把印刷字也一起擦掉，PSNR 会因为
背景大面积一致而虚高，但 SSIM 会明显掉下来 —— 这正是需要两个指标互相制约的地方。
"""

from __future__ import annotations

import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 图像数据都在 [0,1]，所以动态范围是 1.0
DATA_RANGE = 1.0

# 判定「输出退化成接近纯色」的标准差阈值
FLAT_STD_THRESHOLD = 5e-3


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    """把张量压成 (H, W) 的 numpy 数组，取值 [0,1]。

    调用方传进来的通常是 (1, 1, H, W)（batch=1、单通道），也可能是 (1, H, W) 或 (H, W)。
    skimage 的 SSIM 只接受二维数组，所以要把前面的长度 1 维度全部挤掉 ——
    只 squeeze 一次的话，四维张量会原样传进去，SSIM 会把它当成很小的图而报
    「win_size exceeds image extent」。
    """
    array = image.detach().cpu().float()
    while array.dim() > 2:
        array = array.squeeze(0)
    return array.numpy()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """峰值信噪比，单位 dB，越高越好。"""
    return float(peak_signal_noise_ratio(_to_numpy(target), _to_numpy(pred), data_range=DATA_RANGE))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """结构相似度，取值 [-1, 1]，越接近 1 越好。"""
    return float(structural_similarity(_to_numpy(target), _to_numpy(pred), data_range=DATA_RANGE))


def image_stats(pred: torch.Tensor) -> dict:
    """输出图的亮度均值与标准差，用来判断有没有退化成纯色。

    输出退化成常数是这类任务最典型的失败模式
    最典型的失败模式（网络发现「输出一片浅灰」能在 L1 损失下拿到不错的平均分），
    所以每个 epoch 都顺手记一下这两个数，标准差接近 0 就说明模型塌了。
    """
    array = _to_numpy(pred)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "is_flat": bool(array.std() < FLAT_STD_THRESHOLD),
    }


def summarize(per_image: list[dict]) -> dict:
    """把逐图指标汇总成均值，并统计有多少张退化成了纯色。

    数值字段是自动识别的（布尔除外）—— 这样调用方多传一个 loss 进来，
    汇总结果就自动多一项，不需要在这里维护字段清单。
    """
    if not per_image:
        return {}

    numeric_keys = [
        key
        for key, value in per_image[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary = {key: float(np.mean([item[key] for item in per_image])) for key in numeric_keys}

    summary["n_images"] = len(per_image)
    summary["n_flat"] = int(sum(item["is_flat"] for item in per_image))
    summary["psnr_min"] = float(np.min([item["psnr"] for item in per_image]))
    summary["psnr_max"] = float(np.max([item["psnr"] for item in per_image]))
    return summary
