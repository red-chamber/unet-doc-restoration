"""通用工具：路径常量、随机种子、参数量统计、运行环境、指标导出。"""

from __future__ import annotations

import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# src/ 的上一级就是项目根目录，所有产物都落在它下面
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同样的配置能跑出同样的结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计参数量。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_git_commit() -> str:
    """取当前 commit id，把实验结果和代码版本对应起来。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    """记录运行环境，便于日后复现。"""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def gpu_peak_memory_mb(device: str) -> float:
    """GPU 显存峰值（MB）；用 CPU 时返回 0。"""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def save_metrics(metrics: dict, tag: str) -> Path:
    """把指标写成 reports/metrics/<tag>.json。"""
    path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def load_metrics(tag: str) -> dict:
    """读取训练时导出的指标 json（用于取逐轮 history）。"""
    path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到实验指标：{path}\n请先训练对应模型（train.py）再运行本脚本。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
