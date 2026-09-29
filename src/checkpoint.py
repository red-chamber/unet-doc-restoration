"""权重的存与读。

只存 state_dict 加一份结构配置，不存整个模型对象：存对象会把类的定义一起序列化
进去，代码结构一改或换了环境就可能加载不了。有了 model_config，加载时先按它把
模型搭出来再灌权重，结构不用手写。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, *, model: nn.Module, model_config: dict, **extra) -> Path:
    """把权重、结构参数和若干元信息写成一个 .pt 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "model_config": model_config, **extra},
        path,
    )
    return path


def load_weights(path: str | Path, device: str, build_fn) -> tuple[nn.Module, dict]:
    """加载权重并重建模型，返回 (模型, checkpoint 字典)。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the weights file: {path}\nPlease run the training script first."
        )

    # weights_only=False 关掉 2.6 起收紧的 pickle 限制：本文件里除了张量还存了
    # args 等普通对象。代价是只应该加载自己训练出来的文件。
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_fn(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()  # 推理模式：关掉 Dropout、BN 用累计统计量
    return model, ckpt
