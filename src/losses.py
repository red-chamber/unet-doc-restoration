"""训练用的损失函数。

提供三种损失，用 --loss 切换：

    l1       逐像素绝对值误差（默认）
    mse      逐像素平方误差，等价于 L2
    l1_grad  L1 + 梯度（边缘）损失，属于「组合」

为什么默认用 L1 而不是 MSE
--------------------------

MSE 会放大大的误差、对小的误差很宽容，优化目标偏向把误差「摊平」到所有像素上，
结果就是边缘变软、笔迹边界留下灰影。L1 对所有量级的误差一视同仁，倾向保留
清晰的边界 —— 图像恢复任务里通常比 MSE 更锐利。

这不只是经验说法：MSE 的最优解是条件均值，而 L1 的最优解是条件中位数。
在笔迹这种「要么是黑的要么是白的」二值化边缘附近，条件均值会把不同可能性的
灰度平均起来（糊），条件中位数则会挑一个更接近真实的取值（锐）。

为什么还要梯度损失
------------------

L1 逐像素独立计算，网络只要把每个像素凑得差不多就行，并不直接关心相邻像素之间的
关系，所以边界仍可能不够干净。`l1_grad` 额外比较预测和目标在水平/垂直方向的
差分（相当于边缘），对边缘错位直接罚 —— 目标是把手写笔迹的边缘擦干净，
不留残影。grad_weight 控制这一项的权重，默认 0.5。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_NAMES = ("l1", "mse", "l1_grad")


class L1GradLoss(nn.Module):
    """L1 损失 + 梯度损失。

    梯度用最简单的前向差分算（相邻像素相减），不引入 Sobel 之类的卷积核，
    目的只是「比较两个方向的边缘差异」，形式越简单越不容易出错。
    """

    def __init__(self, grad_weight: float = 0.5) -> None:
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel = F.l1_loss(pred, target)

        # 垂直方向的差分：第 i 行减第 i-1 行
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        # 水平方向的差分：第 j 列减第 j-1 列
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]

        # 先分别比较，再取绝对值求均值。不要先把两个方向的梯度加起来再比较 ——
        # 那样正负会互相抵消，边缘信息就丢了。
        grad = (pred_dy - target_dy).abs().mean() + (pred_dx - target_dx).abs().mean()

        return pixel + self.grad_weight * grad


def build_loss(name: str, *, grad_weight: float = 0.5) -> nn.Module:
    """按名字建损失函数。"""
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1_grad":
        return L1GradLoss(grad_weight=grad_weight)
    raise ValueError(f"未知损失 {name!r}，可选：{list(LOSS_NAMES)}")
