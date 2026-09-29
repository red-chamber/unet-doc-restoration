"""U-Net model 主体

结构是原版的标准形态：编码器逐级下采样、解码器逐级上采样，同一层级的编码器特征通过跳跃连接直接接到解码器上。
Encoder-Bottleneck-Decoder
1. pool 下采样，是 2*2 最大池化使空间减半
2. up 上采样，转置卷积，使通道数减半但空间加倍
3. concat 拼接，把 enc and dec 的通道数加起来

    Input: (1, H, W)
      ├─ enc1: ConvBlock(  1 ->  64)  ────────────────────┐    # skip connection 跳跃连接
      │   pool: output(64,H/2,W/2)                        │    # encoder 与 decoder 直连存放原始副本
      |                                                   |
      ├─ enc2: ConvBlock( 64 -> 128)  ─────────────┐      │
      │   pool: output(128,H/4,W/4)                │      │
      |                                            |      |
      ├─ enc3: ConvBlock(128 -> 256)  ──────┐      │      │
      │   pool: output(256,H/8,W/8)         │      │      │
      |                                     |      |      |
      ├─ enc4: ConvBlock(256 -> 512)  ─┐    │      │      │
      │   pool: output(256,H/16,W/16)  │    │      │      │
      |                                |    |      |      |
      └─ bottleneck: ConvBlock(512 -> 1024) │      │      │
                                       |    |      |      |
          up + concat ─────────────────┘    │      │      │    # up: C/=2;H*=2;W*=2; concat: C=C_1+C_2;
                                            |      |      |
          dec4: ConvBlock(1024+512 -> 512)  │      │      │
          up + concat ──────────────────────┘      │      │
                                                   |      |
          dec3: ConvBlock( 512+256 -> 256)         │      │
          up + concat ─────────────────────────────┘      │
                                                          |
          dec2: ConvBlock( 256+128 -> 128)                │
          up + concat ────────────────────────────────────┘

          dec1: ConvBlock( 128+ 64 ->  64)
          Output: Conv1x1(64 -> 1) + Sigmoid

跳跃连接的作用：

编码器每下采样一次，分辨率减半，位置信息就丢一部分，语义强但像素位置关系非常模糊。这个任务要求逐像素的修改，跳跃连接把编码器的高分辨率特征直接送到解码器，补充了相对原始的位置信息。

与 ResNet 残差连接的作用区别：这里是为了传递位置信息，ResNet 是为了传递梯度
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


from utils import count_parameters  # noqa: E402

__all__ = ["UNet", "ConvBlock", "build_model_from_config", "count_parameters"]


class ConvBlock(nn.Module):
    """Structure: 3*3 Conv-> BatchNorm-> ReLU-> 3*3 Conv-> BatchNorm-> ReLU

    padding=1 的 same padding 使得解码器和编码器的特征图尺寸对上
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """用于图像到图像恢复的 U-Net

    Args:
        base_channels: 第一层的通道数，之后每下采样一级翻倍
        depth:         下采样次数。输入边长需要能被 16 整除(2^4)
        in_channels:   输入通道数。灰度图 1
        out_channels:  输出通道数。灰度图 1
        bilinear:      上采样方式。False 用转置卷积，True 用双线性插值+卷积
    """

    def __init__(
        self,
        base_channels: int = 64,
        depth: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = False,
    ) -> None:
        super().__init__()

        self.base_channels = base_channels
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # encoder
        channels = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = ConvBlock(prev, bottleneck_channels)

        # decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = bottleneck_channels
        for ch in reversed(channels):
            if bilinear:
                # 双线性插值+卷积
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                # 转置卷积：kernel=2、stride=2，正好把边长翻倍
                self.ups.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch + ch, ch))  # 跳跃连接拼接
            prev = ch

        # 输出层
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """卷积用 Kaiming，BatchNorm: weight=1/bias=0"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 形状 (N, 1, H, W)
        Returns:
            形状 (N, 1, H, W)，取值在 [0,1]
        """
        # 编码
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # 解码：上采样 -> 拼接跳跃连接 -> ConvBlock
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"上采样后尺寸 {tuple(x.shape[-2:])} 与跳跃连接 {tuple(skip.shape[-2:])} 不一致，"
                    f"请把输入边长补齐到 2**{self.depth}={2 ** self.depth} 的整数倍"
                )
            x = decoder(torch.cat([skip, x], dim=1))

        return self.activation(self.head(x))

    def config(self) -> dict:
        """返回重建这个模型所需的结构参数，供保存 checkpoint 时记录。"""
        return {
            "arch": "unet",
            "base_channels": self.base_channels,
            "depth": self.depth,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "bilinear": self.bilinear,
        }


def build_model_from_config(config: dict) -> UNet:
    """根据 checkpoint 里存的结构参数重建模型（推理时用）
    """
    return UNet(
        base_channels=config.get("base_channels", 64),
        depth=config.get("depth", 4),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        bilinear=config.get("bilinear", False),
    )


if __name__ == "__main__":
    # 直接运行本文件时，打印结构、参数量和一次前向的形状变化，方便自己核对。
    crop = 256
    dummy = torch.randn(2, 1, crop, crop)
    print(f"输入形状: {tuple(dummy.shape)}\n")

    for base in (32, 64):
        model = UNet(base_channels=base).eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"base_channels={base:<3} 参数量 {count_parameters(model):>11,}  "
              f"输出 {tuple(out.shape)}")

    model = UNet(base_channels=64).eval()
    print("\n各阶段形状变化（base_channels=64）:")
    with torch.no_grad():
        x = dummy[:1]
        print(f"  输入          {tuple(x.shape)}")
        skips = []
        for i, encoder in enumerate(model.encoders, start=1):
            x = encoder(x)
            skips.append(x)
            print(f"  enc{i}          {tuple(x.shape)}")
            x = model.pool(x)
        x = model.bottleneck(x)
        print(f"  bottleneck    {tuple(x.shape)}")
        for i, (up, decoder, skip) in enumerate(
            zip(model.ups, model.decoders, reversed(skips)), start=1
        ):
            x = up(x)
            print(f"  up{i}           {tuple(x.shape)}")
            x = decoder(torch.cat([skip, x], dim=1))
            print(f"  dec{i}         {tuple(x.shape)}")
        out = model.activation(model.head(x))
        print(f"  输出          {tuple(out.shape)}   取值 [{out.min():.3f}, {out.max():.3f}]")
