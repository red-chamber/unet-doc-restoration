"""配对文档数据集

数据布局
数据按批次分组，每组一个目录，里面是 input/output 两个子目录，文件名一一对应：

    <data_root>/
    ├── <分组 A>/dataset/{input,output}/
    ├── <分组 B>/dataset/{input,output}/
    └── <分组 C>/dataset/{input,output}/

input 是带手写内容的扫描件，output 是同一张图擦掉手写后的干净版本。

划分方式是按分组整体留出，而不是把图打散随机分。同一批扫描件之间高度相关
（同一套卷子的印刷版面一样、光照和噪声模式也一致），随机划分会让几乎相同的图
同时出现在训练集和测试集里，指标虚高。整组留出测的才是「换一批没见过的东西
还能不能行」。

特点：

1. input 的格式和通道数不统一：部分文件扩展名是 .jpg 但内容其实是带 alpha 的 PNG；
   另一部分是普通 JPEG 三通道彩色。output 则统一是单通道灰度 JPEG。所以 input 必须
   统一转灰度，带 alpha 的要先合成到白底
2. 尺寸差异极大：长边从一百多像素到两千像素，横竖版都有，长宽比各不相同
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_groups(data_root: str | Path) -> list[str]:
    """列出数据根目录下所有含 input/output 的分组名（按名称排序）。"""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{data_root}")

    groups = [
        p.name
        for p in sorted(data_root.iterdir())
        if p.is_dir() and (p / "dataset" / "input").is_dir()
    ]
    if not groups:
        raise FileNotFoundError(
            f"{data_root} 下没有找到任何分组目录。\n"
            f"期望布局：<data_root>/<分组名>/dataset/{{input,output}}/"
        )
    return groups


def scan_pairs(data_root: str | Path, dates: tuple[str, ...] | list[str]) -> list[tuple[Path, Path]]:
    """扫描指定日期下所有 input/output 配对，返回 [(input_path, output_path), ...]
    并保持顺序不变
    """
    data_root = Path(data_root)
    pairs: list[tuple[Path, Path]] = []

    for date in dates:
        input_dir = data_root / date / "dataset" / "input"
        output_dir = data_root / date / "dataset" / "output"
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Target dir does not exist: {input_dir}\n"
                f"Expected input: <data_root>/<date>/dataset/{{input,output}}/，"
            )

        # 两边都按文件名建索引，只保留同时存在的
        input_by_stem = {p.stem: p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
        output_stems = {p.stem for p in output_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}

        missing = sorted(set(input_by_stem) - output_stems)
        if missing:
            print(f"Warning: In {date} there are {len(missing)} wrong inputs.Skipped.")

        for stem in sorted(input_by_stem):
            if stem in output_stems:
                pairs.append((input_by_stem[stem], output_dir / f"{stem}.jpg"))

    if not pairs:
        raise FileNotFoundError(f"{data_root} 下没有扫到任何配对")

    return pairs


def load_gray(path: Path) -> Image.Image:
    """读图并统一成单通道灰度
    """
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("L")


class PairedDocDataset(Dataset):
    """配对文档数据集

    Args:
        pairs:      (input_path, output_path) 列表。
        crop_size:  大于 0 时随机裁剪成 crop_size*crop_size（训练用），为 0 时返回整图
        augment:    是否做随机缩放抖动。注意文档图不能水平翻转 —— 文字翻转后不再合理
        min_side:   图像最小边小于 crop_size 时，先等比放大到最小边等于 crop_size，否则裁不出完整窗口
        seed:       裁剪位置的随机源。DataLoader 会给每个 worker 派生不同的种子，但配合 utils.set_seed 可以复现
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        crop_size: int = 0,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.crop_size = crop_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        input_path, output_path = self.pairs[index]
        input_image = load_gray(input_path)
        target_image = load_gray(output_path)

        # 尺寸不一致时以 input 为准
        if target_image.size != input_image.size:
            target_image = target_image.resize(input_image.size, Image.BILINEAR)

        if self.crop_size:
            input_image, target_image = self._random_crop(input_image, target_image)

        # 转成 [0,1] 的 float32 张量，形状 (1, H, W)
        input_tensor = self._to_tensor(input_image)
        target_tensor = self._to_tensor(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": input_path.stem,
            "size": input_tensor.shape[-2:],
        }

    def _random_crop(self, input_image: Image.Image, target_image: Image.Image):
        """对输入和目标用同一个窗口裁剪"""
        crop = self.crop_size
        width, height = input_image.size

        # 最小边不够就先等比放大
        scale = 1.0
        if self.augment:
            scale *= random.uniform(0.85, 1.15)
        scale = max(scale, crop / min(width, height))
        if scale > 1.0:
            new_size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
            input_image = input_image.resize(new_size, Image.BILINEAR)
            target_image = target_image.resize(new_size, Image.BILINEAR)
            width, height = new_size

        left = random.randint(0, width - crop)
        top = random.randint(0, height - crop)
        box = (left, top, left + crop, top + crop)
        return input_image.crop(box), target_image.crop(box)

    @staticmethod
    def _to_tensor(image: Image.Image):
        array = np.asarray(image, dtype="float32") / 255.0
        return torch.from_numpy(array).unsqueeze(0)  # (H, W) -> (1, H, W)


def split_pairs(
    data_root: str | Path,
    *,
    val_ratio: float = 0.1,
    train_dates: tuple[str, ...] | list[str] | None = None,
    holdout_date: str | None = None,
) -> dict[str, list[tuple[Path, Path]]]:
    """按分组目录划分训练/验证/测试，返回 {"train": [...], "val": [...], "test": [...]}

    不指定 train_dates / holdout_date 时自动发现：按名称排序后把最后一组整个留出
    作验证+测试，其余各组用来训练。
    """
    if train_dates is None or holdout_date is None:
        groups = list_groups(data_root)
        if len(groups) < 2:
            raise ValueError(
                f"{data_root} 下只有 {len(groups)} 个分组，无法同时划出训练集和留出集，"
                f"至少需要两个分组目录"
            )
        holdout_date = holdout_date or groups[-1]
        train_dates = train_dates or [g for g in groups if g != holdout_date]

    train_pairs = scan_pairs(data_root, train_dates)
    holdout_pairs = scan_pairs(data_root, (holdout_date,))

    n_val = max(1, int(len(holdout_pairs) * val_ratio))
    return {
        "train": train_pairs,
        "val": holdout_pairs[:n_val],
        "test": holdout_pairs[n_val:],
    }


def build_loaders(
    splits: dict[str, list[tuple[Path, Path]]],
    *,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    augment: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, DataLoader]:
    """构建三组 DataLoader

    训练集：固定尺寸的裁剪块，可以批处理
    验证/测试集：整图，尺寸各不相同，所以 batch_size 固定为 1
    """
    loaders: dict[str, DataLoader] = {}

    loaders["train"] = DataLoader(
        PairedDocDataset(splits["train"], crop_size=crop_size, augment=augment, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    for name in ("val", "test"):
        loaders[name] = DataLoader(
            PairedDocDataset(splits[name], crop_size=0, seed=seed),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

    if verbose:
        for name in ("train", "val", "test"):
            print(f"{name:>5}: {len(splits[name]):,} 对")
        if augment:
            print("数据增强: 随机缩放抖动 0.85~1.15")

    return loaders
