"""推理辅助：尺寸补齐、分块推理、整数据集评估
"""

from __future__ import annotations

import torch

from metrics import image_stats, psnr, ssim, summarize


def _block_starts(length: int, tile: int, stride: int) -> list[int]:
    """计算分块的起始位置。

    按 stride 依次推进，并保证最后一块贴住右/下边界 —— 否则图幅不是
    stride 的整数倍时，最右和最下一小条会漏掉、在拼回去的结果里留下未覆盖区域。
    """
    last = max(0, length - tile)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """把右下方向补齐到 multiple 的整数倍，返回 (补齐后的张量, (pad_h, pad_w))。

    用 replicate 而不是补 0：文档背景通常是白的（接近 1），补 0 会在边界拉出一条
    黑框，卷积经过时会在附近区域产生虚假的边缘响应。
    """
    if multiple <= 1:
        return x, (0, 0)

    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    # F.pad 的最后一个维度是宽，倒数第二个是高；顺序是 (左, 右, 上, 下)
    padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: str = "cuda",
    tile: int = 0,
    overlap: int = 64,
) -> torch.Tensor:
    """对一张图（或一个 batch，尺寸相同）做推理，返回与输入同尺寸的预测。

    Args:
        image:  形状 (1, 1, H, W)，取值 [0,1]。
        tile:   大于 0 时启用分块推理，块边长。0 表示整图直推（显存够时更快）。
        overlap: 相邻块的重叠像素数，最后按重叠次数平均。
    """
    model.eval()
    multiple = 2**getattr(model, "depth", 4)
    image = image.to(device)

    if tile <= 0 or (image.shape[-2] <= tile and image.shape[-1] <= tile):
        padded, (pad_h, pad_w) = pad_to_multiple(image, multiple)
        output = model(padded)
        if pad_h or pad_w:
            output = output[..., : output.shape[-2] - pad_h, : output.shape[-1] - pad_w]
        return output.cpu()

    height, width = image.shape[-2:]
    stride = max(1, tile - overlap)

    # 累加所有块的结果再除以覆盖次数。用 float32 累加，避免多次相加的精度损失。
    # 累加器建在 image 所在的设备上（和 prediction 一致），最后再整体搬回 CPU ——
    # 否则就地相加会因为两个张量分处 CPU / GPU 而报错。
    accumulator = torch.zeros_like(image)
    counts = torch.zeros_like(image)

    ys = _block_starts(height, tile, stride)
    xs = _block_starts(width, tile, stride)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            window = image[..., y0:y1, x0:x1]

            padded, (pad_h, pad_w) = pad_to_multiple(window, multiple)
            prediction = model(padded)
            if pad_h or pad_w:
                prediction = prediction[..., : prediction.shape[-2] - pad_h, : prediction.shape[-1] - pad_w]

            accumulator[..., y0:y1, x0:x1] += prediction
            counts[..., y0:y1, x0:x1] += 1.0

    # 防止某处没被覆盖导致除零（正常情况下 counts 全为 1 以上）
    return (accumulator / counts.clamp(min=1.0)).cpu()


def evaluate_dataset(
    model: torch.nn.Module,
    loader,
    *,
    device: str = "cuda",
    tile: int = 0,
    max_images: int | None = None,
    keep_samples: int = 0,
    criterion=None,
) -> tuple[dict, list[dict], list[dict]]:
    """在给定数据集上跑一遍，返回 (汇总指标, 逐图指标, 用于画图的样本)。

    验证和测试走同一段代码，保证两边口径一致。

    Args:
        max_images:  只评估前 N 张（训练中途做验证时用它控制损失时间）。
        keep_samples: 保留前 N 张的图，供画对比图用。
        criterion:   传入时额外算一列整图损失，便于对照训练损失看是否过拟合。
    """
    per_image: list[dict] = []
    samples: list[dict] = []

    for index, batch in enumerate(loader):
        if max_images is not None and index >= max_images:
            break

        input_image = batch["input"]
        target_image = batch["target"]
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else batch["name"]

        prediction = predict_full(model, input_image, device=device, tile=tile)

        stats = image_stats(prediction)
        record = {
            "name": name,
            "psnr": psnr(prediction, target_image),
            "ssim": ssim(prediction, target_image),
            "pred_mean": stats["mean"],
            "pred_std": stats["std"],
            "is_flat": stats["is_flat"],
        }
        if criterion is not None:
            # 整图上的损失，和训练时的「随机小块损失」口径不同，但趋势可比
            with torch.no_grad():
                record["loss"] = float(criterion(prediction.to(device), target_image.to(device)).item())
        per_image.append(record)

        if len(samples) < keep_samples:
            samples.append(
                {
                    "name": name,
                    "input": input_image[0, 0].numpy(),
                    "target": target_image[0, 0].numpy(),
                    "pred": prediction[0, 0].numpy(),
                    "psnr": record["psnr"],
                    "ssim": record["ssim"],
                }
            )

        del input_image, target_image, prediction

    return summarize(per_image), per_image, samples
