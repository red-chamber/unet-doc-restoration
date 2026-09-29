"""单张 / 批量推理

这是这个项目最终的实用入口：给一张带手写的文档照片，输出擦掉手写后的干净图。

    # 单张
    python src/infer.py --checkpoint checkpoints/unet_best.pt --input photo.jpg

    # 整个目录
    python src/infer.py --checkpoint checkpoints/unet_best.pt \
        --input-dir /path/to/dataset/<留出组>/dataset/input --limit 10

如果输入路径里带有 `.../input/...`，脚本会自动到同级的 `output/` 目录找对应的
真值图，找到就把 PSNR / SSIM 一起算出来并画四列对比图；找不到（真实使用场景下
本来就没有真值）就只输出预测图。

产出：
    reports/samples/<名称>_pred.png          预测的干净图（可直接用）
    reports/samples/<名称>_compare.png       四列对比图（有真值时才生成）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


from checkpoint import load_weights  # noqa: E402
from utils import PROJECT_ROOT, count_parameters  # noqa: E402
from data import IMAGE_EXTS, load_gray  # noqa: E402
from metrics import psnr, ssim  # noqa: E402
from model import build_model_from_config  # noqa: E402
from predict import predict_full  # noqa: E402
from viz import plot_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用训练好的 U-Net 擦除文档上的手写内容")
    p.add_argument("--checkpoint", type=str, required=True, help="权重路径")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=str, help="单张图片路径")
    source.add_argument("--input-dir", type=str, help="批量推理的目录")

    p.add_argument("--limit", type=int, default=0, help="批量模式下最多处理多少张，0 表示全部")
    p.add_argument("--tile", type=int, default=0,
                   help="分块推理的块边长，0 表示整图直推；显存不够时设 512")
    p.add_argument("--out-dir", type=str, default=None,
                   help="输出目录，默认 reports/samples/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def find_target(image_path: Path) -> Path | None:
    """输入路径里含 input 目录时，去同级 output 目录找同名真值图。

    数据集的组织方式是 <日期>/dataset/{input,output}/<同名>.jpg，
    所以这个推断在数据集上总能命中；换成真实照片时自然返回 None。
    """
    parts = list(image_path.parts)
    if "input" not in parts:
        return None

    index = len(parts) - 1 - parts[::-1].index("input")
    candidate_dir = Path(*parts[:index]) / "output"
    for ext in (".jpg", ".jpeg", ".png", image_path.suffix):
        candidate = candidate_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_input(image_path: Path) -> tuple[torch.Tensor, Image.Image]:
    """读入并转成 (1, 1, H, W) 的 [0,1] 张量；同时返回原图供直接保存。"""
    image = load_gray(image_path)
    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor, image


def save_prediction(prediction: torch.Tensor, out_path: Path) -> None:
    """把预测保存成 8 位灰度 PNG。

    用 PNG 而不是 JPEG：JPEG 有损，会在印刷字边缘和刚擦干净的区域再压出伪影，
    看起来像是没擦干净。这份图是最终产物，不该再引入额外损失。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    array = prediction[0, 0].detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((array * 255.0).round().astype("uint8"), mode="L").save(out_path)


def process_one(
    model: torch.nn.Module,
    image_path: Path,
    out_dir: Path,
    device: str,
    tile: int,
) -> dict | None:
    """处理一张图，返回指标（无真值时返回 None）。"""
    input_tensor, _ = load_input(image_path)
    prediction = predict_full(model, input_tensor, device=device, tile=tile)

    save_prediction(prediction, out_dir / f"{image_path.stem}_pred.png")

    target_path = find_target(image_path)
    if target_path is None:
        print(f"{image_path.name:<40} {tuple(input_tensor.shape[-2:])}  已输出预测图（无真值，跳过指标）")
        return None

    target_tensor, _ = load_input(target_path)
    if target_tensor.shape != prediction.shape:
        # 真值和输入尺寸不一致（数据集里没有这种情况，但真实数据可能有）
        target_tensor = F.interpolate(target_tensor, size=prediction.shape[-2:], mode="nearest")

    score_psnr = psnr(prediction, target_tensor)
    score_ssim = ssim(prediction, target_tensor)
    print(f"{image_path.name:<40} {tuple(input_tensor.shape[-2:])}  "
          f"PSNR {score_psnr:>6.2f} dB  SSIM {score_ssim:.4f}")

    plot_comparison(
        [{
            "name": image_path.stem,
            "input": input_tensor[0, 0].numpy(),
            "target": target_tensor[0, 0].numpy(),
            "pred": prediction[0, 0].numpy(),
            "psnr": score_psnr,
            "ssim": score_ssim,
        }],
        out_dir / f"{image_path.stem}_compare.png",
        f"Inference: {image_path.name}",
        max_rows=1,
    )
    return {"name": image_path.stem, "psnr": score_psnr, "ssim": score_ssim}


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "reports" / "samples"

    print("=" * 72)
    print("文档手写擦除推理")
    print("=" * 72)
    model, ckpt = load_weights(Path(args.checkpoint), args.device, build_model_from_config)
    print(f"权重      : {args.checkpoint}")
    print(f"训练轮次  : {ckpt.get('epoch', 'unknown')}")
    print(f"验证 PSNR : {ckpt.get('val_psnr', float('nan')):.2f} dB")
    print(f"验证 SSIM : {ckpt.get('val_ssim', float('nan')):.4f}")
    print(f"参数量    : {count_parameters(model):,}")
    print(f"输出目录  : {out_dir}")
    if args.tile > 0:
        print(f"分块推理  : 块边长 {args.tile}")
    print("-" * 72)

    if args.input:
        paths = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"不是目录：{input_dir}")
        paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.limit > 0:
            paths = paths[: args.limit]
        if not paths:
            raise SystemExit(f"{input_dir} 下没有找到图片")

    records = [process_one(model, path, out_dir, args.device, args.tile) for path in paths]
    scored = [record for record in records if record is not None]

    if scored:
        mean_psnr = float(np.mean([r["psnr"] for r in scored]))
        mean_ssim = float(np.mean([r["ssim"] for r in scored]))
        print("-" * 72)
        print(f"共 {len(scored)} 张有真值的图：平均 PSNR {mean_psnr:.2f} dB，平均 SSIM {mean_ssim:.4f}")
    print(f"结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
