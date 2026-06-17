"""
Segment objects in an image from a text prompt using SAM 3.

Usage:
  python scripts/segment_image.py --image /path/to/image.jpg --prompt "box"
  python scripts/segment_image.py --image image.jpg --prompt "dark green box" --output-dir outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM 3 image segmentation from a text prompt."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help='Text prompt to segment, for example "box" or "dark green box".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam3_image_segments"),
        help="Directory where masks and visualizations will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the input image stem.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local SAM 3 checkpoint path. If omitted, downloads from HF.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for returned masks.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity for masks in the overlay image.",
    )
    parser.add_argument(
        "--color-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random overlay colors.",
    )
    return parser.parse_args()


def as_numpy_mask(mask_tensor):
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask.astype(bool)


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_cutout(image, mask, path):
    cutout = np.array(image.convert("RGBA"))
    cutout[..., 3] = mask.astype(np.uint8) * 255
    Image.fromarray(cutout).save(path)


def generate_random_colors(num_colors, seed=None):
    rng = np.random.default_rng(seed)
    return [
        tuple(int(value) for value in rng.integers(32, 256, size=3))
        for _ in range(num_colors)
    ]


def save_overlay(image, masks, colors, path, alpha):
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    for mask, color_rgb in zip(masks, colors):
        color = np.array(color_rgb, dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    overlay_image.save(path)


def main():
    args = parse_args()
    image_path = args.image.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or image_path.stem

    image = Image.open(image_path).convert("RGB")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks_tensor = output["masks"]
    scores_tensor = output["scores"]
    boxes_tensor = output["boxes"]

    if len(scores_tensor) == 0:
        fallback_output = processor.set_confidence_threshold(0.0, state=state)
        fallback_scores = fallback_output["scores"].detach().cpu().numpy()
        if len(fallback_scores) == 0:
            print(
                f"No masks found for prompt {args.prompt!r}. "
                "No candidate confidence scores were returned."
            )
            raise RuntimeError(
                "No masks found. Try a different or more specific prompt."
            )
        best_fallback_idx = int(fallback_scores.argmax())
        best_fallback_score = float(fallback_scores[best_fallback_idx])
        print(
            f"No masks passed confidence threshold {args.confidence_threshold:.3f} "
            f"for prompt {args.prompt!r}."
        )
        print(
            f"Best candidate confidence below threshold: {best_fallback_score:.3f} "
            f"(candidate {best_fallback_idx})"
        )
        raise RuntimeError(
            "No masks found. Try a lower --confidence-threshold or a more specific "
            "prompt."
        )

    scores = scores_tensor.detach().cpu().numpy()
    boxes = boxes_tensor.detach().cpu().numpy()
    masks = [as_numpy_mask(mask) for mask in masks_tensor]
    best_idx = int(scores.argmax())

    best_mask_path = args.output_dir / f"{prefix}_best_mask.png"
    combined_mask_path = args.output_dir / f"{prefix}_combined_mask.png"
    cutout_path = args.output_dir / f"{prefix}_best_cutout.png"
    overlay_path = args.output_dir / f"{prefix}_overlay.png"
    metadata_path = args.output_dir / f"{prefix}_metadata.json"
    overlay_colors = generate_random_colors(len(masks), seed=args.color_seed)

    save_mask(masks[best_idx], best_mask_path)
    save_mask(np.logical_or.reduce(masks), combined_mask_path)
    save_cutout(image, masks[best_idx], cutout_path)
    save_overlay(image, masks, overlay_colors, overlay_path, args.overlay_alpha)

    instance_paths = []
    for idx, mask in enumerate(masks):
        instance_path = args.output_dir / f"{prefix}_instance_{idx:02d}_mask.png"
        save_mask(mask, instance_path)
        instance_paths.append(instance_path)

    instances = []
    for idx, (score, box, instance_path) in enumerate(
        zip(scores, boxes, instance_paths)
    ):
        instances.append(
            {
                "index": idx,
                "confidence": float(score),
                "box_xyxy": [float(value) for value in box],
                "mask_path": str(instance_path),
            }
        )

    metadata = {
        "image_path": str(image_path),
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "best_instance": best_idx,
        "best_confidence": float(scores[best_idx]),
        "best_mask_path": str(best_mask_path),
        "combined_mask_path": str(combined_mask_path),
        "best_cutout_path": str(cutout_path),
        "overlay_path": str(overlay_path),
        "overlay_colors_rgb": [list(color) for color in overlay_colors],
        "instances": instances,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Found {len(masks)} mask(s) for prompt: {args.prompt!r}")
    for idx, (score, box) in enumerate(zip(scores, boxes)):
        box_text = ", ".join(f"{value:.1f}" for value in box)
        print(f"  [{idx}] confidence={score:.3f}, box=[{box_text}]")
    print(f"Best instance: {best_idx}")
    print(f"Best confidence: {scores[best_idx]:.3f}")
    print(f"Best mask: {best_mask_path}")
    print(f"Combined mask: {combined_mask_path}")
    print(f"Best cutout: {cutout_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Metadata: {metadata_path}")
    print("Instance masks:")
    for instance_path in instance_paths:
        print(f"  {instance_path}")


if __name__ == "__main__":
    main()
