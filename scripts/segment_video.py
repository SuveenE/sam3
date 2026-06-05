"""
Segment an object in a video from a text prompt using SAM 3.

Instead of segmenting a single image, this script walks through a video frame by
frame looking for the requested object. It skips the first frame, then tries to
detect the object on a candidate frame. If the best detection clears the
confidence threshold (default 0.5) the results for that frame are written out and
the script stops. Otherwise it jumps ahead a few frames (default 5) and tries
again.

Usage:
  python scripts/segment_video.py --video /path/to/clip.mp4 --prompt "box"
  python scripts/segment_video.py --video clip.mp4 --prompt "dark green box" \
      --frame-step 5 --confidence-threshold 0.5 --output-dir outputs
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Thickness (in pixels) of the colored outline drawn around each mask.
BORDER_PX = 3
OVERLAY_COLORS = {
    "green": (0, 255, 0),
    "pink": (255, 0, 180),
}

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM 3 segmentation over a video, frame by frame, until "
        "the prompted object is detected above a confidence threshold."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help='Text prompt to segment, for example "box" or "dark green box".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam3_video_segments"),
        help="Directory where masks and visualizations will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the input video stem.",
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
        help="Minimum best-mask score required to accept a frame.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First frame index to evaluate. Defaults to 1 so the first frame "
        "(index 0) is skipped.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="How many frames to jump ahead after a frame fails to clear the "
        "confidence threshold.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Optional cap on the number of frames to evaluate before giving up.",
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
    parser.add_argument(
        "--overlay-color",
        default="green",
        choices=["green", "pink", "random"],
        help="Color to use for mask overlays. Defaults to green.",
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


def get_overlay_colors(num_colors, color_name, seed=None):
    if color_name == "random":
        return generate_random_colors(num_colors, seed=seed)
    return [OVERLAY_COLORS[color_name]] * num_colors


def draw_mask_border(overlay, mask, color_rgb):
    """Outline the mask with a clean, bright contour in the mask's color."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(
        overlay, contours, -1, tuple(int(c) for c in color_rgb), BORDER_PX
    )


def save_overlay(image, masks, colors, path, alpha):
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    for mask, color_rgb in zip(masks, colors):
        color = np.array(color_rgb, dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    for mask, color_rgb in zip(masks, colors):
        draw_mask_border(overlay, mask, color_rgb)

    Image.fromarray(overlay).save(path)


def read_frame(capture, frame_index):
    """Return the RGB PIL image at frame_index, or None if it cannot be read."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def detect_in_frame(processor, image, prompt):
    """Run the prompt on a single frame.

    Returns a tuple of (masks, scores, boxes, best_idx, best_score). When nothing
    is returned the masks/scores/boxes lists are empty and best_score is the best
    sub-threshold candidate (or None if there were no candidates at all).
    """
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    scores_tensor = output["scores"]
    if len(scores_tensor) == 0:
        fallback_output = processor.set_confidence_threshold(0.0, state=state)
        fallback_scores = fallback_output["scores"].detach().cpu().numpy()
        best_score = (
            float(fallback_scores.max()) if len(fallback_scores) else None
        )
        return [], np.array([]), np.array([]), None, best_score

    scores = scores_tensor.detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    masks = [as_numpy_mask(mask) for mask in output["masks"]]
    best_idx = int(scores.argmax())
    return masks, scores, boxes, best_idx, float(scores[best_idx])


def write_results(args, image, masks, scores, boxes, best_idx, prefix):
    best_mask_path = args.output_dir / f"{prefix}_best_mask.png"
    combined_mask_path = args.output_dir / f"{prefix}_combined_mask.png"
    cutout_path = args.output_dir / f"{prefix}_best_cutout.png"
    overlay_path = args.output_dir / f"{prefix}_overlay.png"
    frame_path = args.output_dir / f"{prefix}_frame.png"
    metadata_path = args.output_dir / f"{prefix}_metadata.json"
    overlay_colors = get_overlay_colors(
        len(masks), args.overlay_color, seed=args.color_seed
    )

    image.save(frame_path)
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
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "best_instance": best_idx,
        "best_confidence": float(scores[best_idx]),
        "frame_path": str(frame_path),
        "best_mask_path": str(best_mask_path),
        "combined_mask_path": str(combined_mask_path),
        "best_cutout_path": str(cutout_path),
        "overlay_path": str(overlay_path),
        "overlay_color": args.overlay_color,
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
    print(f"Frame image: {frame_path}")
    print(f"Best mask: {best_mask_path}")
    print(f"Combined mask: {combined_mask_path}")
    print(f"Best cutout: {cutout_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Metadata: {metadata_path}")
    print("Instance masks:")
    for instance_path in instance_paths:
        print(f"  {instance_path}")


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"

    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_prefix = args.output_prefix or video_path.stem

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    print(
        f"Searching video for {args.prompt!r} "
        f"(threshold={args.confidence_threshold:.2f}, "
        f"start_frame={args.start_frame}, frame_step={args.frame_step}, "
        f"total_frames={total_frames})"
    )

    frame_index = args.start_frame
    attempts = 0
    try:
        while total_frames <= 0 or frame_index < total_frames:
            if args.max_attempts is not None and attempts >= args.max_attempts:
                print(f"Reached --max-attempts ({args.max_attempts}). Stopping.")
                break

            image = read_frame(capture, frame_index)
            if image is None:
                print(f"Could not read frame {frame_index}. Stopping.")
                break

            attempts += 1
            masks, scores, boxes, best_idx, best_score = detect_in_frame(
                processor, image, args.prompt
            )

            if best_idx is not None and best_score >= args.confidence_threshold:
                print(
                    f"Frame {frame_index}: detected {args.prompt!r} with "
                    f"confidence {best_score:.3f} >= "
                    f"{args.confidence_threshold:.2f}."
                )
                prefix = f"{base_prefix}_frame{frame_index:06d}"
                write_results(
                    args, image, masks, scores, boxes, best_idx, prefix
                )
                return

            if best_score is None:
                print(
                    f"Frame {frame_index}: no candidates for {args.prompt!r}. "
                    f"Jumping ahead {args.frame_step} frame(s)."
                )
            else:
                print(
                    f"Frame {frame_index}: best confidence {best_score:.3f} < "
                    f"{args.confidence_threshold:.2f}. "
                    f"Jumping ahead {args.frame_step} frame(s)."
                )

            frame_index += args.frame_step
    finally:
        capture.release()

    raise RuntimeError(
        f"Could not detect {args.prompt!r} above confidence "
        f"{args.confidence_threshold:.2f} in any evaluated frame. "
        "Try a lower --confidence-threshold, a smaller --frame-step, or a "
        "different prompt."
    )


if __name__ == "__main__":
    main()
