"""Batch evaluation harness.

Walks a directory of video clips, runs each through the detector, writes
per-clip results to a CSV, and prints aggregate metrics at the end. Used
both for ad-hoc testing on your own data and as the entry point for the
benchmark work described in docs/ROADMAP.md.

USAGE:
    python scripts/eval_folder.py --input clips/ --output results.csv

    # When clip labels are encoded in the parent directory name
    # (e.g. clips/live/*.mp4 vs clips/spoof/*.mp4):
    python scripts/eval_folder.py --input clips/ --output results.csv \\
        --label-from parent

    # Process only the first N frames of each clip (faster iteration):
    python scripts/eval_folder.py --input clips/ --output results.csv \\
        --max-frames 200

The script is deliberately simple - no parallelism, no resumption, no
ROC curve. Those belong in a proper benchmark harness, which is a
roadmap item.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatic import LivenessDetector
from chromatic.config import configure_logging, load_settings
from chromatic.exceptions import FaceNotFoundError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

logger = logging.getLogger("chromatic.eval")


@dataclass
class ClipResult:
    """One row of the output CSV."""

    path: str
    label: str | None
    verdict: str           # "live", "reject", or "no_face"
    confidence: float
    sustained_confidence: float
    frames_processed: int
    hardware: float
    rppg: float
    texture: float
    geometry: float
    motion: float
    reasons: str
    duration_s: float


def _iter_clips(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _label_for(clip: Path, label_from: str, root: Path) -> str | None:
    if label_from == "none":
        return None
    if label_from == "parent":
        rel = clip.relative_to(root)
        # The first directory below the input root is the label.
        return rel.parts[0] if len(rel.parts) > 1 else None
    raise ValueError(f"unknown --label-from value: {label_from}")


def _process_one(detector: LivenessDetector, clip: Path,
                 max_frames: int | None) -> ClipResult:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return ClipResult(
            path=str(clip), label=None, verdict="no_face",
            confidence=0.0, sustained_confidence=0.0,
            frames_processed=0, hardware=0.0, rppg=0.0,
            texture=0.0, geometry=0.0, motion=0.0,
            reasons="could_not_open", duration_s=0.0,
        )

    t0 = time.perf_counter()
    detector.reset()

    last_diag = None
    face_seen = False
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                last_diag = detector.process_frame(frame)
                face_seen = True
            except FaceNotFoundError:
                pass
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()

    duration = time.perf_counter() - t0

    if last_diag is None or not face_seen:
        return ClipResult(
            path=str(clip), label=None, verdict="no_face",
            confidence=0.0, sustained_confidence=0.0,
            frames_processed=frame_idx, hardware=0.0, rppg=0.0,
            texture=0.0, geometry=0.0, motion=0.0,
            reasons="no_face_detected", duration_s=duration,
        )

    v = last_diag.verdict
    layer = v.layer_scores
    return ClipResult(
        path=str(clip),
        label=None,
        verdict="live" if v.is_live else "reject",
        confidence=float(v.confidence),
        sustained_confidence=float(v.sustained_confidence),
        frames_processed=frame_idx,
        hardware=float(layer.hardware),
        rppg=float(layer.rppg),
        texture=float(layer.texture),
        geometry=float(layer.geometry),
        motion=float(layer.motion),
        reasons="; ".join(v.reasons) if v.reasons else "",
        duration_s=duration,
    )


def _print_summary(results: list[ClipResult]) -> None:
    """Print aggregate metrics. When labels are present, computes TPR/FPR."""
    n = len(results)
    if n == 0:
        print("No clips processed.")
        return

    valid = [r for r in results if r.verdict != "no_face"]
    no_face = n - len(valid)

    print()
    print(f"Processed {n} clips ({no_face} with no detectable face).")
    if not valid:
        return

    live_predicted = sum(1 for r in valid if r.verdict == "live")
    reject_predicted = len(valid) - live_predicted
    print(f"  Predicted LIVE:   {live_predicted}")
    print(f"  Predicted REJECT: {reject_predicted}")

    labelled = [r for r in valid if r.label is not None]
    if not labelled:
        return

    # If labels are present, compute the confusion matrix.
    # Label conventions: "live"/"real" -> positive class.
    positive_labels = {"live", "real", "genuine", "1", "true"}
    tp = fp = fn = tn = 0
    for r in labelled:
        is_positive_truth = r.label.lower() in positive_labels
        is_positive_pred = r.verdict == "live"
        if is_positive_truth and is_positive_pred:
            tp += 1
        elif is_positive_truth and not is_positive_pred:
            fn += 1
        elif (not is_positive_truth) and is_positive_pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    if total == 0:
        return
    accuracy = (tp + tn) / total
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print()
    print("Confusion matrix (positive = LIVE):")
    print(f"  TP = {tp:4d}   FN = {fn:4d}")
    print(f"  FP = {fp:4d}   TN = {tn:4d}")
    print(f"  Accuracy = {accuracy:.3f}")
    print(f"  TPR (recall on real)   = {tpr:.3f}")
    print(f"  FPR (spoof acceptance) = {fpr:.3f}")
    print(f"  FNR (real rejection)   = {fnr:.3f}")
    print()
    print("Note: these numbers are valid only if your labels are reliable.")
    print("This script does not compute ROC curves - that is a roadmap item.")


def _write_csv(path: Path, results: list[ClipResult]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "path", "label", "verdict", "confidence", "sustained_confidence",
            "frames_processed", "hardware", "rppg", "texture", "geometry",
            "motion", "reasons", "duration_s",
        ])
        for r in results:
            writer.writerow([
                r.path, r.label or "", r.verdict,
                f"{r.confidence:.4f}", f"{r.sustained_confidence:.4f}",
                r.frames_processed,
                f"{r.hardware:.4f}", f"{r.rppg:.4f}", f"{r.texture:.4f}",
                f"{r.geometry:.4f}", f"{r.motion:.4f}",
                r.reasons, f"{r.duration_s:.2f}",
            ])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chromatic - batch evaluator")
    p.add_argument("--input", type=Path, required=True,
                   help="Directory of clips to evaluate (recursive).")
    p.add_argument("--output", type=Path, required=True,
                   help="CSV output path.")
    p.add_argument("--label-from", choices=["none", "parent"], default="none",
                   help="How to assign a label to each clip.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop processing each clip after N frames.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_settings()
    configure_logging(settings)
    logging.getLogger().setLevel(args.log_level)

    if not args.input.is_dir():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        return 2

    clips = list(_iter_clips(args.input))
    if not clips:
        print(f"No video files found under {args.input}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(clips)} clips from {args.input}")

    results: list[ClipResult] = []
    with LivenessDetector(settings) as detector:
        for i, clip in enumerate(clips, 1):
            label = _label_for(clip, args.label_from, args.input)
            print(f"  [{i:4d}/{len(clips)}] {clip.name}", end="", flush=True)
            result = _process_one(detector, clip, args.max_frames)
            result.label = label
            results.append(result)
            print(f"  -> {result.verdict}  conf={result.sustained_confidence:.2f}"
                  f"  ({result.duration_s:.1f}s)")

    _write_csv(args.output, results)
    print(f"\nWrote {args.output}")
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
