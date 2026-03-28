from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV is required to run this script. Install it with 'pip install opencv-python'."
    ) from exc


DEFAULT_SOURCE_ROOT = Path(
    r"D:\maize_4_colmap_every_6_timesteps\colmap\frames_by_timestep_every6"
)
DEFAULT_IMAGE_PREFIX = "GX010014_"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "movies"
DEFAULT_OUTPUT_NAME = "GX010014_timelapse.mp4"
DEFAULT_FPS = 24.0
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CODEC_CANDIDATES = ("mp4v",)
SCALE_CANDIDATES = (1.0, 0.85, 0.7, 0.5)


def print_progress(current: int, total: int, label: str) -> None:
    total = max(total, 1)
    width = 30
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
    if current >= total:
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a movie from the same frame file across timestep folders. "
            "The script looks inside each timestep's 'undistorted/images' directory."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root directory that contains the timestep folders. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help=(
            "Filename prefix to collect from each timestep folder. "
            f"Default: {DEFAULT_IMAGE_PREFIX}"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Frames per second for the output movie. Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder where the video should be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output video filename. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=100.0,
        help="Maximum preferred file size in MB before the script retries with more compression.",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=list(CODEC_CANDIDATES),
        help=(
            "Ordered list of codecs to try. Default avoids broken OpenH264 setups by using mp4v only."
        ),
    )
    return parser.parse_args()


def find_frame_file(images_dir: Path, image_prefix: str) -> Path | None:
    matches = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.stem.startswith(image_prefix)
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0] if matches else None


def collect_frame_paths(source_root: Path, image_prefix: str) -> tuple[list[Path], list[Path]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    frame_paths: list[Path] = []
    missing_images_dirs: list[Path] = []

    timestep_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    total_dirs = len(timestep_dirs)
    for index, timestep_dir in enumerate(timestep_dirs, start=1):
        images_dir = timestep_dir / "undistorted" / "images"
        if not images_dir.is_dir():
            missing_images_dirs.append(images_dir)
            print_progress(index, total_dirs, "Scanning folders")
            continue

        frame_path = find_frame_file(images_dir, image_prefix)
        if frame_path is not None:
            frame_paths.append(frame_path)
        print_progress(index, total_dirs, "Scanning folders")

    return frame_paths, missing_images_dirs


def round_down_to_even(value: int) -> int:
    return max(2, value - (value % 2))


def load_and_resize_image(image_path: Path, width: int, height: int):
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    return frame


def build_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    codec: str,
    scale: float,
) -> int:
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read first image: {frame_paths[0]}")

    height = round_down_to_even(int(first_frame.shape[0] * scale))
    width = round_down_to_even(int(first_frame.shape[1] * scale))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for codec '{codec}'.")

    try:
        total_frames = len(frame_paths)
        for index, image_path in enumerate(frame_paths, start=1):
            writer.write(load_and_resize_image(image_path, width, height))
            print_progress(index, total_frames, f"Encoding {codec} @ {scale:.2f}x")
    finally:
        writer.release()

    if not output_path.exists():
        raise RuntimeError(f"Video writer did not create an output file: {output_path}")

    return output_path.stat().st_size


def choose_best_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    max_output_bytes: int,
    codecs: list[str],
) -> tuple[Path, str, float, int]:
    temp_dir = output_path.parent / ".movie_build_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    best_candidate: Path | None = None
    best_codec = ""
    best_scale = 1.0
    best_size: int | None = None

    try:
        for scale in SCALE_CANDIDATES:
            scale_best_path: Path | None = None
            scale_best_codec = ""
            scale_best_size: int | None = None

            for codec in codecs:
                temp_output = temp_dir / f"{output_path.stem}_{codec}_{scale:.2f}{output_path.suffix}"
                if temp_output.exists():
                    temp_output.unlink()

                try:
                    size_bytes = build_video(frame_paths, temp_output, fps, codec, scale)
                except RuntimeError as exc:
                    print(f"Skipping codec {codec} at scale {scale:.2f}: {exc}")
                    continue

                if scale_best_size is None or size_bytes < scale_best_size:
                    if scale_best_path is not None and scale_best_path.exists():
                        scale_best_path.unlink()
                    scale_best_path = temp_output
                    scale_best_codec = codec
                    scale_best_size = size_bytes
                elif temp_output.exists():
                    temp_output.unlink()

            if scale_best_path is None or scale_best_size is None:
                continue

            if best_size is None or scale_best_size < best_size:
                if best_candidate is not None and best_candidate.exists():
                    best_candidate.unlink()
                best_candidate = scale_best_path
                best_codec = scale_best_codec
                best_scale = scale
                best_size = scale_best_size
            elif scale_best_path.exists():
                scale_best_path.unlink()

            if best_size is not None and best_size <= max_output_bytes:
                break

        if best_candidate is None or best_size is None:
            raise RuntimeError("Failed to build the movie with any available codec.")

        if output_path.exists():
            output_path.unlink()
        best_candidate.replace(output_path)
        return output_path, best_codec, best_scale, best_size
    finally:
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass


def main() -> int:
    args = parse_args()
    max_output_bytes = int(args.max_size_mb * 1024 * 1024)
    output_path = args.output_dir / args.output_name

    frame_paths, missing_images_dirs = collect_frame_paths(args.source_root, args.image_prefix)
    if not frame_paths:
        print(
            f"No frames starting with '{args.image_prefix}' were found under {args.source_root}.",
            file=sys.stderr,
        )
        return 1

    print(f"Collected {len(frame_paths)} frames for prefix '{args.image_prefix}'.")
    if missing_images_dirs:
        print(
            f"Skipped {len(missing_images_dirs)} timestep folders because "
            "'undistorted/images' was missing."
        )

    output_path, codec, scale, size_bytes = choose_best_video(
        frame_paths=frame_paths,
        output_path=output_path,
        fps=args.fps,
        max_output_bytes=max_output_bytes,
        codecs=args.codecs,
    )

    print(f"Saved movie to: {output_path}")
    print(f"Codec used: {codec}")
    print(f"FPS: {args.fps}")
    print(f"Scale used for final movie: {scale:.2f}")
    print(f"File size: {size_bytes / (1024 * 1024):.2f} MB")
    if size_bytes > max_output_bytes:
        print(
            f"Warning: final file is larger than {args.max_size_mb:.1f} MB even after retrying with compression."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
