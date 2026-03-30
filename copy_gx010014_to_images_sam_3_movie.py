from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    r"D:\maize_4_colmap_every_6_timesteps\colmap\frames_by_timestep_every6"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "images_and_masks"
    / "maize_4"
    / "images_sam_3_movie"
)
DEFAULT_IMAGE_PREFIX = "GX010014_"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy undistorted images for camera GX010014 from each timestep folder into "
            "images_and_masks/maize_4/images_sam_3_movie."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root folder containing timestep directories. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination folder for copied images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help=f"Filename prefix to copy. Default: {DEFAULT_IMAGE_PREFIX}",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing files from the output directory before copying.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output directory when names collide.",
    )
    return parser.parse_args()


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def iter_matching_images(images_dir: Path, image_prefix: str) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.startswith(image_prefix)
    )


def copy_images(
    source_root: Path,
    output_dir: Path,
    image_prefix: str,
    clear_output: bool,
    overwrite: bool,
) -> tuple[int, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if clear_output:
        clear_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_missing_dirs = 0

    timestep_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for timestep_dir in timestep_dirs:
        images_dir = timestep_dir / "undistorted" / "images"
        if not images_dir.is_dir():
            skipped_missing_dirs += 1
            print(f"Skipping {timestep_dir.name}: missing {images_dir}")
            continue

        matches = iter_matching_images(images_dir, image_prefix)
        if not matches:
            print(f"Skipping {timestep_dir.name}: no files found for prefix {image_prefix}")
            continue

        for source_image in matches:
            destination_path = output_dir / source_image.name
            if destination_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination file already exists: {destination_path}. "
                    "Re-run with --overwrite or --clear-output."
                )

            shutil.copy2(source_image, destination_path)
            copied_count += 1
            print(f"Copied {source_image} -> {destination_path}")

    return copied_count, skipped_missing_dirs


def main() -> int:
    args = parse_args()
    copied_count, skipped_missing_dirs = copy_images(
        source_root=args.source_root,
        output_dir=args.output_dir,
        image_prefix=args.image_prefix,
        clear_output=args.clear_output,
        overwrite=args.overwrite,
    )
    print(f"Done. Copied {copied_count} images into {args.output_dir}.")
    if skipped_missing_dirs:
        print(f"Skipped {skipped_missing_dirs} timestep folders because undistorted/images was missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
