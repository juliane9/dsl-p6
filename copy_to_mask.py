from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    r"D:\maize_4_colmap_every_6_timesteps\colmap\frames_by_timestep_every6"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "to_mask"
DEFAULT_TIMESTEPS = ("1002", "2004", "3006")
DEFAULT_CAMERAS = ("GX010014", "GX010074", "GX010324")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected undistorted camera images into a local 'to_mask' folder. "
            "By default this creates 3 timestep folders with 3 images each."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root folder that contains the timestep directories. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination root for the copied images. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


def find_camera_image(images_dir: Path, camera_prefix: str) -> Path:
    matches = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.startswith(f"{camera_prefix}_")
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find an image starting with '{camera_prefix}_' in {images_dir}"
        )
    return matches[0]


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def copy_images(source_root: Path, output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    for timestep in DEFAULT_TIMESTEPS:
        timestep_name = f"timestep_{timestep}"
        images_dir = source_root / timestep_name / "undistorted" / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing undistorted images folder: {images_dir}")

        destination_dir = output_root / timestep_name
        clear_directory(destination_dir)

        for camera in DEFAULT_CAMERAS:
            source_image = find_camera_image(images_dir, camera)
            destination_path = destination_dir / source_image.name
            shutil.copy2(source_image, destination_path)
            copied_count += 1
            print(f"Copied {source_image.name} -> {destination_path}")

    return copied_count


def main() -> int:
    args = parse_args()
    copied_count = copy_images(args.source_root, args.output_root)
    print(f"Done. Created {len(DEFAULT_TIMESTEPS)} folders and copied {copied_count} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
