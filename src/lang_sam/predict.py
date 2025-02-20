"""Run the prediction."""

from functools import reduce
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image


def predict(
    input_path: Path,
    output_path: Path,
    prompt: str = "vehicle",
    sam_type: str = "sam2.1_hiera_small",
    batch_size: int = 1,
):
    """Run the prediction on the input image/dir and save to the output path."""
    # Initialize the LangSAM model
    sam_model = LangSAM(sam_type=sam_type)

    if input_path.is_dir():
        input_images = list(sorted(input_path.rglob("*.*g")))
    else:
        input_images = [input_path]

    output_path.mkdir(exist_ok=True, parents=True)

    for batch_idx in range(0, len(input_images), batch_size):
        batch_paths = input_images[
            batch_idx : min(batch_idx + batch_size, len(input_images))
        ]
        batch_images = [Image.open(image_path) for image_path in batch_paths]
        results = sam_model.predict(batch_images, [prompt] * len(batch_images))
        for image, result, image_path in zip(batch_images, results, batch_paths):
            image_stem = image_path.stem
            np.savez_compressed(output_path / f"{image_stem}.npz", **result)
            masks = result["masks"]
            mask = reduce(np.logical_or, masks)
            Image.fromarray(np.uint8(mask) * 255.0).convert("L").save(
                output_path / f"{image_stem}_mask.png"
            )
            vis = draw_image(
                np.array(image),
                result["masks"],
                result["boxes"],
                result["scores"],
                result["labels"],
            )
            Image.fromarray(np.uint8(vis)).convert("RGB").save(
                output_path / f"{image_stem}_vis.png"
            )


def main():
    tyro.cli(predict)


if __name__ == "__main__":
    main()
