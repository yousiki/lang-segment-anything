# Language Segment-Anything

Language Segment-Anything is an open-source project that combines the power of instance segmentation and text prompts to generate masks for specific objects in images. Built on the recently released Meta model, Segment Anything Model 2, and the GroundingDINO detection model, it's an easy-to-use and effective tool for object detection and image segmentation.

![person.png](/assets/outputs/person.png)

## Features

- Zero-shot text-to-bbox approach for object detection.
- GroundingDINO detection model integration.
- SAM 2.1
- Batch inference support.
- Easy endpoint deployment using the Lightning AI litserve platform.
- Customizable text prompts for precise object segmentation.

## Getting Started

### Prerequisites

- Python 3.11 or higher

### Installation

#### Local Installation

Install python dependencies via [rye](https://rye.astral.sh/).

```bash
rye sync --no-lock --no-dev
```

#### Docker Installation

Build and run the image.

```bash
# Build docker image locally
docker build --t lang-segment-anything:latest .

# Or use pre-built image from ghcr
docker pull ghcr.io/yousiki/lang-segment-anything:latest
docker tag ghcr.io/yousiki/lang-segment-anything:latest lang-segment-anything:latest

# Start docker container
docker run --rm \
    --device=nvidia.com/gpu=all \
    -p 8000:8000 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -v $HOME/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
    -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    lang-segment-anything:latest
```

### Usage

#### Web

To run the gradio APP:

```bash
rye run app
```

And open `http://localhost:8000/gradio`

#### Command Line

```bash
rye run predict --input-path images --output-path results
```

#### Library

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])
```

## Examples

![car.png](/assets/outputs/car.png)

![fruits.png](/assets/outputs/fruits.png)

## Acknowledgments

**This repo is adapted from [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)** for personal usage.

This project is based on/used the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything-2)
- [LitServe](https://github.com/Lightning-AI/LitServe/)
- [Supervision](https://github.com/roboflow/supervision)

## License

This project is licensed under the Apache 2.0 License
