"""
Image Classification Using a Pretrained Model (PyTorch)

Loads a small pretrained model (MobileNetV2, trained on ImageNet),
runs it on a handful of sample images, and prints the top predictions.

Usage:
    python classify.py                # classifies every image in ./images
    python classify.py my_photo.jpg   # classifies the images you pass in

The `load_model` and `classify` functions are also imported by app.py,
the web interface.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

IMAGES_DIR = Path(__file__).parent / "images"
TOP_K = 3  # how many guesses to show per image


def load_model():
    # Download (once) and load MobileNetV2 with weights already trained on ImageNet.
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.eval()  # inference mode: no training, no dropout
    # The weights object also gives us the matching preprocessing steps
    # and the 1000 human-readable ImageNet class names.
    return model, weights.transforms(), weights.meta["categories"]


def classify(image_source, model, preprocess, labels, top_k=TOP_K):
    """image_source can be a file path or any file-like object (e.g. an upload)."""
    image = Image.open(image_source).convert("RGB")

    # Resize / crop / normalize the image exactly the way the model expects,
    # then add a batch dimension: (3, 224, 224) -> (1, 3, 224, 224).
    batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():  # no gradients needed for prediction
        logits = model(batch)

    # Turn raw scores into probabilities that sum to 1.
    probs = torch.softmax(logits[0], dim=0)
    top_probs, top_ids = probs.topk(top_k)
    return [(labels[i], p.item()) for p, i in zip(top_probs, top_ids)]


def main():
    if len(sys.argv) > 1:
        image_paths = [Path(p) for p in sys.argv[1:]]
    else:
        image_paths = sorted(
            p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not image_paths:
        print(f"No images found. Put some .jpg/.png files in {IMAGES_DIR}")
        return

    model, preprocess, labels = load_model()

    for path in image_paths:
        print(f"\n{path.name}")
        for rank, (label, prob) in enumerate(classify(path, model, preprocess, labels), 1):
            print(f"  {rank}. {label:<25} {prob:6.1%}")


if __name__ == "__main__":
    main()
