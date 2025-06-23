import argparse
from pathlib import Path
import torch
from PIL import Image

try:
    import open_clip
except Exception:
    raise SystemExit('open_clip_torch is required. Install with `pip install open_clip_torch`.')

TEXT_PROMPT = "a photo of a smartphone with a broken screen"

def encode_images(img_dir: Path, out_path: Path, model_name="ViT-B-32", pretrained="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No jpg images found in {img_dir}")
    feats = []
    with torch.no_grad():
        for p in image_paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            feats.append(model.encode_image(img).cpu())
    image_feats = torch.cat(feats)
    tokens = open_clip.tokenize([TEXT_PROMPT])
    with torch.no_grad():
        text_feat = model.encode_text(tokens).cpu()
    torch.save({"image_features": image_feats, "text_feature": text_feat}, out_path)
    print(f"Saved features to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, default=Path("datasets_phones"))
    parser.add_argument("--out", type=Path, default=Path("anomalyclip_features.pt"))
    args = parser.parse_args()
    encode_images(args.image_dir, args.out)

if __name__ == "__main__":
    main()
