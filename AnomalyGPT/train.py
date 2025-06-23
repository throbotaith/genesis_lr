import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

try:
    import open_clip
except Exception:
    raise SystemExit('open_clip_torch is required. Install with `pip install open_clip_torch`.')

CAPTION = "a smartphone with a broken screen"

class PhoneDataset(Dataset):
    def __init__(self, root: Path, preprocess):
        self.paths = sorted(Path(root).glob('*.jpg'))
        if not self.paths:
            raise ValueError(f'No jpg images found in {root}')
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.preprocess(img)
        return img, CAPTION

def train(image_dir: Path, epochs: int, out: Path, model_name="ViT-B-32", pretrained="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset = PhoneDataset(image_dir, preprocess)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    loss_fn = open_clip.loss.CLIPLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        for images, text in loader:
            tokens = tokenizer(list(text))
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            loss = loss_fn(image_features, text_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    torch.save(model.state_dict(), out)
    print(f"Model saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, default=Path('datasets_phones'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path('anomalyclip_finetuned.pt'))
    args = parser.parse_args()
    train(args.image_dir, args.epochs, args.out)

if __name__ == '__main__':
    main()
