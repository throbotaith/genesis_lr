# AnomalyGPT Dataset and Training

This folder contains utilities for preparing and training an AnomalyCLIP model using images of smartphones with cracked screens.

1. **datasets_phones/** – place 300 photos of broken-screen smartphones here (JPEG format).
2. **prepare_anomalyclip_dataset.py** – encodes all images with a CLIP model and saves the features.
3. **train.py** – fine-tunes a CLIP model on the images using the caption "a smartphone with a broken screen".

## Preparing features

Install dependencies (requires internet access):

```bash
pip install open_clip_torch
```

Then run:

```bash
python prepare_anomalyclip_dataset.py --image_dir datasets_phones --out anomalyclip_features.pt
```

## Training

To fine-tune the CLIP model on the dataset:

```bash
python train.py --image_dir datasets_phones --epochs 5 --out anomalyclip_finetuned.pt
```

The resulting weights will be saved to `anomalyclip_finetuned.pt`.
