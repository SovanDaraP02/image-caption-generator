# Getting Flickr8k onto Colab (Day 1 task)

This sandbox can't reach Kaggle or Hugging Face, so this step has to happen on Colab itself, where you have full internet access. Two options below — pick whichever is easier for you.

## Option A: Hugging Face mirror (no API key needed — easiest)

```python
# in a Colab cell
!pip install -q datasets
from datasets import load_dataset
import os

ds = load_dataset("ariG23498/flickr8k", split="train")

os.makedirs("data/flickr8k/Images", exist_ok=True)
rows = []
for i, example in enumerate(ds):
    fname = f"img_{i}.jpg"
    example["image"].convert("RGB").save(f"data/flickr8k/Images/{fname}")
    rows.append((fname, example["text"]))  # check the actual column name -- print(ds[0]) first

# then build TRAIN_PAIRS / VAL_PAIRS from `rows` with an 80/10/10-ish split
```

Run `print(ds[0])` first to confirm the exact column names (`image`, and
either `text` or `caption` depending on how the mirror is structured) —
mirrors occasionally get re-uploaded with small schema changes.

## Option B: Kaggle API (original source, `adityajn105/flickr8k`)

1. Get a Kaggle API token: kaggle.com → account settings → "Create New API Token" → downloads `kaggle.json`
2. In Colab:

```python
from google.colab import files
files.upload()  # upload kaggle.json when prompted

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d adityajn105/flickr8k
!unzip -q flickr8k.zip -d data/flickr8k
```

This gives you `data/flickr8k/Images/` (8,091 jpgs) and
`data/flickr8k/captions.txt` (columns: `image,caption` — one row per
caption, 5 rows per image).

## Turning captions.txt into TRAIN_PAIRS / VAL_PAIRS

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/flickr8k/captions.txt")  # columns: image, caption
unique_images = df["image"].unique()

train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

def pairs_for(image_list):
    subset = df[df["image"].isin(image_list)]
    return list(zip(subset["image"], subset["caption"]))

TRAIN_PAIRS = pairs_for(train_imgs)
VAL_PAIRS = pairs_for(val_imgs)
TEST_PAIRS = pairs_for(test_imgs)
```

Drop these into `train.py` in place of the two empty `TRAIN_PAIRS` /
`VAL_PAIRS` lists, and you're ready to run it.
