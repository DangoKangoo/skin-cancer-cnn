from pathlib import Path
import sys

# Add repo root to Python path so "import src..." works
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from torch.utils.data import DataLoader
from src.data.dataset import ISICBinaryDataset

train_csv = repo / "data" / "splits" / "train.csv"

ds = ISICBinaryDataset(train_csv, split="train")
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

x, y = next(iter(dl))
print("x shape:", tuple(x.shape))  # (8, 3, 224, 224)
print("y shape:", tuple(y.shape))  # (8,)
print("y:", y.tolist())