import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    labels_path = repo_root / "data" / "raw" / "labels.csv"
    images_dir = repo_root / "data" / "raw" / "images"
    splits_dir = repo_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    # Load labels
    df = pd.read_csv(labels_path)

    # ISIC 2018 Task 3 ground truth is usually one-hot columns: MEL, NV, BCC, AKIEC, BKL, DF, VASC
    required_cols = {"image", "MEL"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols} in labels.csv, found: {list(df.columns)[:20]}"
        )

    # Convert to binary: malignant = melanoma (MEL == 1), benign = everything else
    df = df[["image", "MEL"]].copy()
    df["label"] = df["MEL"].astype(int)  # 1 = melanoma, 0 = non-melanoma
    df.drop(columns=["MEL"], inplace=True)

    # Verify image files exist
    df["filepath"] = df["image"].apply(lambda x: str(images_dir / f"{x}.jpg"))
    missing = df[~df["filepath"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        # show a few missing to debug
        sample = missing.head(10)[["image", "filepath"]]
        raise FileNotFoundError(
            f"{len(missing)} images referenced in CSV not found on disk. Sample:\n{sample.to_string(index=False)}"
        )

    # Reproducible split: 70/15/15, stratified
    seed = 42
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    # Save
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print summary
    def summarize(name, part):
        total = len(part)
        pos = int(part["label"].sum())
        neg = total - pos
        print(f"{name:5s}: {total:5d}  malignant(MEL)= {pos:4d}  benign= {neg:4d}  pos%={pos/total:.3f}")

    print("\nSaved splits to:", splits_dir)
    summarize("train", train_df)
    summarize("val", val_df)
    summarize("test", test_df)


if __name__ == "__main__":
    main()