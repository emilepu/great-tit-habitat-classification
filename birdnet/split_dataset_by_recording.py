#!/usr/bin/env python3

# code for splitting the dataset by recording ids into train/val/test sets


from pathlib import Path
import pandas as pd
import numpy as np

#---- paths and parameters ----
IN_DIR = Path("") # replace with merged parquet path
SPLIT_INFO_OUT = Path("") # replace with output split info path

TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.15, 0.15
SEED = 42

# ---- main -----
def main():
    df = pd.read_parquet(IN_DIR)
    print(f"Loaded merged dataset: {len(df):,} segments from {df['recording_id'].nunique():,} recordings")

    # find the dominant habitat per recording
    rec_level = (
        df.groupby("recording_id")[["habitat_forest","habitat_urban","habitat_open"]]
        .mean()
        .idxmax(axis=1)
        .rename("dominant_class")
    )
    rec_df = rec_level.reset_index()

    # stratified split 
    rng = np.random.default_rng(SEED)
    train_ids, val_ids, test_ids = [], [], []

    for cls, sub in rec_df.groupby("dominant_class"):
        ids = sub["recording_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * TRAIN_PCT)
        n_val   = int(n * VAL_PCT)
        train_ids += ids[:n_train].tolist()
        val_ids   += ids[n_train:n_train + n_val].tolist()
        test_ids  += ids[n_train + n_val:].tolist()
        print(f"{cls:<8} -> train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

    # save split info 
    split_df = pd.DataFrame({
        "recording_id": train_ids + val_ids + test_ids,
        "split":       ["train"] * len(train_ids)
                       + ["val"] * len(val_ids)
                       + ["test"] * len(test_ids)
    })
    split_df.to_csv(SPLIT_INFO_OUT, index=False)

    check = split_df.merge(rec_df, on="recording_id")
    dist = check.groupby(["split","dominant_class"]).size().unstack(fill_value=0)
    print("\nClass counts per split:")
    print(dist)

if __name__ == "__main__":
    main()
