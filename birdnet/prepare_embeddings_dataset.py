#!/usr/bin/env python3

# code for preparing the dataset by filtering out short and low energy segments, applies cap for long recordings, and merges with metadata.
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


# ------ paths -------
EMB_DIR = Path("") # replace with embedding csv path
META_PATH = Path("") # replace with confident metadata path
OUT_PATH = Path("") # output path for merged parquet
RESULTS = Path("") # replace with path to save results
RESULTS.mkdir(parents=True, exist_ok=True)

# ----- parameters-----
LOW_ENERGY_PERCENTILE = 5
MIN_SEGMENT_LENGTH_S  = 1.5
MAX_SEGMENTS_PER_REC  = 100

# ----- some helper functions ---- 
def stem_to_id(stem: str) -> int: # gets the id from filename
    try:
        return int(stem.split("_")[0])
    except Exception:
        return None

def read_embedding_csv(path: Path) -> pd.DataFrame: # reads embeddings (written with LLM)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty CSV (no embeddings).")

    if {"Start (s)", "End (s)"}.issubset(df.columns):
        rec_id = stem_to_id(path.stem)
        df["recording_id"] = rec_id
        df["duration"] = df["End (s)"] - df["Start (s)"]
        return df

    elif {"start", "end", "embedding"}.issubset(df.columns):
        rec_id = stem_to_id(path.stem)
        df["recording_id"] = rec_id
        df.rename(columns={"start": "Start (s)", "end": "End (s)"}, inplace=True)
        df["duration"] = df["End (s)"] - df["Start (s)"]

        # Parse embedding column only if not all NaN/empty
        if df["embedding"].dropna().empty:
            raise ValueError("No embedding data in file.")

        import ast
        parsed = []
        for x in df["embedding"]:
            try:
                parsed.append(np.array(ast.literal_eval(x), dtype=np.float32))
            except Exception:
                parsed.append(np.empty((0,), dtype=np.float32))

        # Keep only non-empty vectors
        parsed = [p for p in parsed if p.size > 0]
        if not parsed:
            raise ValueError("No valid embeddings parsed.")

        mat = np.vstack(parsed)
        feat_cols = [f"Feature_{i}" for i in range(mat.shape[1])]
        feat_df = pd.DataFrame(mat, columns=feat_cols).reset_index(drop=True)
        df = df.loc[: len(feat_df) - 1].reset_index(drop=True)
        df = pd.concat([df.drop(columns=["embedding"]), feat_df], axis=1)
        return df

    else:
        raise ValueError(f"Unrecognized embedding format in {path.name}")

# ---- main function ---- 
def main():
    emb_files = sorted(EMB_DIR.glob("*_Embeddings.csv"))
    if not emb_files:
        raise SystemExit(f"No embedings found in {EMB_DIR}")

    frames = []
    for f in emb_files: #read the embeddings
        try:
            df = read_embedding_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"SKIPPED THE FILE {f.name}: {e}")
    all_emb = pd.concat(frames, ignore_index=True)

    # filtering the short embeddings
    before = len(all_emb)
    all_emb = all_emb[all_emb["duration"] >= MIN_SEGMENT_LENGTH_S].copy()
    print(f"Removed {before - len(all_emb)} short segments ")

    # filtering low energy embeddings
    feat_cols = [c for c in all_emb.columns if c.startswith("Feature_")]
    emb = all_emb[feat_cols].to_numpy(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    all_emb["emb_norm"] = norms
    thr = np.percentile(norms, LOW_ENERGY_PERCENTILE)
    all_emb = all_emb[all_emb["emb_norm"] >= thr].reset_index(drop=True)

    # limiting long recordings
    counts = all_emb["recording_id"].value_counts()
    too_long = counts[counts > MAX_SEGMENTS_PER_REC].index
    trimmed = []
    for rid in too_long:
        trimmed.append(all_emb[all_emb["recording_id"] == rid].iloc[:MAX_SEGMENTS_PER_REC])
    if trimmed:
        capped = pd.concat(trimmed)
        short = all_emb[~all_emb["recording_id"].isin(too_long)]
        all_emb = pd.concat([short, capped], ignore_index=True)
        print(f"long recordings: {len(too_long)}")

    # merge with metadata
    meta = pd.read_csv(META_PATH)
    meta = meta.rename(columns={"id": "recording_id"})
    keep_cols = ["recording_id", "habitat", "habitat_forest", "habitat_urban", "habitat_open"]
    keep_cols = [c for c in keep_cols if c in meta.columns]
    merged = all_emb.merge(meta[keep_cols], on="recording_id", how="inner")

    # save the merged dataset
    merged.to_parquet(OUT_PATH, index=False)

    # ----- some summary plots done wiith an llm -----
    # 1) Energy histogram
    plt.figure()
    plt.hist(merged["emb_norm"], bins=60)
    plt.xlabel("Embedding L2 norm"); plt.ylabel("Count")
    plt.title("Embedding energy distribution")
    plt.tight_layout(); plt.savefig(RESULTS / "energy_hist.png", dpi=180); plt.close()

    # 2) Duration histogram
    plt.figure()
    plt.hist(merged["duration"], bins=50)
    plt.xlabel("Segment duration (s)"); plt.ylabel("Count")
    plt.title("Segment duration distribution")
    plt.tight_layout(); plt.savefig(RESULTS / "duration_hist.png", dpi=180); plt.close()

    # 3) Class balance (dominant habitat per recording)
    if {"habitat_forest","habitat_urban","habitat_open"}.issubset(merged.columns):
        rec_soft = merged.groupby("recording_id")[["habitat_forest","habitat_urban","habitat_open"]].mean()
        dom = rec_soft.idxmax(axis=1).value_counts().sort_index()
        plt.figure()
        plt.bar(dom.index, dom.values)
        plt.xlabel("Dominant habitat (recording-level)")
        plt.ylabel("Number of recordings")
        plt.title("Class balance after filtering")
        plt.tight_layout(); plt.savefig(RESULTS / "class_balance_bar.png", dpi=180); plt.close()

    # 4) Segments per recording
    seg_counts = merged["recording_id"].value_counts()
    plt.figure()
    plt.hist(seg_counts.values, bins=50)
    plt.xlabel("Segments per recording")
    plt.ylabel("Number of recordings")
    plt.title("Segments per recording (post-filter)")
    plt.tight_layout(); plt.savefig(RESULTS / "segments_per_recording_hist.png", dpi=180); plt.close()

    print(f"📊 Saved summary plots in {RESULTS}")

# ----------------------

if __name__ == "__main__":
    main()
