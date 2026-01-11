#!/usr/bin/env python3

# Code for preprocessing for BirdNET embeddings 
# Does mono conversion, resampling, peak normalisation, and silence trimming


import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

#------ parameters and directories ------
INPUT_DIR = Path("") # replace with confident subset recording directory  
OUTPUT_DIR = Path("") # replace with output directory
TRIM_DB = 25.0
TARGET_SR = 48000
TARGET_DBFS = -3.0

# ------ helper functions ------ 
def normalize_peak(y, target_dbfs=-3.0): #peak normalisation
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    target_amp = 10 ** (target_dbfs / 20)
    return y * (target_amp / peak)

def preprocess_file(in_path, out_path, sr_target=48000, trim_db=25, target_dbfs=-3.0):
    try: # leads files, trims silence, normalises peak and saves as wav
        y, sr = librosa.load(in_path, sr=sr_target, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=trim_db)
        y_norm = normalize_peak(y_trimmed, target_dbfs)
        sf.write(out_path, y_norm, sr_target, subtype="PCM_16")
    except Exception as e:
        print(f"error processing {in_path.name}: {e}")

# ------ main script ------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in [".wav", ".mp3", ".flac"]]
    print(f"Found {len(files)} audio files in {INPUT_DIR}")

    for f in tqdm(files, desc="Preprocessing"):
        out_path = OUTPUT_DIR / (f.stem + ".wav")
        preprocess_file(f, out_path, sr_target=TARGET_SR, trim_db=TRIM_DB, target_dbfs=TARGET_DBFS)

    print(f"Done")

if __name__ == "__main__":
    main()
