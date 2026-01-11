#!/usr/bin/env python3

#code for CNN14 preprocessing
#Pipeline: mono conversion, resampling, silence trimming, (kept in an optional bandpass), segmentation, extracting spoctrogams, mean/std calculation and normalisation.

# Outputs the spectrograms as .npy files

import os
import sys
import yaml
import random
import platform
import importlib.metadata as metadata
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

# - setup --------
CONFIG_PATH = Path(__file__).parent / "preprocess_config.yaml" # adjust path to where preprocess_config.yaml is 
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

raw_dir = Path(cfg["input_dir"]).expanduser()
out_base = Path(cfg["output_dir"]).expanduser()
out_base.mkdir(parents=True, exist_ok=True)
variant_dir = out_base / "logmel_norm_global"
variant_dir.mkdir(parents=True, exist_ok=True)

seed = cfg.get("random_seed", 42)
np.random.seed(seed)
random.seed(seed)

versions = {"python": sys.version,"platform": platform.platform(),"librosa": metadata.version("librosa"),"numpy": metadata.version("numpy"),"scipy": metadata.version("scipy"),"soundfile": metadata.version("soundfile"),"yaml": metadata.version("PyYAML"),}
with open(out_base / "environment_versions.yaml", "w") as vf: # save versions
    yaml.safe_dump(versions, vf)
with open(out_base / "config_used.yaml", "w") as cf: # save config
    yaml.safe_dump(cfg, cf)

# --- get the files -----
wav_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() == ".wav"])
if cfg.get("max_files"):
    wav_files = wav_files[: int(cfg["max_files"])]
print(f"{len(wav_files)} files found")

# ---- functions ------
def butter_bandpass(lowcut, highcut, fs, order=4): #bandpass filter function
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass(x, fs, fmin, fmax, order=4): # applying bandpass filter funstion
    b, a = butter_bandpass(fmin, fmax, fs, order)
    return filtfilt(b, a, x)

def remove_silence(x, sr, thresh_db): #silence trimming function
    frame_length = int(0.025 * sr)   
    hop_length = int(0.010 * sr)   
    eps = 1e-8
    energies_db = []
    for i in range(0, len(x) - frame_length, hop_length):
        frame = x[i : i + frame_length]
        e = np.mean(frame**2)
        energies_db.append(10 * np.log10(e + eps))
    energies_db = np.asarray(energies_db)

    mask = energies_db > thresh_db
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return x

    start = max(0, idx[0] * hop_length)
    end = min(len(x), idx[-1] * hop_length + frame_length)
    return x[start:end]

def segment_audio(x, sr, segment_length, overlap): #audio segmentation function
    seg_samples = int(segment_length * sr)
    stride = int(seg_samples * (1 - overlap))

    if len(x) < seg_samples:
        pad_amount = seg_samples - len(x)
        x = np.pad(x, (0, pad_amount), mode="reflect")
        return [x]

    segments = []
    for start in range(0, len(x) - seg_samples + 1, stride):
        segments.append(x[start : start + seg_samples])
    return segments

def waveform_to_logmel(x, sr): # converting waveform to log-mel spectrogram function
    mel = librosa.feature.melspectrogram(y=x,sr=sr,n_fft=cfg["n_fft"],hop_length=cfg["hop_length"],win_length=cfg["win_length"],n_mels=cfg["n_mels"],fmin=cfg["fmin"],fmax=cfg["fmax"],window="hann",power=2.0,)
    logmel = np.log(mel + 1e-6).astype(np.float32)
    return logmel  

class RunningStats: # class for mean/std calculation
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / max(self.n - 1, 1)

    @property
    def std(self):
        return np.sqrt(self.var)

# --- computing global mean/std -----
stats = RunningStats()
print("\nComputing global mean/std")
for fpath in tqdm(wav_files, desc="Pass 1"):
    try:
        x, sr = librosa.load(fpath, sr=cfg["sample_rate"], mono=True)
        x = remove_silence(x, sr, cfg["silence_threshold_db"])

        if cfg.get("use_bandpass", False):
            x = apply_bandpass(x, sr,cfg["bandpass_fmin"], cfg["bandpass_fmax"],cfg["bandpass_order"])

        segments = segment_audio(x, sr, cfg["segment_length"], cfg["overlap"])
        for seg in segments:
            logmel = waveform_to_logmel(seg, sr)
            stats.update(logmel.flatten())
    except Exception as e:
        print(f"[Pass 1] Error processing {fpath.name}: {e}")

global_mean = float(stats.mean)
global_std = float(stats.std if stats.std > 1e-12 else 1.0)

np.savez(out_base / "global_mean_std.npz", mean=global_mean, std=global_std)
print(f"Global mean: {global_mean:.6f}, std: {global_std:.6f}")

# ---- normalise and save the spectrograms ----
print("\nNormalising and saving spectrograms")
for fpath in tqdm(wav_files, desc="Pass 2"):
    try:
        x, sr = librosa.load(fpath, sr=cfg["sample_rate"], mono=True)

        #remove silence 
        x = remove_silence(x, sr, cfg["silence_threshold_db"])

        # bandpass - not used i in the end
        if cfg.get("use_bandpass", False):
            x = apply_bandpass(x, sr,cfg["bandpass_fmin"], cfg["bandpass_fmax"],cfg["bandpass_order"])

        # segmenting audio
        segments = segment_audio(x, sr, cfg["segment_length"], cfg["overlap"])

        # converting to spectrograms
        for i, seg in enumerate(segments):
            logmel = waveform_to_logmel(seg, sr)
            logmel_norm = (logmel - global_mean) / (global_std + 1e-8)

            out_name = f"{fpath.stem}_seg{i}.npy"
            np.save(variant_dir / out_name, logmel_norm.astype(np.float32))

    except Exception as e:
        print(f"[Pass 2] Error processing {fpath.name}: {e}")

print("\nDone")

