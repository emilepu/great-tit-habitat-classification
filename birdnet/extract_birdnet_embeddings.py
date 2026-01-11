#!/usr/bin/env python3

# code for extracting BirdNET embeddings from wav files

from pathlib import Path
import os
import shutil
from birdnet_analyzer.embeddings.core import embeddings

# --- directories and parameters ---
INPUT_DIR = Path("") # replace with preprocessed file directory
OUTPUT_DIR = Path("") # replace with output directory for embeddings
TEMP_DIR = Path("") # replace with directory for temporary files
THREADS = 8
FMIN, FMAX = 0, 15000
BATCH_SIZE = 16

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        print(f"No wav files found in {INPUT_DIR}")
        return
    print(f"Found {len(wavs)} recordings")
    
   
    for i, wav in enumerate(wavs, 1): # each file individually processed
        out_csv = OUTPUT_DIR / f"{wav.stem}_Embeddings.csv"
        # skip if already exists
        if out_csv.exists():
            print(f"[{i}/{len(wavs)}] skip {wav.name}")
            continue
        print(f"[{i}/{len(wavs)}] processing {wav.name}")
        
        temp_db = TEMP_DIR / f"temp_{wav.stem}" # temporary directory 
        temp_db.mkdir(parents=True, exist_ok=True)
        
        try:
            # extracting the embedding
            embeddings(audio_input=str(wav), database=str(temp_db), overlap=0.0, audio_speed=1.0, fmin=FMIN, fmax=FMAX, threads=THREADS, batch_size=BATCH_SIZE, file_output=str(temp_db / "embeddings.csv"),)
            # move from temp to output
            temp_csv = temp_db / "embeddings.csv"
            if temp_csv.exists():
                shutil.move(str(temp_csv), str(out_csv))
            else:
                # checking for different naming in case that happens
                csv_files = list(temp_db.glob("*.csv"))
                if csv_files:
                    shutil.move(str(csv_files[0]), str(out_csv))
                    print(f"Saved {out_csv.name}")
                else:
                    print(f"No csv for {wav.name}")
            
        except Exception as e:
            print(f"error: {e}")
        
        finally:
            if temp_db.exists(): # clean up temp directory
                shutil.rmtree(temp_db, ignore_errors=True)
    
    if TEMP_DIR.exists(): # clean up main temp directory
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print()
    print(f"{len(list(OUTPUT_DIR.glob('*_Embeddings.csv')))} total files")
    print("Done")

if __name__ == "__main__":
    main()