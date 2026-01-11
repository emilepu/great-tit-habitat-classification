import requests
import pandas as pd
import os
import time
from tqdm import tqdm
import subprocess
from pathlib import Path

# code to download the Great Tit recordings from Xeno-Canto
# and do initail quality, year and coordinate filtering

API_KEY = # replace with key from Xeno-canto
SPECIES = "Parus major"
OUTPUT_DIR = # replace with directory where recordings will be saved
MAX_RETRIES = 3  
os.makedirs(OUTPUT_DIR, exist_ok=True)
QUERY = f'sp:"{SPECIES}"'\

# -- get the recordings from xeno-canto ---
recordings = []
page = 1

while True:
    url = f'https://xeno-canto.org/api/3/recordings?query={QUERY}&page={page}&key={API_KEY}'
    resp = requests.get(url)
    if resp.status_code != 200:
        raise SystemExit(f"API request failed with status {resp.status_code}")

    data = resp.json()
    recs = data.get("recordings", [])
    if not recs:
        break
    recordings.extend(recs)

    if page >= data.get("numPages", 0):
        break
    page += 1

print(f"Number of recordings: {len(recordings)}")


# -- filtering ---
# filter for quality
df = pd.DataFrame(recordings)
if 'q' in df.columns: 
    df = df[df['q'].isin(['A', 'B'])]
    print(f"Num of recordings after quality filter: {len(df)}")

# filter for coordinates
coords_cols = [col for col in ['latitude', 'longitude', 'lat', 'lon'] if col in df.columns]
df = df.dropna(subset=coords_cols) 
print(f"Num recordings after coordinate filter: {len(df)}")

# filter for year
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date_parsed'].dt.year
df = df[df['year'] >= 2001] 
print(f"After filtering by year >= 2001: {len(df)} recordings")


# saving metadata
keep_cols = ["id", "gen", "sp", "en", "file", "rec", "cnt", "loc"] + coords_cols + ["alt", "type", "date", "time", "q", "length", "rmk", "year"]
keep_cols_existing = [col for col in keep_cols if col in df.columns]
df = df[keep_cols_existing]

metadata_file = os.path.join(OUTPUT_DIR, f"{SPECIES.replace(' ', '_')}_metadata.csv")
df.to_csv(metadata_file, index=False)


# --- download, verify, convert (written with the assistance of ChatGPT) ---
failed_log = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
corrupt_log = os.path.join(OUTPUT_DIR, "corrupt_files.txt")
wav_dir = os.path.join(OUTPUT_DIR, "wav_all")
os.makedirs(wav_dir, exist_ok=True)

with open(failed_log, "w") as flog, open(corrupt_log, "w") as clog:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rec_id = row["id"]
        file_url = row['file']
        if file_url.startswith("//"):
            file_url = f"https:{file_url}"
        mp3_path = os.path.join(OUTPUT_DIR, f"{rec_id}.mp3")
        wav_path = os.path.join(wav_dir, f"{rec_id}.wav")

        # Skip if WAV already exists
        if os.path.exists(wav_path):
            continue

        # Download MP3 with retries
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(file_url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:  # avoid tiny corrupt files
                    with open(mp3_path, "wb") as f:
                        f.write(r.content)
                    success = True
                    break
            except Exception as e:
                print(f"Retry {attempt + 1} for {rec_id} due to {e}")
                time.sleep(3)

        if not success:
            print(f"Failed to download {rec_id}")
            flog.write(f"{rec_id},{file_url}\n")
            continue

        # Check for corruption using ffmpeg
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", mp3_path, "-f", "null", "-"]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        except subprocess.CalledProcessError:
            print(f"Corrupt file detected: {mp3_path}")
            clog.write(f"{rec_id},{mp3_path}\n")
            continue

        # Convert to WAV
        try:
            cmd = ["ffmpeg", "-y", "-i", mp3_path, wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {mp3_path} to WAV: {e}")
            clog.write(f"{rec_id},{mp3_path}\n")
            continue

# --- remove mp3 files -----
for file in os.listdir(OUTPUT_DIR):
    if file.endswith(".mp3"):
        file_path = os.path.join(OUTPUT_DIR, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"could not delete {file_path}: {e}")

print("Done!")

