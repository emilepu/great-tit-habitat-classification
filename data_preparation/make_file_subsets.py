import pandas as pd
from pathlib import Path
import shutil

# code to make a subset of recordings that are 
# at least 70% confident a habitat, and remove
# those that are over 50% Misc habitat

# progress bar
from tqdm import tqdm
HAS_TQDM = True


# --- input pats and configs ---
metadata_path = # replace with path of metadata CSV
recordings_root = # replace with path of all recordings
output_root = # replace with desired output path
misc_threshold = 0.5        
conf_threshold = 0.7  
log_file = output_root / "filtering_log.csv"
# --- output paths --- 
filtered_dir = output_root / "wav_filtered_all" # without misc
confident_dir = output_root / "wav_confident_subset" # only confident
deleted_dir = output_root / "wav_deleted_misc" #backup of deleted misc
for d in [filtered_dir, confident_dir, deleted_dir]:
    d.mkdir(parents=True, exist_ok=True)




# - prepare metadata variants
metadata = pd.read_csv(metadata_path)
habitat_cols = ['habitat_forest', 'habitat_urban', 'habitat_open', 'habitat_misc']
metadata[habitat_cols] = metadata[habitat_cols].apply(pd.to_numeric, errors='coerce')
metadata['id'] = metadata['id'].astype(str)
high_misc = metadata[metadata['habitat_misc'] > misc_threshold]
metadata_clean = metadata[metadata['habitat_misc'] <= misc_threshold].copy()
metadata_clean['max_habitat'] = metadata_clean[habitat_cols].max(axis=1)
confident = metadata_clean[metadata_clean['max_habitat'] > conf_threshold]
confident['dominant_habitat'] = confident[habitat_cols].idxmax(axis=1).str.replace('habitat_', '')

high_misc_ids = set(high_misc['id'])
confident_ids = set(confident['id'])
all_ids = set(metadata['id'])
mixed_ids = all_ids - high_misc_ids - confident_ids
print(f"High misc to remove: {len(high_misc_ids)} \n  confident to keep: {len(confident_ids)} \n  mixed: {len(mixed_ids)}")



# --- process the recordings ---

wav_files = list(recordings_root.glob("*.wav"))
processed_log = []
stats = {
    'deleted_misc': 0,
    'kept_mixed': 0,
    'kept_confident': 0,
    'no_metadata': 0,
    'already_processed': 0
}
if HAS_TQDM: #for progress bar
    iterator = tqdm(wav_files, desc="Filtering", ncols=100)
else:
    iterator = wav_files

for wav_file in iterator: # ---- was written with the help of an LLM ----
    file_id = wav_file.stem
    if file_id not in all_ids:
        stats['no_metadata'] += 1
        continue

    rec = metadata.loc[metadata['id'] == file_id].iloc[0]

    # deleting the high misc
    if file_id in high_misc_ids:
        dest = deleted_dir / wav_file.name
        if dest.exists():
            stats['already_processed'] += 1
        else:
            shutil.move(str(wav_file), str(dest))
            stats['deleted_misc'] += 1
        action = "deleted_misc"

    # --- copy confident to filtered and confident subsets
    elif file_id in confident_ids:
        dest_all = filtered_dir / wav_file.name
        dest_conf = confident_dir / wav_file.name

        copied = False
        if not dest_all.exists():
            shutil.copy(str(wav_file), str(dest_all))
            copied = True
        if not dest_conf.exists():
            shutil.copy(str(wav_file), str(dest_conf))
            copied = True

        if copied:
            stats['kept_confident'] += 1
        else:
            stats['already_processed'] += 1
        action = "kept_confident"

    # mixed copied to filtered
    else:
        dest = filtered_dir / wav_file.name
        if dest.exists():
            stats['already_processed'] += 1
        else:
            shutil.copy(str(wav_file), str(dest))
            stats['kept_mixed'] += 1
        action = "kept_mixed"

    processed_log.append({
        'id': file_id,
        'filename': wav_file.name,
        'action': action,
        'habitat': rec['habitat'],
        'habitat_forest': rec['habitat_forest'],
        'habitat_urban': rec['habitat_urban'],
        'habitat_open': rec['habitat_open'],
        'habitat_misc': rec['habitat_misc'],
        'max_habitat_prob': rec[habitat_cols].max(),
        'dominant_habitat': rec[habitat_cols].idxmax().replace('habitat_', '')
    })

log_df = pd.DataFrame(processed_log)
log_df.to_csv(log_file, index=False)

metadata_filtered = metadata[~metadata['id'].isin(high_misc_ids)]
metadata_confident = metadata[metadata['id'].isin(confident_ids)]
metadata_filtered.to_csv(output_root / "metadata_filtered.csv", index=False)
metadata_confident.to_csv(output_root / "metadata_confident.csv", index=False)

print("Done!")
print(f"Total WAV files processed: {len(wav_files)} \n Deleted misc files: {stats['deleted_misc']}\n Kept confident files: {stats['kept_confident']}\n Kept mixed: {stats['kept_mixed']}\n No metadata files: {stats['no_metadata']}\n Already processed files: {stats['already_processed']}")

