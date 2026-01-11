# Great Tit habitat classification using deep learning
Author: Emilė Pundzevičiūtė
Bachelor thesis project (Tilburg University, Cognitive Science and AI)

## Data
Audio recordings were sourced from [Xeno-canto](https://xeno-canto.org/).
Habitat labels derived from MODIS and Dynamic World land cover datasets.

## Structure of running the code 
First, for the unified beginning (code in data_preparation folder):
1. download_recordings.py
2. habitat_matching.py - > needs GEE setup before
3. make_file_subsets.py 

Then, for 
(A) the CNN14 approach (cnn14 folder):
4. preprocessing.py (together with preprocess_config.yaml)
5. cnn14_finetune.py (with some parameters stored in run_finetune.sh)

(B) the BirdNET approach (birdnet folder):
4. preprocess_wav.py
5. extract_birdnet_embeddings.py
6. prepare_embeddings_dataset.py
7. split_dataset_by_recording.py
8. train_habitat_models_final.py

