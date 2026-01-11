#!/bin/bash
#
#SBATCH --job-name=cnn14_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/u222506/thesis_project/code/cnn14/results/%x_%j.out
#SBATCH --error=/home/u222506/thesis_project/code/cnn14/results/%x_%j.err

# CHANGE THE ABOVE OPTIONS AND PATHS AS NEEDED

echo "===== Training job started on $(hostname) at $(date) ====="
nvidia-smi
echo "==============================================="

# --- SETUP-----
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate thesis

cd /home/u222506/thesis_project/code/cnn14 # Change to code directory

# ---- PATHS -----
DATA_DIR="" # replace with path to recording directory
META_CSV="" # replace with path to metadata CSV file
OUT_DIR="" # replace with output directory
PRETRAINED="" # replace with path to pretrained model checkpoint

mkdir -p "$OUT_DIR"

#  ------ RUN ------
# dont forget to change path for cnn14_finetune.py 
python cnn14_finetune.py \
    --data_dir "$DATA_DIR" \
    --metadata "$META_CSV" \
    --out_dir "$OUT_DIR" \
    --pretrained_ckpt "$PRETRAINED" \
    --epochs 40 \
    --batch_size 32 \
    --lr 3e-5 \
    --weight_decay 1e-3 \
    --dropout 0.5 \
    --freeze_warmup_epochs 5 \
    --early_stop_patience 7 \
    --num_workers 6 \
    \
    --p_shift 0.6 \
    --shift_max_frac 0.05 \
    --p_gain 0.7 \
    --gain_min_db -3 \
    --gain_max_db 3 \
    --p_mask 0.6 \
    --time_mask_param 12 \
    --freq_mask_param 6 \
    --num_time_masks 1 \
    --num_freq_masks 1 \
    \
    --p_bg_mix 0.5 \
    --bg_mix_alpha 0.2 \
    \
    --p_mixup 0.5 \
    --mixup_alpha 0.2 \
    \
    --grad_clip 3.0 \
    --warmup_frac 0.05

echo "===== Training job finished at $(date) ====="