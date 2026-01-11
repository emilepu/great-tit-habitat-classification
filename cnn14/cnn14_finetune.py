#!/usr/bin/env python3

# code for finetuning the CNN14 with data augemntations

import os
import argparse
import random
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# --- loading the model ----
def get_cnn14_model(num_classes, pretrained_ckpt=None, device="cuda", dropout_p=0.5):
    pann_dir = Path(__file__).resolve().parent / "panns_cnn14" / "pytorch" # adjust path of cnn14
    models_py = pann_dir / "models.py"
    if not models_py.exists():
        raise FileNotFoundError(f"models.py not found at: {models_py}")

    sys.path.insert(0, str(pann_dir))
    from models import Cnn14
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=num_classes)

    if hasattr(model, "bn0"):
        model.bn0 = nn.BatchNorm2d(64)
        nn.init.ones_(model.bn0.weight)
        nn.init.zeros_(model.bn0.bias)

    if pretrained_ckpt:
        # loading pretrained weights
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        own_state = model.state_dict()
        matched, skipped = 0, 0
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                matched += 1
            else:
                skipped += 1
        print(f"loaded layers:{matched}, skipped layers: {skipped}")
        
    # disabling the built in feature extractor so i can use my own preprocessing
    if hasattr(model, "spectrogram_extractor"):
        model.spectrogram_extractor = nn.Identity()
    if hasattr(model, "logmel_extractor"):
        model.logmel_extractor = nn.Identity()

    # changing the final layer
    try:
        if hasattr(model, "fc"):
            in_f = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes))
        elif hasattr(model, "classifier"):
            in_f = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes))
    except Exception:
        pass

    return model.to(device)


# ------ data augmentation functions (done with Claude) -----------
def spec_time_shift(x, max_frac=0.05): # applies time shift to spectrogram
    if max_frac <= 0:
        return x
    _, _, T, _ = x.shape
    shift = int(random.uniform(-max_frac, max_frac) * T)
    return torch.roll(x, shifts=shift, dims=2)

def spec_gain(x, min_db=-3, max_db=3): # applies gain to spectrogram
    gain = 10 ** (random.uniform(min_db, max_db) / 20)
    return x * gain

def spec_mask(x, time_mask_param=12, freq_mask_param=6, 
              num_time_masks=1, num_freq_masks=1):
    B, C, T, F_ = x.shape
    out = x.clone()

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, T - t))
        out[:, :, t0:t0 + t, :] = 0

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, F_ - f))
        out[:, :, :, f0:f0 + f] = 0

    return out

def apply_augmentations(x, aug_cfg): # applies augmentations based on configs
    if not aug_cfg["enabled"]:
        return x
    out = x
    if random.random() < aug_cfg["p_shift"]:
        out = spec_time_shift(out, aug_cfg["shift_max_frac"])
    if random.random() < aug_cfg["p_gain"]:
        out = spec_gain(out, aug_cfg["gain_min_db"], aug_cfg["gain_max_db"])
    if random.random() < aug_cfg["p_mask"]:
        out = spec_mask(
            out,
            time_mask_param=aug_cfg["time_mask_param"],
            freq_mask_param=aug_cfg["freq_mask_param"],
            num_time_masks=aug_cfg["num_time_masks"],
            num_freq_masks=aug_cfg["num_freq_masks"],
        )
    return out

def mixup_batch(x, y, alpha=0.2, p=0.5): # batch mixup augmentation function
    if alpha <= 0 or random.random() > p:
        return x, y
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], lam * y + (1 - lam) * y[index]

def background_mix(x, y_soft, file_ids,
                   id_to_domhab_idx, id_to_label_dist,
                   by_hab_files, data_dir,
                   alpha=0.2, p=0.5): # background mixing augmentation
    if random.random() > p:
        return x, y_soft

    B = x.size(0)
    x2 = x.clone()
    y2 = y_soft.clone()
    data_dir = Path(data_dir)

    for i in range(B):
        id_i = file_ids[i]
        hab_i = id_to_domhab_idx[id_i]

        other_habs = [h for h in by_hab_files.keys()
                      if h != hab_i and len(by_hab_files[h]) > 0]
        if not other_habs:
            continue

        hab_j = random.choice(other_habs)
        fname_j = random.choice(by_hab_files[hab_j])

        arr_j = np.load(data_dir / fname_j)
        if arr_j.ndim != 2:
            continue
        if arr_j.shape[0] != 64 and arr_j.shape[1] == 64:
            arr_j = arr_j.T
        if arr_j.shape[0] != 64:
            continue

        t_j = torch.tensor(arr_j, dtype=torch.float32).T.unsqueeze(0).to(x.device)

        lam = random.uniform(0.0, alpha)
        x2[i] = (1 - lam) * x2[i] + lam * t_j

        id_j = Path(fname_j).stem.split("_")[0]
        if id_j in id_to_label_dist:
            dist_j = id_to_label_dist[id_j]
            yj = torch.tensor(
                [dist_j[lbl] for lbl in ["Forest", "Urban", "Open"]],
                dtype=torch.float32,
                device=y2.device
            )
            y2[i] = (1 - lam) * y2[i] + lam * yj

    return x2, y2
# ------ --------- ----------



# ---- dataset class ---- 
class NpySegmentDatasetSoftLabels(Dataset):
    def __init__(self, data_dir, file_list, id_to_label_dist, label_order):
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.id_to_label_dist = id_to_label_dist
        self.label_order = label_order

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        arr = np.load(self.data_dir / fname)

        if arr.ndim != 2:
            raise ValueError(f"Bad spec shape {arr.shape} for {fname}.")
        if arr.shape[0] != 64:
            if arr.shape[1] == 64:
                arr = arr.T
            else:
                raise ValueError(f"Unexpected mel dimension in {fname}: {arr.shape}")

        t = torch.tensor(arr, dtype=torch.float32).T.unsqueeze(0) 

        id_part = Path(fname).stem.split("_")[0]
        dist = self.id_to_label_dist[id_part]
        y = torch.tensor([dist[lbl] for lbl in self.label_order], dtype=torch.float32)

        return t, y, fname, id_part


# --- class for soft cross-entropy loss ---- 
class SoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


# ---- evaluation functions ----
def evaluate_epoch_soft(y_true_soft, y_prob, label_order):
    y_true_hard = np.argmax(y_true_soft, axis=1)
    y_pred_hard = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true_hard, y_pred_hard)
    f1 = f1_score(y_true_hard, y_pred_hard, average="macro")

    aucs = []
    for c in range(y_prob.shape[1]):
        if np.sum(y_true_soft[:, c] > 0) >= 5:
            try:
                aucs.append(roc_auc_score(y_true_soft[:, c], y_prob[:, c]))
            except Exception:
                pass
    auc_mean = np.mean(aucs) if aucs else np.nan

    kl = np.mean(np.sum(
        y_true_soft * np.log((y_true_soft + 1e-10) / (y_prob + 1e-10)),
        axis=1
    ))

    cm = confusion_matrix(y_true_hard, y_pred_hard, labels=[0, 1, 2])
    return dict( accuracy=float(acc), f1=float(f1), auc=float(auc_mean), kl_divergence=float(kl), confmat=cm.tolist())

def evaluate_recording_level_soft(y_true_soft, y_prob, rec_ids, label_order): # evaluate at recording level
    rec_ids = np.asarray(rec_ids)
    unique_ids = np.unique(rec_ids)

    y_prob_rec, y_true_rec = [], []
    for rid in unique_ids:
        mask = rec_ids == rid
        prob_mean = y_prob[mask].mean(axis=0)
        true_mean = y_true_soft[mask].mean(axis=0)
        prob_mean = prob_mean/(prob_mean.sum() + 1e-8)
        true_mean = true_mean/(true_mean.sum() + 1e-8)

        y_prob_rec.append(prob_mean)
        y_true_rec.append(true_mean)

    y_prob_rec = np.vstack(y_prob_rec)
    y_true_rec = np.vstack(y_true_rec)
    return evaluate_epoch_soft(y_true_rec, y_prob_rec, label_order)


# ---- split functions ---- 
def build_id_splits(df, seed=42, test_size=0.15, val_size=0.15):
    ids = df["id"].astype(str)
    labels = df["habitat"].astype(str)

    id_train_val, id_test, y_train_val, y_test = train_test_split(ids, labels, test_size=test_size, stratify=labels, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    id_train, id_val, _, _ = train_test_split(id_train_val, y_train_val, test_size=val_rel, stratify=y_train_val, random_state=seed)
    return dict(train_ids=id_train.tolist(), val_ids=id_val.tolist(), test_ids=id_test.tolist())

def files_for_ids(data_dir, ids):
    p = Path(data_dir)
    out = []
    for i in set(ids):
        out += [f.name for f in p.glob(f"{i}_seg*.npy")]
    return sorted(out)


# ------ lr scheduler with warmup class -----
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / max(self.warmup_steps, 1)
            else:
                progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                lr = 0.5 * base_lr * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs

def freeze_backbone(model):
    for n, p in model.named_parameters():
        if "fc" not in n and "classifier" not in n:
            p.requires_grad = False

def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = True


#-------------------- training loop -------------------------
def train_and_evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(Path(args.metadata).expanduser())
    req = ["id", "habitat", "habitat_forest", "habitat_urban", "habitat_open", "habitat_misc"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    df["id"] = df["id"].astype(str)
    df["habitat"] = df["habitat"].astype(str)

    # ---------------- 3-class setup  ----------------
    label_order = ["Forest", "Urban", "Open"]
    label_to_idx = {l: i for i, l in enumerate(label_order)}

    id_to_label_dist = {}
    for _, r in df.iterrows():
        f = float(r["habitat_forest"])
        u = float(r["habitat_urban"])
        o = float(r["habitat_open"])
        s = f + u + o
        if s <= 0:
            f, u, o = 1/3, 1/3, 1/3
        else:
            f /= s
            u /= s
            o /= s

        id_to_label_dist[str(r["id"])] = {
            "Forest": f,
            "Urban": u,
            "Open": o,
        }

    # identify dominant habitat for each id
    id_to_domhab_idx = {}
    for i_, dist in id_to_label_dist.items():
        dom_lbl = max(dist, key=dist.get)
        id_to_domhab_idx[i_] = label_to_idx[dom_lbl]

    splits = build_id_splits(df, seed=args.seed, test_size=args.test_size, val_size=args.val_size)
    train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

    train_files = files_for_ids(args.data_dir, train_ids)
    val_files = files_for_ids(args.data_dir, val_ids)
    test_files = files_for_ids(args.data_dir, test_ids)
    print(f"Split: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    train_ds = NpySegmentDatasetSoftLabels(args.data_dir, train_files, id_to_label_dist, label_order)
    val_ds = NpySegmentDatasetSoftLabels(args.data_dir, val_files, id_to_label_dist, label_order)
    test_ds = NpySegmentDatasetSoftLabels(args.data_dir, test_files, id_to_label_dist, label_order)

    # balanced sampling based on dominant habitat
    dom_indices = []
    for f in train_files:
        id_part = Path(f).stem.split("_")[0]
        dom_indices.append(id_to_domhab_idx[id_part])
    counts = Counter(dom_indices)
    weights = torch.tensor([1.0 / counts[i] for i in dom_indices], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_cnn14_model(len(label_order), args.pretrained_ckpt, device, args.dropout)

    if args.freeze_warmup_epochs > 0:
        print(f"freezing backbone for {args.freeze_warmup_epochs} epochs")
        freeze_backbone(model)

    loss_fn = SoftCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineLR(optimizer,warmup_steps=int(args.warmup_frac * total_steps),total_steps=total_steps)

    aug_cfg = dict(
        enabled=not args.no_augment,
        p_shift=args.p_shift, shift_max_frac=args.shift_max_frac,
        p_gain=args.p_gain, gain_min_db=args.gain_min_db, gain_max_db=args.gain_max_db,
        p_mask=args.p_mask, time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        num_time_masks=args.num_time_masks, num_freq_masks=args.num_freq_masks,
    )

    by_hab_files = defaultdict(list)
    for fname in train_files:
        id_part = Path(fname).stem.split("_")[0]
        hab = id_to_domhab_idx[id_part]
        by_hab_files[hab].append(fname)

    best_val, patience = -1.0, 0
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_warmup_epochs + 1:
            print("---- unfreezing backbone")
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = WarmupCosineLR(
                optimizer,
                warmup_steps=int(args.warmup_frac * total_steps),
                total_steps=total_steps
            )

        # ---------------- train ----------------
        model.train()
        tr_losses, tr_true, tr_pred = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for xb, yb, fnames, id_parts in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            xb = apply_augmentations(xb, aug_cfg)
            xb, yb = background_mix(xb, yb, id_parts, id_to_domhab_idx=id_to_domhab_idx, id_to_label_dist=id_to_label_dist, by_hab_files=by_hab_files, data_dir=args.data_dir, alpha=args.bg_mix_alpha, p=args.p_bg_mix)
            xb, yb = mixup_batch(xb, yb, alpha=args.mixup_alpha, p=args.p_mixup)

            out = model(xb)
            logits = out.get("clipwise_output") if isinstance(out, dict) else out
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            tr_losses.append(loss.item())

            probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
            tr_pred.append(probs)
            tr_true.append(yb.detach().cpu().numpy())

            pbar.set_postfix({"loss": float(np.mean(tr_losses))})

        tr_metrics = evaluate_epoch_soft(np.vstack(tr_true), np.vstack(tr_pred), label_order)

        # ---------------- validation (recording level) ----------------
        model.eval()
        val_losses, val_true, val_pred = [], [], []
        val_rec_ids = []

        with torch.no_grad():
            for xb, yb, fnames, id_parts in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                logits = out.get("clipwise_output") if isinstance(out, dict) else out

                val_losses.append(loss_fn(logits, yb).item())
                val_pred.append(torch.softmax(logits.detach().cpu(), dim=1).numpy())
                val_true.append(yb.detach().cpu().numpy())
                val_rec_ids += list(id_parts)

        val_metrics = evaluate_recording_level_soft(np.vstack(val_true), np.vstack(val_pred), val_rec_ids, label_order)
        epoch_log = dict(epoch=epoch, train_loss=float(np.mean(tr_losses)), val_loss=float(np.mean(val_losses)), train_f1=tr_metrics["f1"], val_f1=val_metrics["f1"], train_acc=tr_metrics["accuracy"], val_acc=val_metrics["accuracy"], train_kl=tr_metrics["kl_divergence"], val_kl=val_metrics["kl_divergence"],)
        metrics_log.append(epoch_log)
        print(f"\nEpoch {epoch} summary: {epoch_log}")

        if val_metrics["f1"] > best_val + 1e-4:
            best_val = val_metrics["f1"]
            patience = 0
            ckpt_path = Path(args.out_dir) / f"best_epoch{epoch}_valf1{best_val:.4f}.pt"
            torch.save({"epoch": epoch,"model_state": model.state_dict(),"optimizer_state": optimizer.state_dict(),"label_order": label_order,"val_f1": best_val,}, ckpt_path)
            print("BEST CHECKPOINT SAVED:", ckpt_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print("EARLY STOP")
                break

        pd.DataFrame(metrics_log).to_csv(Path(args.out_dir) / "metrics_log.csv", index=False)

    # Load best checkpoint before test evaluation
    best_ckpt = sorted(Path(args.out_dir).glob("best_epoch*.pt"))[-1]
    print(f"Loading best checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # ---------------- test (recording-level) ----------------
    print("\nTest set evaluation")
    model.eval()
    test_true, test_pred = [], []
    test_rec_ids = []

    with torch.no_grad():
        for xb, yb, fnames, id_parts in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            logits = out.get("clipwise_output") if isinstance(out, dict) else out

            test_pred.append(torch.softmax(logits.detach().cpu(), dim=1).numpy())
            test_true.append(yb.detach().cpu().numpy())
            test_rec_ids += list(id_parts)

    test_metrics = evaluate_recording_level_soft(
        np.vstack(test_true),
        np.vstack(test_pred),
        test_rec_ids,
        label_order
    )
    print("Test metrics:", test_metrics)

    pd.DataFrame([test_metrics]).to_csv(Path(args.out_dir) / "test_metrics.csv", index=False)
    print("Done")


# ----- passing arguments -----
def parse_args():
    p = argparse.ArgumentParser(description="Cnn14 training with soft labels")

    p.add_argument("--data_dir", required=True, help="Dir with normalized .npy spectrogram segments")
    p.add_argument("--metadata", required=True, help="CSV with soft habitat labels")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)

    p.add_argument("--freeze_warmup_epochs", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=7)

    p.add_argument("--no_augment", action="store_true")

    p.add_argument("--p_shift", type=float, default=0.6)
    p.add_argument("--shift_max_frac", type=float, default=0.05)

    p.add_argument("--p_gain", type=float, default=0.7)
    p.add_argument("--gain_min_db", type=float, default=-3)
    p.add_argument("--gain_max_db", type=float, default=3)

    p.add_argument("--p_mask", type=float, default=0.6)
    p.add_argument("--time_mask_param", type=int, default=12)
    p.add_argument("--freq_mask_param", type=int, default=6)
    p.add_argument("--num_time_masks", type=int, default=1)
    p.add_argument("--num_freq_masks", type=int, default=1)

    p.add_argument("--p_bg_mix", type=float, default=0.5)
    p.add_argument("--bg_mix_alpha", type=float, default=0.2)

    p.add_argument("--p_mixup", type=float, default=0.5)
    p.add_argument("--mixup_alpha", type=float, default=0.2)

    p.add_argument("--grad_clip", type=float, default=3.0)

    p.add_argument("--warmup_frac", type=float, default=0.05)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_and_evaluate(args)
