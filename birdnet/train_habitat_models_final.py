#!/usr/bin/env python3

#code for training BirdNET embedding classifiers


from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
try: #importing xgboost
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost is not installed") from e


# ------- color palette ------
COLORS = {
    "gold": "#cebb43",
    "green": "#91ae6b",
    "navy": "#2c3343",
    "sage": "#bec98e",
    "bluegrey": "#a7b3c7",
}
HABITAT_COLORS = {
    "forest": COLORS["green"],
    "urban": COLORS["gold"],
    "open": COLORS["navy"],
}

# ------- paths ------
MERGED_PARQUET = Path("") # replace with merged parquet path
SPLIT_FILE = Path("") # replace with split info path
RESULTS_DIR = Path("") # replace with path to save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#------ parameters -----
SEED = 42
TRAIN_BATCH = 256
LR_MLP = 3e-4  
EPOCHS = 80    
PATIENCE = 7  
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.15
np.random.seed(SEED)
torch.manual_seed(SEED)
LABEL_COLS_3 = ["habitat_forest", "habitat_urban", "habitat_open"]
LABEL_COLS_4 = ["habitat_forest", "habitat_urban", "habitat_open", "habitat_misc"]
LABEL_NAMES = ["forest", "urban", "open"]


# ---- expanding soft labels (written with an llm) ----
def expand_soft_labels(X, y_soft, class_weights=None, eps=1e-8):
    """
    Expand each sample into K samples (K=3 classes).
    For each original x_i with soft probs p_i:
      create (x_i, label=c, weight=p_i[c] * class_weight[c]) for c in {0,1,2}.
    """
    K = y_soft.shape[1]
    weights = y_soft.reshape(-1)  # length N*K
    X_rep = np.repeat(X, K, axis=0)
    y_rep = np.tile(np.arange(K), reps=len(y_soft))
    
    # Apply class weights if provided
    if class_weights is not None:
        class_weight_rep = np.tile(class_weights, reps=len(y_soft))
        weights = weights * class_weight_rep
    
    mask = weights > eps
    return X_rep[mask], y_rep[mask], weights[mask]
# ----------------------

# ------ MLP model ------
class MLPclassifier(nn.Module):
    def __init__(self, in_dim=1024, hidden=(128, 64), out_dim=3, p=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden[1], out_dim)
        )
    def forward(self, x):
        return self.net(x)

#------ weighted focal loss (written with an LLM) ------
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        
        if self.alpha is not None:
            # Apply per-class weights
            alpha_t = self.alpha.to(inputs.device).unsqueeze(0)
            focal = alpha_t * focal
            
        return focal.mean()

# ------ mixup data augmentation function------
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y


# ----- aggregation function -----
def aggregate_recording(df_probs):
    agg = df_probs.groupby("recording_id")[LABEL_NAMES].mean()
    agg["pred"] = agg[LABEL_NAMES].idxmax(axis=1)
    return agg

# ----- evaluation functions (written with llm) -----
def kl_divergence(true_soft, pred_soft):
    return np.mean(np.sum(
        true_soft * np.log((true_soft + 1e-10) / (pred_soft + 1e-10)),
        axis=1
    ))

def evaluate_split(proba_seg, y_soft_seg, rec_ids, name):
    seg_df = pd.DataFrame({
        "recording_id": rec_ids,
        **{LABEL_NAMES[i]: proba_seg[:, i] for i in range(3)}
    })
    rec = aggregate_recording(seg_df)

    # recording-level true soft labels
    rec_true_soft = []
    for rid in rec.index:
        first_idx = np.where(rec_ids == rid)[0][0]
        rec_true_soft.append(y_soft_seg[first_idx])
    rec_true_soft = np.vstack(rec_true_soft)

    rec_pred_soft = rec[LABEL_NAMES].to_numpy()

    rec_true_hard = rec_true_soft.argmax(axis=1)
    rec_pred_hard = rec_pred_soft.argmax(axis=1)

    acc = accuracy_score(rec_true_hard, rec_pred_hard)
    f1  = f1_score(rec_true_hard, rec_pred_hard, average="macro")
    f1_per_class = f1_score(rec_true_hard, rec_pred_hard, average=None, labels=[0,1,2])
    kl  = kl_divergence(rec_true_soft, rec_pred_soft)

    cm  = confusion_matrix(rec_true_hard, rec_pred_hard, labels=[0,1,2])

    print(f"{name:5s} - acc={acc:.3f}, f1={f1:.3f}, kl={kl:.3f} | "
          f"F1 per class: forest={f1_per_class[0]:.3f}, urban={f1_per_class[1]:.3f}, open={f1_per_class[2]:.3f}")
    
    return acc, f1, kl, cm, rec_pred_soft, f1_per_class


def evaluate_recording_level_mlp(model, X, y_soft_seg, rec_ids, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
    acc, f1, kl, _, _, f1_per_class = evaluate_split(probs, y_soft_seg, rec_ids, "Val")
    return acc, f1, kl, f1_per_class


def analyze_class_distribution(y_soft, split_mask, split_name): # class distribution analysis
    y_split = y_soft[split_mask]
    counts = y_split.argmax(axis=1)
    
    print(f"\n{split_name.upper()} CLASS DISTRIBUTION:")
    for i, name in enumerate(LABEL_NAMES):
        n = (counts == i).sum()
        pct = 100 * n / len(counts)
        print(f"  {name:8s}: {n:6d} recordings ({pct:5.1f}%)")
    
    return counts
# -------------------------------

# ------ plotting functions (also written with an llm)------
def plot_embedding_space(X, y_hard, save_path):
    """Visualize embedding space with PCA."""
    print("\n🔍 Visualizing embedding space with PCA...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    for i, name in enumerate(LABEL_NAMES):
        mask = y_hard == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=name.capitalize(), alpha=0.4, s=15, c=HABITAT_COLORS[name])
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('BirdNET Embedding Space (PCA Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_detailed_confusion_matrix(cm, f1_score_val, save_path, title_suffix=""):
    """Plot both count and normalized confusion matrices side by side."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Forest","Urban","Open"],
                yticklabels=["Forest","Urban","Open"],
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f"Confusion Matrix - Counts\n{title_suffix}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Normalized matrix with annotations
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1],
                xticklabels=["Forest","Urban","Open"],
                yticklabels=["Forest","Urban","Open"], 
                vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f"Confusion Matrix - Normalized\nTest F1={f1_score_val:.3f}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_mlp_training_curves(history, save_path):
    """
    Plot MLP training dynamics: macro F1 (train + val) and per-class validation F1.
    Uses the thesis color palette.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history["val_f1"]) + 1)
    
    # Left plot: Training and Validation Macro F1
    axes[0].plot(epochs, history["train_f1"], label="Train F1", color=COLORS["green"], linewidth=2)
    axes[0].plot(epochs, history["val_f1"], label="Val F1", color=COLORS["navy"], linewidth=2)
    axes[0].set_title("MLP: Training and Validation Macro-F1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.3, 1.0])
    
    # Right plot: Per-class Validation F1
    axes[1].plot(epochs, history["val_f1_forest"], label="Forest", color=HABITAT_COLORS["forest"], linewidth=2)
    axes[1].plot(epochs, history["val_f1_urban"], label="Urban", color=HABITAT_COLORS["urban"], linewidth=2)
    axes[1].plot(epochs, history["val_f1_open"], label="Open", color=HABITAT_COLORS["open"], linewidth=2)
    axes[1].set_title("Per-Class Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.4, 0.8])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {save_path}")


# ------- main ------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load data -----
    df = pd.read_parquet(MERGED_PARQUET)
    split_info = pd.read_csv(SPLIT_FILE)
    df = df.merge(split_info, on="recording_id")

    # ----- prepare data -----
    feat_cols = [c for c in df.columns if c.startswith("Feature_")]
    X = df[feat_cols].to_numpy(np.float32)

    # prepare soft and hard labels
    if all(c in df.columns for c in LABEL_COLS_4):
        raw = df[LABEL_COLS_4].to_numpy(np.float32)
        y_fuo = raw[:, :3]
        s = y_fuo.sum(axis=1, keepdims=True)
        s[s == 0] = 1e-8
        y_soft = y_fuo / s
    else:
        y_soft = df[LABEL_COLS_3].to_numpy(np.float32)
        y_soft = y_soft / (y_soft.sum(axis=1, keepdims=True) + 1e-8)
    y_hard = y_soft.argmax(axis=1)

    # recording ids and splits
    rid = df["recording_id"].to_numpy()
    split = df["split"].to_numpy()

    Xtr, ytr, ytr_h, rid_tr = X[split=="train"], y_soft[split=="train"], y_hard[split=="train"], rid[split=="train"]
    Xva, yva, yva_h, rid_va = X[split=="val"],   y_soft[split=="val"],   y_hard[split=="val"],   rid[split=="val"]
    Xte, yte, yte_h, rid_te = X[split=="test"],  y_soft[split=="test"],  y_hard[split=="test"],  rid[split=="test"]

    print(f"Train: {len(rid_tr):,} segments ({len(np.unique(rid_tr))} recordings)")
    print(f"Val: {len(rid_va):,} segments ({len(np.unique(rid_va))} recordings)")
    print(f"Test: {len(rid_te):,} segments ({len(np.unique(rid_te))} recordings)")

    # ------ class distribution analysis ------
    print("Class distribution analysis:")
    
    train_counts = analyze_class_distribution(y_soft, split=="train", "train")
    analyze_class_distribution(y_soft, split=="val", "val")
    analyze_class_distribution(y_soft, split=="test", "test")

    # ------ compute class weights ------
    class_weights = compute_class_weight('balanced', classes=np.arange(3), y=ytr_h)

    print("\nClass weights:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:8s}: {class_weights[i]:.3f}")
    weight_dict = {i: class_weights[i] for i in range(3)}

    # scaling for focal loss
    alpha_weights = torch.tensor([
        class_weights[0] * 1.0,  
        class_weights[1] * 2.0, # boost urban
        class_weights[2] * 0.8 # slightly reduce open
    ])
    print("\nFocal loss alpha weights:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:8s}: {alpha_weights[i]:.3f}")

    # ----- robust scaling -----
    scaler = RobustScaler().fit(Xtr)
    Xtr_sc = scaler.transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    Xte_sc = scaler.transform(Xte)

    # plot the embedding space
    plot_embedding_space(Xtr_sc[:5000], ytr_h[:5000], RESULTS_DIR / "embedding_space_pca.png")

    # Expand soft labels with class weights (TRAIN ONLY)
    Xtr_exp, ytr_exp, wtr_exp = expand_soft_labels(Xtr_sc, ytr, class_weights)
    print(f"\nExpanded training set: {len(Xtr_exp):,} samples")




    # -------- training logistic regression --------
    print("LOGISTIC REGRESSION")
    lr_model = LogisticRegression(max_iter=3000, C=0.1, class_weight=weight_dict, penalty="l2", solver="saga", multi_class="multinomial", n_jobs=-1, verbose=0, random_state=SEED)
    lr_model.fit(Xtr_exp, ytr_exp, sample_weight=wtr_exp)

    proba_lr_train = lr_model.predict_proba(Xtr_sc)
    proba_lr_val = lr_model.predict_proba(Xva_sc)
    proba_lr_test = lr_model.predict_proba(Xte_sc)

    # evaluation
    print("\nLogistic Regression results:")
    lr_train_acc, lr_train_f1, lr_train_kl, _, _, lr_train_f1_per = evaluate_split(proba_lr_train, ytr, rid_tr, "Train")
    lr_val_acc, lr_val_f1, lr_val_kl, _, _, lr_val_f1_per = evaluate_split(proba_lr_val, yva, rid_va, "Val")
    lr_test_acc, lr_test_f1, lr_test_kl, lr_cm, _, lr_test_f1_per = evaluate_split(proba_lr_test, yte, rid_te, "Test")


    # --------- training XGBoost ---------
    print("\nXGBOOST TRAINING")
    xgb_model = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    xgb_model.fit(Xtr_exp, ytr_exp, sample_weight=wtr_exp)

    proba_xgb_train = xgb_model.predict_proba(Xtr_sc)
    proba_xgb_val = xgb_model.predict_proba(Xva_sc)
    proba_xgb_test = xgb_model.predict_proba(Xte_sc)
    # evaluation
    print("\nXGBoost results:")
    xgb_train_acc, xgb_train_f1, xgb_train_kl, _, _, xgb_train_f1_per = evaluate_split(proba_xgb_train, ytr, rid_tr, "Train")
    xgb_val_acc, xgb_val_f1, xgb_val_kl, _, _, xgb_val_f1_per = evaluate_split(proba_xgb_val, yva, rid_va, "Val")
    xgb_test_acc, xgb_test_f1, xgb_test_kl, xgb_cm, _, xgb_test_f1_per = evaluate_split(proba_xgb_test, yte, rid_te, "Test")

    # =-------- MLP training ---------
    print("MLP TRAINING")

    ytr_smooth = ytr * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 3.0

    # data loader
    tr_ds = TensorDataset(torch.from_numpy(Xtr_sc.astype(np.float32)), torch.from_numpy(ytr_smooth.astype(np.float32)))
    tr_dl = DataLoader(tr_ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=2)

    mlp = MLPclassifier(in_dim=Xtr_sc.shape[1], hidden=(128, 64), out_dim=3, p=0.6).to(device)

    # optimizer, scheduler, criterion
    opt = torch.optim.AdamW(mlp.parameters(), lr=LR_MLP, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    criterion = WeightedFocalLoss(alpha=alpha_weights, gamma=2.0)

    best_state, best_f1, wait = None, -np.inf, 0
    history = {"train_loss": [], "train_f1": [],  "val_acc": [], "val_f1": [], "val_kl": [], "val_f1_forest": [], "val_f1_urban": [], "val_f1_open": []}

    # training loop
    for epoch in range(EPOCHS):
        mlp.train()
        epoch_loss = 0.0
        
        train_preds = []
        train_targets = []
        print("Epoch", epoch+1)

        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            train_targets.append(yb.detach().cpu().numpy())
            
            if MIXUP_ALPHA > 0: # apply mixup
                xb, yb = mixup_data(xb, yb, MIXUP_ALPHA)

            opt.zero_grad()
            logits = mlp(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
                train_preds.append(probs)
        
        # training F1 at recording level
        train_preds_all = np.vstack(train_preds)
        train_acc, train_f1_epoch, _, _ = evaluate_recording_level_mlp(mlp, Xtr_sc, ytr, rid_tr, device)

        # evaluate on validation set
        val_acc, val_f1, val_kl, val_f1_per_class = evaluate_recording_level_mlp(mlp, Xva_sc, yva, rid_va, device)
        scheduler.step(val_f1)
        
        # add to history
        history["train_loss"].append(epoch_loss / len(tr_dl))
        history["train_f1"].append(train_f1_epoch)  
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_kl"].append(val_kl)
        history["val_f1_forest"].append(val_f1_per_class[0])
        history["val_f1_urban"].append(val_f1_per_class[1])
        history["val_f1_open"].append(val_f1_per_class[2])

        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = mlp.state_dict()
            status = "New best"
        else:
            wait += 1
            status = f"Wait {wait}/{PATIENCE}"

        if epoch % 5 == 0 or wait == 0:  # print every 5 epochs or when new best
            print(f"Epoch {epoch+1:03d} | loss={history['train_loss'][-1]:.4f} | train_f1={train_f1_epoch:.3f} | "
                  f"val_f1={val_f1:.3f} (F={val_f1_per_class[0]:.2f}, U={val_f1_per_class[1]:.2f}, O={val_f1_per_class[2]:.2f}) | {status}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # ----- final evaluation -----
    mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        probs_mlp_train = torch.sigmoid(mlp(torch.from_numpy(Xtr_sc).to(device))).cpu().numpy()
        probs_mlp_val   = torch.sigmoid(mlp(torch.from_numpy(Xva_sc).to(device))).cpu().numpy()
        probs_mlp_test  = torch.sigmoid(mlp(torch.from_numpy(Xte_sc).to(device))).cpu().numpy()

    probs_mlp_train /= (probs_mlp_train.sum(axis=1, keepdims=True) + 1e-8)
    probs_mlp_val   /= (probs_mlp_val.sum(axis=1, keepdims=True) + 1e-8)
    probs_mlp_test  /= (probs_mlp_test.sum(axis=1, keepdims=True) + 1e-8)

    print("\nMLP results:")
    mlp_train_acc, mlp_train_f1, mlp_train_kl, _, _, mlp_train_f1_per = evaluate_split(probs_mlp_train, ytr, rid_tr, "Train")
    mlp_val_acc, mlp_val_f1, mlp_val_kl, _, _, mlp_val_f1_per = evaluate_split(probs_mlp_val, yva, rid_va, "Val")
    mlp_test_acc, mlp_test_f1, mlp_test_kl, mlp_cm, _, mlp_test_f1_per = evaluate_split(probs_mlp_test, yte, rid_te, "Test")

    # ------- ensemble --------
    print("ENSEMBLE TRAINING")
    
    # simple weighted average ensemble
    ensemble_train = 0.4 * proba_lr_train + 0.6 * probs_mlp_train
    ensemble_val = 0.4 * proba_lr_val + 0.6 * probs_mlp_val
    ensemble_test = 0.4 * proba_lr_test + 0.6 * probs_mlp_test

    print("\nEnsemble Results:")
    ens_train_acc, ens_train_f1, ens_train_kl, _, _, ens_train_f1_per = evaluate_split(ensemble_train, ytr, rid_tr, "Train")
    ens_val_acc, ens_val_f1, ens_val_kl, _, _, ens_val_f1_per = evaluate_split(ensemble_val, yva, rid_va, "Val")
    ens_test_acc, ens_test_f1, ens_test_kl, ens_cm, _, ens_test_f1_per = evaluate_split(ensemble_test, yte, rid_te, "Test")

    # ---------- visuals ---------
    # MLP training history
    plot_mlp_training_curves(history, RESULTS_DIR / "mlp_training_history.png")

    # Confusion matrices for all models
    plot_detailed_confusion_matrix(lr_cm, lr_test_f1, RESULTS_DIR / "confusion_matrix_logreg.png", "Logistic Regression")
    plot_detailed_confusion_matrix(xgb_cm, xgb_test_f1, RESULTS_DIR / "confusion_matrix_xgboost.png", "XGBoost")
    plot_detailed_confusion_matrix(mlp_cm, mlp_test_f1, RESULTS_DIR / "confusion_matrix_mlp.png", "MLP")
    plot_detailed_confusion_matrix(ens_cm, ens_test_f1, RESULTS_DIR / "confusion_matrix_ensemble.png", "Ensemble (LR+MLP)")
    
    # Bar chart comparing all models
    models = ['LogReg', 'XGBoost', 'MLP', 'Ensemble']
    test_f1s = [lr_test_f1, xgb_test_f1, mlp_test_f1, ens_test_f1]
    test_accs = [lr_test_acc, xgb_test_acc, mlp_test_acc, ens_test_acc]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 scores
    bar_colors = [COLORS["bluegrey"], COLORS["sage"], COLORS["green"], COLORS["gold"]]
    axes[0].bar(models, test_f1s, color=bar_colors, alpha=0.8)
    axes[0].set_ylabel('Macro F1 Score')
    axes[0].set_title('Test Set Performance - F1 Score')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(test_f1s):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Accuracy
    axes[1].bar(models, test_accs, color=bar_colors, alpha=0.8)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Set Performance - Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(test_accs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {RESULTS_DIR / 'model_comparison.png'}")
    
    # Per-class F1 comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    
    forest_f1s = [lr_test_f1_per[0], xgb_test_f1_per[0], mlp_test_f1_per[0], ens_test_f1_per[0]]
    urban_f1s = [lr_test_f1_per[1], xgb_test_f1_per[1], mlp_test_f1_per[1], ens_test_f1_per[1]]
    open_f1s = [lr_test_f1_per[2], xgb_test_f1_per[2], mlp_test_f1_per[2], ens_test_f1_per[2]]
    
    ax.bar(x - width, forest_f1s, width, label='Forest', color=HABITAT_COLORS["forest"], alpha=0.8)
    ax.bar(x, urban_f1s, width, label='Urban', color=HABITAT_COLORS["urban"], alpha=0.8)
    ax.bar(x + width, open_f1s, width, label='Open', color=HABITAT_COLORS["open"], alpha=0.8)
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_f1_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'per_class_f1_comparison.png'}")

    # ----- summary csv -----   
    summary = pd.DataFrame([
        {"model":"logreg", "split":"train", "acc":lr_train_acc, "f1":lr_train_f1, "kl":lr_train_kl, 
         "f1_forest":lr_train_f1_per[0], "f1_urban":lr_train_f1_per[1], "f1_open":lr_train_f1_per[2]},
        {"model":"logreg", "split":"val", "acc":lr_val_acc, "f1":lr_val_f1, "kl":lr_val_kl,
         "f1_forest":lr_val_f1_per[0], "f1_urban":lr_val_f1_per[1], "f1_open":lr_val_f1_per[2]},
        {"model":"logreg", "split":"test", "acc":lr_test_acc, "f1":lr_test_f1, "kl":lr_test_kl,
         "f1_forest":lr_test_f1_per[0], "f1_urban":lr_test_f1_per[1], "f1_open":lr_test_f1_per[2]},

        {"model":"xgboost", "split":"train", "acc":xgb_train_acc, "f1":xgb_train_f1, "kl":xgb_train_kl,
         "f1_forest":xgb_train_f1_per[0], "f1_urban":xgb_train_f1_per[1], "f1_open":xgb_train_f1_per[2]},
        {"model":"xgboost", "split":"val", "acc":xgb_val_acc, "f1":xgb_val_f1, "kl":xgb_val_kl,
         "f1_forest":xgb_val_f1_per[0], "f1_urban":xgb_val_f1_per[1], "f1_open":xgb_val_f1_per[2]},
        {"model":"xgboost", "split":"test", "acc":xgb_test_acc, "f1":xgb_test_f1, "kl":xgb_test_kl,
         "f1_forest":xgb_test_f1_per[0], "f1_urban":xgb_test_f1_per[1], "f1_open":xgb_test_f1_per[2]},

        {"model":"mlp", "split":"train", "acc":mlp_train_acc, "f1":mlp_train_f1, "kl":mlp_train_kl,
         "f1_forest":mlp_train_f1_per[0], "f1_urban":mlp_train_f1_per[1], "f1_open":mlp_train_f1_per[2]},
        {"model":"mlp", "split":"val", "acc":mlp_val_acc, "f1":mlp_val_f1, "kl":mlp_val_kl,
         "f1_forest":mlp_val_f1_per[0], "f1_urban":mlp_val_f1_per[1], "f1_open":mlp_val_f1_per[2]},
        {"model":"mlp", "split":"test", "acc":mlp_test_acc, "f1":mlp_test_f1, "kl":mlp_test_kl,
         "f1_forest":mlp_test_f1_per[0], "f1_urban":mlp_test_f1_per[1], "f1_open":mlp_test_f1_per[2]},

        {"model":"ensemble", "split":"train", "acc":ens_train_acc, "f1":ens_train_f1, "kl":ens_train_kl,
         "f1_forest":ens_train_f1_per[0], "f1_urban":ens_train_f1_per[1], "f1_open":ens_train_f1_per[2]},
        {"model":"ensemble", "split":"val", "acc":ens_val_acc, "f1":ens_val_f1, "kl":ens_val_kl,
         "f1_forest":ens_val_f1_per[0], "f1_urban":ens_val_f1_per[1], "f1_open":ens_val_f1_per[2]},
        {"model":"ensemble", "split":"test", "acc":ens_test_acc, "f1":ens_test_f1, "kl":ens_test_kl,
         "f1_forest":ens_test_f1_per[0], "f1_urban":ens_test_f1_per[1], "f1_open":ens_test_f1_per[2]},
    ])
    summary.to_csv(RESULTS_DIR / "results_summary_improved.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'results_summary_improved.csv'}")

    # final summary
    print("\nFINAL TEST RESULTS:")
    print(f"\n{'Model':<15} {'Acc':>6} {'F1':>6} {'KL':>6} | {'Forest':>6} {'Urban':>6} {'Open':>6}")
    print("-" * 70)
    print(f"{'LogReg':<15} {lr_test_acc:>6.3f} {lr_test_f1:>6.3f} {lr_test_kl:>6.3f} | "
          f"{lr_test_f1_per[0]:>6.3f} {lr_test_f1_per[1]:>6.3f} {lr_test_f1_per[2]:>6.3f}")
    print(f"{'XGBoost':<15} {xgb_test_acc:>6.3f} {xgb_test_f1:>6.3f} {xgb_test_kl:>6.3f} | "
          f"{xgb_test_f1_per[0]:>6.3f} {xgb_test_f1_per[1]:>6.3f} {xgb_test_f1_per[2]:>6.3f}")
    print(f"{'MLP':<15} {mlp_test_acc:>6.3f} {mlp_test_f1:>6.3f} {mlp_test_kl:>6.3f} | "
          f"{mlp_test_f1_per[0]:>6.3f} {mlp_test_f1_per[1]:>6.3f} {mlp_test_f1_per[2]:>6.3f}")
    print(f"{'Ensemble':<15} {ens_test_acc:>6.3f} {ens_test_f1:>6.3f} {ens_test_kl:>6.3f} | "
          f"{ens_test_f1_per[0]:>6.3f} {ens_test_f1_per[1]:>6.3f} {ens_test_f1_per[2]:>6.3f}")


if __name__ == "__main__":
    main()